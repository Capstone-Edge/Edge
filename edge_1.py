#!/usr/bin/env python3
"""
Jetson Orin Nano - 음성비서 + 예약어 + Backend 연동

기능:
- 예약어: "개구리"
- 일반 상태에서는 "개구리 + 명령" 형식일 때만 백엔드로 전송
- 재질문 대기 상태에서는 예약어 없이도 사용자 답변을 백엔드로 전송
- 말하면 자동 녹음
- Whisper로 STT
- Backend FastAPI로 명령 전송
- Backend 응답을 gTTS로 블루투스 스피커 출력
- TTS 출력 중/직후에는 마이크 입력 무시
- VAD 녹음 파일을 last_vad_record.wav로 저장해서 디버깅 가능

실행 예:
BACKEND_URL=http://100.104.72.38:8000 INPUT_DEVICE=24 WHISPER_MODEL=small python3 voice_ai_wake_frog.py

빠른 테스트:
BACKEND_URL=http://100.104.72.38:8000 INPUT_DEVICE=24 WHISPER_MODEL=base python3 voice_ai_wake_frog.py

종료:
Ctrl + C
"""

import os
import sys
import time
import queue
import tempfile
import collections
import subprocess
import random
import uuid
from typing import Any

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import webrtcvad
import whisper
import requests
from gtts import gTTS


# ============================================================
# 기본 설정
# ============================================================

# tiny / base / small 중 선택 가능
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# sounddevice 입력 장치 번호.
# 확인 명령:
# python3 -c "import sounddevice as sd; print(sd.query_devices())"
INPUT_DEVICE_ENV = os.getenv("INPUT_DEVICE", "").strip()
INPUT_DEVICE = int(INPUT_DEVICE_ENV) if INPUT_DEVICE_ENV else 24

# USB PnP Audio Device가 16kHz를 직접 지원하지 않으므로 48kHz 사용
# WebRTC VAD는 8000/16000/32000/48000 지원
SAMPLE_RATE = 48000
CHANNELS = 1
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

# VAD 설정
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "0"))
START_TRIGGER_FRAMES = int(os.getenv("START_TRIGGER_FRAMES", "2"))
END_SILENCE_FRAMES = int(os.getenv("END_SILENCE_FRAMES", "80"))
PRE_ROLL_FRAMES = int(os.getenv("PRE_ROLL_FRAMES", "50"))
MIN_RECORD_SEC = float(os.getenv("MIN_RECORD_SEC", "1.5"))
MAX_RECORD_SEC = float(os.getenv("MAX_RECORD_SEC", "15.0"))

# TTS 출력 후 이 시간 동안 마이크 입력 무시
IGNORE_AFTER_TTS_SEC = float(os.getenv("IGNORE_AFTER_TTS_SEC", "2.0"))

# VAD가 실제로 Whisper에 넣은 녹음 파일 저장 여부
SAVE_LAST_VAD_RECORD = os.getenv("SAVE_LAST_VAD_RECORD", "1") == "1"
LAST_VAD_RECORD_PATH = os.getenv("LAST_VAD_RECORD_PATH", "last_vad_record.wav")

# 블루투스 스피커 PulseAudio sink
# 확인:
# pactl list short sinks
PULSE_SINK = os.getenv(
    "PULSE_SINK",
    "bluez_sink.CB_81_E4_47_7B_56.handsfree_head_unit",
)

# 예약어 후보
WAKE_WORDS = [
    "개구리",
    "깨구리",
    "개굴이",
    "개구리야",
    "깨구리야",
    "개굴아",
    "개구라",
    "깨굴이",
]


# ============================================================
# Backend 연동 설정
# ============================================================

BACKEND_URL = os.getenv("BACKEND_URL", "http://100.104.72.38:8000").rstrip("/")
EDGE_DEVICE_ID = os.getenv("EDGE_DEVICE_ID", "edge-desktop")
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "10"))

# parse/clarify 결과 commands가 있으면 execute까지 자동 호출
AUTO_EXECUTE_COMMANDS = os.getenv("AUTO_EXECUTE_COMMANDS", "1") == "1"

# 백엔드 연결 실패 시 simple_ai_reply로 임시 응답할지 여부
USE_FALLBACK_REPLY = os.getenv("USE_FALLBACK_REPLY", "0") == "1"


# ============================================================
# 전역 상태
# ============================================================

audio_q = queue.Queue(maxsize=100)

is_tts_playing = False
ignore_audio_until = 0.0

# 백엔드 대화 상태
current_session_id: str | None = None
waiting_for_clarification = False
current_clarification_turn = 0


# ============================================================
# 오디오 큐 / 콜백
# ============================================================

def clear_audio_queue():
    """마이크 큐에 쌓인 오래된 오디오 제거"""
    try:
        while True:
            audio_q.get_nowait()
    except queue.Empty:
        pass


def audio_callback(indata, frames, time_info, status):
    """
    마이크 콜백.

    핵심:
    - TTS 출력 중에는 입력을 버림
    - TTS 직후 잔향 구간도 입력을 버림
    - 큐가 가득 차면 새 입력을 버려 overflow 누적을 줄임
    """
    global is_tts_playing, ignore_audio_until

    if status:
        print(f"[AUDIO] {status}", file=sys.stderr)

    now = time.time()

    if is_tts_playing or now < ignore_audio_until:
        return

    try:
        audio_q.put_nowait(bytes(indata))
    except queue.Full:
        pass


# ============================================================
# 예약어 처리
# ============================================================

def normalize_text(text: str) -> str:
    """예약어 비교용 정규화"""
    return (
        text.lower()
        .replace(" ", "")
        .replace(",", "")
        .replace(".", "")
        .replace("!", "")
        .replace("?", "")
        .replace("~", "")
        .replace(":", "")
        .replace(";", "")
    )


def remove_wake_words_from_text(text: str) -> str:
    """원문에서 예약어 후보를 제거"""
    command = text.strip()

    for w in sorted(WAKE_WORDS, key=len, reverse=True):
        command = command.replace(w, "")
        command = command.replace(w.upper(), "")
        command = command.replace(w.capitalize(), "")

    command = command.strip()
    command = command.strip(" ,.!?~:;")
    return command


def extract_command_with_wake_word(text: str):
    """
    예약어가 있으면 (True, command, wake_word) 반환.
    예약어가 없으면 (False, "", "") 반환.

    예:
    "개구리 조명 켜줘" -> True, "조명 켜줘", "개구리"
    "조명 켜줘" -> False, "", ""
    """
    original = text.strip()
    normalized = normalize_text(original)

    for wake in WAKE_WORDS:
        nwake = normalize_text(wake)
        if nwake in normalized:
            command = remove_wake_words_from_text(original)
            return True, command, wake

    return False, "", ""


# ============================================================
# 임시 fallback 응답 함수
# ============================================================

def simple_ai_reply(user_text: str) -> str:
    """
    백엔드 연결 실패 시 쓸 수 있는 fallback 응답.
    기본 실행에서는 USE_FALLBACK_REPLY=0 이므로 거의 사용하지 않음.
    """
    text = user_text.strip()
    compact = text.replace(" ", "")

    if not text:
        return "네, 말씀해 주세요."

    if any(word in compact for word in ["안녕", "하이", "반가워"]):
        return "안녕하세요. 음성비서 테스트를 시작합니다."

    if any(word in compact for word in ["잘돼", "작동", "테스트"]):
        return "네, 현재 음성 인식과 음성 출력 테스트가 동작 중입니다."

    if "불" in compact or "조명" in compact:
        if "켜" in compact:
            return "조명을 켰습니다."
        if "꺼" in compact:
            return "조명을 껐습니다."
        return "조명을 어떻게 할까요?"

    if "에어컨" in compact or "냉방" in compact:
        if "켜" in compact:
            return "에어컨을 켰습니다. 원하는 온도를 말씀해 주세요."
        if "꺼" in compact:
            return "에어컨을 껐습니다."
        if "도" in compact:
            return "알겠습니다. 말씀하신 온도로 에어컨을 설정하겠습니다."
        return "에어컨을 어떻게 제어할까요?"

    if "청소" in compact or "청소기" in compact:
        if "시작" in compact or "켜" in compact:
            return "청소를 시작하겠습니다."
        if "멈춰" in compact or "중지" in compact or "꺼" in compact:
            return "청소를 중지하겠습니다."
        return "청소기를 어떻게 할까요?"

    if (
        "티비" in compact
        or "tv" in compact.lower()
        or "영화" in compact
        or "보고싶" in compact
    ):
        if "매드맥스" in compact or "매드 맥스" in text:
            return "매드맥스를 재생하겠습니다."
        return "원하시는 콘텐츠를 재생하겠습니다."

    if "몇시" in compact or "시간" in compact:
        return "현재 시간 확인 기능은 아직 연결되지 않았습니다."

    if "날씨" in compact:
        return "날씨 조회 기능은 아직 연결되지 않았습니다."

    candidates = [
        f"제가 들은 명령은, {text}, 입니다.",
        f"{text}라고 말씀하셨습니다.",
        "좋습니다. 해당 명령을 정상적으로 인식했습니다.",
        "현재는 백엔드 연결 실패 시 임시 응답 모드로 동작 중입니다.",
    ]
    return random.choice(candidates)


# ============================================================
# Backend API 연동
# ============================================================

def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """
    백엔드 FastAPI 서버에 JSON POST 요청.
    """
    url = f"{BACKEND_URL}{path}"

    print(f"[HTTP] POST {url}")
    print(f"[HTTP] payload={payload}")

    response = requests.post(
        url,
        json=payload,
        timeout=HTTP_TIMEOUT_SEC,
    )

    print(f"[HTTP] status={response.status_code}")

    if response.status_code >= 400:
        print(f"[HTTP ERROR] {response.text}")
        response.raise_for_status()

    data = response.json()
    print(f"[HTTP] response={data}")
    return data


def send_parse_to_backend(command_text: str) -> dict[str, Any]:
    """
    최초 사용자 명령을 /api/v1/commands/parse 로 전송.
    """
    global current_session_id

    if current_session_id is None:
        current_session_id = f"edge-{uuid.uuid4().hex[:12]}"

    payload = {
        "session_id": current_session_id,
        "device_id": EDGE_DEVICE_ID,
        "stt_text": command_text,
        "source": "edge",
    }

    return post_json("/api/v1/commands/parse", payload)


def send_clarify_to_backend(answer_text: str) -> dict[str, Any]:
    """
    백엔드 재질문에 대한 사용자 답변을 /api/v1/dialogues/clarify 로 전송.
    """
    global current_session_id, current_clarification_turn

    if current_session_id is None:
        raise RuntimeError("clarify 요청을 보낼 session_id가 없습니다.")

    payload = {
        "device_id": EDGE_DEVICE_ID,
        "session_id": current_session_id,
        "user_answer": answer_text,
        "clarification_turn": current_clarification_turn,
    }

    return post_json("/api/v1/dialogues/clarify", payload)


def execute_backend_commands(
    parse_or_clarify_result: dict[str, Any],
    raw_user_input: str,
) -> dict[str, Any]:
    """
    parse/clarify 결과에 commands가 있으면 /api/v1/commands/execute 호출.
    """
    commands = parse_or_clarify_result.get("commands") or []

    if not commands:
        return parse_or_clarify_result

    session_id = parse_or_clarify_result.get("session_id") or current_session_id
    if not session_id:
        session_id = f"edge-{uuid.uuid4().hex[:12]}"

    intent = parse_or_clarify_result.get("intent") or "device_control"
    response_text = (
        parse_or_clarify_result.get("response_text")
        or "명령을 실행했습니다."
    )

    payload = {
        "session_id": session_id,
        "raw_user_input": raw_user_input,
        "intent": intent,
        "commands": commands,
        "response_text": response_text,
    }

    return post_json("/api/v1/commands/execute", payload)


def get_tts_text_from_result(result: dict[str, Any]) -> str:
    """
    백엔드 응답 dict에서 TTS로 읽을 문장을 뽑는다.
    """
    return (
        result.get("response_text")
        or result.get("clarification_question")
        or result.get("message")
        or "백엔드 응답을 받았습니다."
    )


def backend_ai_reply(user_text: str) -> str:
    """
    엣지 STT 결과를 백엔드로 보내고, TTS로 읽을 response_text를 반환.

    일반 명령:
    - /api/v1/commands/parse

    재질문 답변:
    - /api/v1/dialogues/clarify

    명령 실행:
    - commands가 있으면 /api/v1/commands/execute 자동 호출
    """
    global current_session_id
    global waiting_for_clarification
    global current_clarification_turn

    try:
        if waiting_for_clarification:
            result = send_clarify_to_backend(user_text)
        else:
            result = send_parse_to_backend(user_text)

        current_session_id = result.get("session_id") or current_session_id
        current_clarification_turn = result.get(
            "clarification_turn",
            current_clarification_turn,
        )

        # 재질문 필요
        if result.get("clarification_needed") is True:
            waiting_for_clarification = True
            return get_tts_text_from_result(result)

        # 재질문 종료 또는 일반 명령 완료
        waiting_for_clarification = False
        current_clarification_turn = 0

        # commands가 있으면 execute 호출
        if AUTO_EXECUTE_COMMANDS and result.get("commands"):
            execute_result = execute_backend_commands(result, user_text)
            return (
                execute_result.get("response_text")
                or result.get("response_text")
                or "명령을 실행했습니다."
            )

        return get_tts_text_from_result(result)

    except requests.exceptions.ConnectionError:
        print("[BACKEND ERROR] 백엔드 서버에 연결할 수 없습니다.", file=sys.stderr)
        if USE_FALLBACK_REPLY:
            return simple_ai_reply(user_text)
        return "백엔드 서버에 연결할 수 없습니다."

    except requests.exceptions.Timeout:
        print("[BACKEND ERROR] 백엔드 응답 시간이 초과되었습니다.", file=sys.stderr)
        if USE_FALLBACK_REPLY:
            return simple_ai_reply(user_text)
        return "백엔드 응답 시간이 초과되었습니다."

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        print(f"[BACKEND ERROR] HTTP {status_code}", file=sys.stderr)
        if USE_FALLBACK_REPLY:
            return simple_ai_reply(user_text)
        return f"백엔드 요청 중 오류가 발생했습니다. 상태 코드 {status_code}."

    except Exception as e:
        print(f"[BACKEND ERROR] {e}", file=sys.stderr)
        if USE_FALLBACK_REPLY:
            return simple_ai_reply(user_text)
        return "백엔드 처리 중 오류가 발생했습니다."


# ============================================================
# TTS
# ============================================================

def warmup_speaker():
    """블루투스 스피커가 SUSPENDED 상태에서 깨어나도록 짧은 무음 재생"""
    try:
        subprocess.run(
            [
                "python3",
                "-c",
                (
                    "import wave, numpy as np;"
                    "sr=48000;"
                    "silence=np.zeros(int(sr*0.8), dtype=np.int16);"
                    "wf=wave.open('/tmp/tts_warmup.wav','w');"
                    "wf.setnchannels(1);"
                    "wf.setsampwidth(2);"
                    "wf.setframerate(sr);"
                    "wf.writeframes(silence.tobytes());"
                    "wf.close()"
                ),
            ],
            check=False,
        )

        subprocess.run(
            ["paplay", f"--device={PULSE_SINK}", "/tmp/tts_warmup.wav"],
            check=False,
        )

        time.sleep(0.15)

    except Exception as e:
        print(f"[WARN] 스피커 워밍업 실패: {e}", file=sys.stderr)


def speak_tts(text: str):
    """
    gTTS로 한국어 음성을 생성하고 PulseAudio sink로 재생.

    중요:
    - TTS 시작 전 큐 비움
    - TTS 중 마이크 입력 무시
    - TTS 종료 후 일정 시간 마이크 입력 무시
    - TTS 중 쌓인 오디오 큐 제거
    """
    global is_tts_playing, ignore_audio_until

    text = text.strip()
    if not text:
        return

    mp3_path = None

    try:
        clear_audio_queue()
        is_tts_playing = True

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            mp3_path = tmp.name

        print(f"[TTS] {text}")

        tts = gTTS(text=text, lang="ko")
        tts.save(mp3_path)

        warmup_speaker()

        env = os.environ.copy()
        env["PULSE_SINK"] = PULSE_SINK

        subprocess.run(
            ["mpg123", "-q", mp3_path],
            env=env,
            check=False,
        )

    except Exception as e:
        print(f"[ERROR] TTS 실패: {e}", file=sys.stderr)

    finally:
        is_tts_playing = False
        ignore_audio_until = time.time() + IGNORE_AFTER_TTS_SEC
        clear_audio_queue()

        if mp3_path and os.path.exists(mp3_path):
            os.remove(mp3_path)


# ============================================================
# Whisper STT
# ============================================================

def transcribe_whisper(model, audio_float32: np.ndarray) -> str:
    """
    float32 오디오를 wav로 저장한 뒤 Whisper STT 수행.
    SAVE_LAST_VAD_RECORD=True면 last_vad_record.wav를 남김.
    """
    tmp_path = None
    should_delete = True

    try:
        audio_int16 = np.clip(
            audio_float32 * 32767,
            -32768,
            32767,
        ).astype(np.int16)

        if SAVE_LAST_VAD_RECORD:
            tmp_path = LAST_VAD_RECORD_PATH
            should_delete = False
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            should_delete = True

        wav.write(tmp_path, SAMPLE_RATE, audio_int16)

        if SAVE_LAST_VAD_RECORD:
            print(f"[DEBUG] VAD 녹음 저장: {LAST_VAD_RECORD_PATH}")

        print("[STT] Whisper 변환 중...")

        result = model.transcribe(
            tmp_path,
            language="ko",
            task="transcribe",
            fp16=False,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.3,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
        )

        return result.get("text", "").strip()

    except Exception as e:
        print(f"[ERROR] Whisper 실패: {e}", file=sys.stderr)
        return ""

    finally:
        if should_delete and tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def bytes_to_float32_audio(audio_bytes: bytes) -> np.ndarray:
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


# ============================================================
# 디버그 출력
# ============================================================

def print_audio_devices():
    print("\n[DEVICE] sounddevice 장치 목록")
    try:
        print(sd.query_devices())
    except Exception as e:
        print(f"[WARN] 장치 목록 확인 실패: {e}", file=sys.stderr)


def reset_vad_state(pre_roll):
    """VAD 상태 초기화 보조"""
    pre_roll.clear()
    clear_audio_queue()


def ready_message() -> str:
    if waiting_for_clarification:
        return "[READY] 백엔드 재질문 대기 중입니다. 예약어 없이 답변하세요."
    return "[READY] 계속 듣는 중입니다. '개구리 + 명령'으로 말하세요."


# ============================================================
# 메인 루프
# ============================================================

def main():
    global ignore_audio_until
    global waiting_for_clarification

    print("=" * 60)
    print(" Jetson Orin Nano 음성비서 + 예약어 + Backend 연동")
    print(f" Whisper 모델: {WHISPER_MODEL}")
    print(f" Input Device: {INPUT_DEVICE}")
    print(f" Sample Rate: {SAMPLE_RATE}")
    print(f" Pulse Sink: {PULSE_SINK}")
    print(f" VAD 민감도: {VAD_AGGRESSIVENESS}")
    print(f" TTS 후 마이크 무시: {IGNORE_AFTER_TTS_SEC}초")
    print(" 예약어 기능: ON")
    print(f" 예약어 후보: {', '.join(WAKE_WORDS)}")
    print(" 사용 예: 개구리 조명 켜줘")
    print(" 백엔드 연동: ON")
    print(f" Backend URL: {BACKEND_URL}")
    print(f" Edge Device ID: {EDGE_DEVICE_ID}")
    print(f" Auto Execute Commands: {AUTO_EXECUTE_COMMANDS}")
    print(f" Fallback Reply: {USE_FALLBACK_REPLY}")
    print(" 종료: Ctrl + C")
    print("=" * 60)

    print_audio_devices()

    print(f"[INIT] Whisper {WHISPER_MODEL} 모델 로딩 중...")
    model = whisper.load_model(WHISPER_MODEL)
    print("[INIT] Whisper 로딩 완료")

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    pre_roll = collections.deque(maxlen=PRE_ROLL_FRAMES)
    speech_frames = []
    triggered = False
    voiced_count = 0
    silence_count = 0
    record_start_time = None

    clear_audio_queue()
    ignore_audio_until = time.time() + 0.5

    print(ready_message())

    stream_kwargs = dict(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=CHANNELS,
        callback=audio_callback,
    )

    if INPUT_DEVICE is not None:
        stream_kwargs["device"] = INPUT_DEVICE

    with sd.RawInputStream(**stream_kwargs):
        while True:
            frame = audio_q.get()

            # TTS 직후 잔향 구간이면 큐 비우고 무시
            if time.time() < ignore_audio_until:
                clear_audio_queue()
                triggered = False
                speech_frames = []
                pre_roll.clear()
                silence_count = 0
                voiced_count = 0
                record_start_time = None
                continue

            expected_bytes = FRAME_SAMPLES * 2
            if len(frame) != expected_bytes:
                continue

            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if not triggered:
                pre_roll.append(frame)

                if is_speech:
                    voiced_count += 1
                else:
                    voiced_count = 0

                if voiced_count >= START_TRIGGER_FRAMES:
                    triggered = True
                    record_start_time = time.time()
                    silence_count = 0
                    speech_frames = list(pre_roll)

                    print("\n[VOICE] 말소리 감지 → 녹음 시작")
                    voiced_count = 0

            else:
                speech_frames.append(frame)

                if is_speech:
                    silence_count = 0
                else:
                    silence_count += 1

                elapsed = time.time() - record_start_time

                should_stop_by_silence = silence_count >= END_SILENCE_FRAMES
                should_stop_by_max_time = elapsed >= MAX_RECORD_SEC

                if should_stop_by_silence or should_stop_by_max_time:
                    print("[VOICE] 말소리 종료 → STT 시작")

                    audio_bytes = b"".join(speech_frames)
                    audio_float32 = bytes_to_float32_audio(audio_bytes)
                    duration = len(audio_float32) / SAMPLE_RATE

                    # 녹음 상태 초기화
                    triggered = False
                    speech_frames = []
                    pre_roll.clear()
                    silence_count = 0
                    voiced_count = 0
                    record_start_time = None

                    if duration < MIN_RECORD_SEC:
                        print(f"[SKIP] 녹음이 너무 짧음: {duration:.2f}초")
                        print(ready_message())
                        continue

                    print(f"[AUDIO] 녹음 길이: {duration:.2f}초")

                    user_text = transcribe_whisper(model, audio_float32)

                    if not user_text:
                        print('[USER] ""')
                        print("[IGNORE] 인식된 문장이 없습니다.")
                        print(ready_message())
                        continue

                    print(f'[USER] "{user_text}"')

                    # ----------------------------------------------------
                    # 핵심 분기:
                    # 1. 재질문 대기 중이면 예약어 없이도 백엔드로 전송
                    # 2. 일반 상태면 예약어가 있어야 백엔드로 전송
                    # ----------------------------------------------------

                    if waiting_for_clarification:
                        command = user_text.strip()
                        print(f'[CLARIFY_ANSWER] "{command}"')

                        if not command:
                            reply = "다시 말씀해 주세요."
                        else:
                            reply = backend_ai_reply(command)

                    else:
                        has_wake, command, detected_wake = extract_command_with_wake_word(user_text)

                        if not has_wake:
                            print("[IGNORE] 예약어 없음. 무시합니다.")
                            print(ready_message())
                            continue

                        print(f'[WAKE] 예약어 감지: "{detected_wake}"')

                        if not command:
                            reply = "네, 말씀해 주세요."
                        else:
                            print(f'[COMMAND] "{command}"')
                            reply = backend_ai_reply(command)

                    print(f'[BACKEND_REPLY] "{reply}"')
                    speak_tts(reply)

                    # TTS 후 VAD 상태 재초기화
                    triggered = False
                    speech_frames = []
                    pre_roll.clear()
                    silence_count = 0
                    voiced_count = 0
                    record_start_time = None
                    clear_audio_queue()

                    print(ready_message())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] 종료")
        sys.exit(0)