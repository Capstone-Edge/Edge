#!/usr/bin/env python3
"""
Jetson Orin Nano - 음성비서 + 예약어 + Backend process API 연동

기능:
- 예약어: "개구리"
- 일반 상태에서는 "개구리 + 명령" 형식일 때만 백엔드로 전송
- 재질문 대기 상태에서는 예약어 없이도 사용자 답변을 백엔드로 전송
- 말하면 자동 녹음
- Whisper로 STT
- Backend FastAPI /api/v1/commands/process 로 명령 전송
- Backend 응답을 gTTS로 블루투스 스피커 출력
- TTS 출력 중/직후에는 마이크 입력 무시
- VAD 녹음 파일을 last_vad_record.wav로 저장해서 디버깅 가능

실행 예:
BACKEND_URL=http://100.104.72.38:8000 INPUT_DEVICE=24 WHISPER_MODEL=small python3 edge_1.py

빠른 테스트:
BACKEND_URL=http://100.104.72.38:8000 INPUT_DEVICE=24 WHISPER_MODEL=base python3 edge_1.py

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

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# sounddevice 입력 장치 번호
# 확인 명령:
# python3 -c "import sounddevice as sd; print(sd.query_devices())"
INPUT_DEVICE_ENV = os.getenv("INPUT_DEVICE", "").strip()
INPUT_DEVICE = int(INPUT_DEVICE_ENV) if INPUT_DEVICE_ENV else 24

# USB PnP Audio Device가 16kHz를 직접 지원하지 않으므로 48kHz 사용
# WebRTC VAD는 8000 / 16000 / 32000 / 48000 지원
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
    "개고리",
    "메구리",
    "데구리"
]


# ============================================================
# Backend 연동 설정
# ============================================================

BACKEND_URL = os.getenv("BACKEND_URL", "http://100.104.72.38:8000").rstrip("/")

CLIENT_ID = os.getenv("EDGE_CLIENT_ID", "edge-pi-01")
DEVICE_ID = os.getenv("EDGE_DEVICE_ID", "edge-pi-01")

HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "10"))


# ============================================================
# 전역 상태
# ============================================================

audio_q = queue.Queue(maxsize=100)

is_tts_playing = False
ignore_audio_until = 0.0

# Backend 대화 세션 상태
current_session_id: str | None = None
waiting_for_clarification = False


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
    global is_tts_playing
    global ignore_audio_until

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

    for wake_word in sorted(WAKE_WORDS, key=len, reverse=True):
        command = command.replace(wake_word, "")
        command = command.replace(wake_word.upper(), "")
        command = command.replace(wake_word.capitalize(), "")

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

    for wake_word in WAKE_WORDS:
        normalized_wake_word = normalize_text(wake_word)

        if normalized_wake_word in normalized:
            command = remove_wake_words_from_text(original)
            return True, command, wake_word

    return False, "", ""


# ============================================================
# Backend API 연동
# ============================================================

def post_json(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    """백엔드 FastAPI 서버에 JSON POST 요청"""
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
        print(f"[HTTP ERROR] {response.text}", file=sys.stderr)
        response.raise_for_status()

    data = response.json()
    print(f"[HTTP] response={data}")

    return data


def get_tts_text_from_result(result: dict[str, Any]) -> str:
    """백엔드 응답 dict에서 TTS로 읽을 문장을 뽑는다."""
    return (
        result.get("response_text")
        or result.get("clarification_question")
        or result.get("message")
        or "백엔드 응답을 받았습니다."
    )


def update_session_state_from_result(result: dict[str, Any]) -> None:
    """
    Backend /api/v1/commands/process 응답 status에 따라
    Edge의 current_session_id를 저장하거나 비운다.

    요구사항:
    1. waiting_clarification이면 session_id 저장
    2. executed / cancelled / expired이면 session_id 초기화
    """
    global current_session_id
    global waiting_for_clarification

    status = result.get("status")

    if status == "waiting_clarification":
        backend_session_id = result.get("session_id")

        if backend_session_id:
            current_session_id = backend_session_id
            waiting_for_clarification = True
            print(f"[SESSION] waiting_clarification → session_id 저장: {current_session_id}")
        else:
            current_session_id = None
            waiting_for_clarification = False
            print(
                "[SESSION WARN] waiting_clarification 응답인데 session_id가 없습니다.",
                file=sys.stderr,
            )

        return

    if status in ("executed", "cancelled", "expired"):
        print(f"[SESSION] status={status} → session_id 초기화")
        current_session_id = None
        waiting_for_clarification = False
        return

    # 백엔드가 다른 status를 주는 경우 안전하게 세션을 비운다.
    print(f"[SESSION WARN] 알 수 없는 status={status} → session_id 초기화")
    current_session_id = None
    waiting_for_clarification = False


def send_stt_text_to_backend(stt_text: str) -> dict[str, Any]:
    """
    STT 결과를 Backend의 단일 process API로 전송한다.

    규칙:
    - client_id는 항상 고정해서 보낸다.
    - device_id도 고정해서 보낸다.
    - session_id는 저장된 값이 있으면 포함하고, 없으면 None으로 보낸다.
    - 백엔드 응답 status에 따라 current_session_id를 갱신한다.
    """
    payload = {
        "client_id": CLIENT_ID,
        "device_id": DEVICE_ID,
        "session_id": current_session_id,
        "stt_text": stt_text,
        "source": "edge",
    }

    result = post_json("/api/v1/commands/process", payload)
    update_session_state_from_result(result)

    return result


def backend_ai_reply(user_text: str) -> str:
    """
    엣지 STT 결과를 백엔드 /api/v1/commands/process 로 보내고,
    TTS로 읽을 response_text를 반환한다.
    """
    try:
        result = send_stt_text_to_backend(user_text)
        return get_tts_text_from_result(result)

    except requests.exceptions.ConnectionError:
        print("[BACKEND ERROR] 백엔드 서버에 연결할 수 없습니다.", file=sys.stderr)
        return "백엔드 서버에 연결할 수 없습니다."

    except requests.exceptions.Timeout:
        print("[BACKEND ERROR] 백엔드 응답 시간이 초과되었습니다.", file=sys.stderr)
        return "백엔드 응답 시간이 초과되었습니다."

    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response is not None else "unknown"
        print(f"[BACKEND ERROR] HTTP {status_code}", file=sys.stderr)
        return f"백엔드 요청 중 오류가 발생했습니다. 상태 코드 {status_code}."

    except Exception as e:
        print(f"[BACKEND ERROR] {e}", file=sys.stderr)
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
    global is_tts_playing
    global ignore_audio_until

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

        # TTS 재생 속도
        TTS_SPEED = float(os.getenv("TTS_SPEED", "0.15"))

        subprocess.run(
            ["mpg123", "-q", "--pitch", str(TTS_SPEED), mp3_path],
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


def ready_message() -> str:
    if waiting_for_clarification:
        return "[READY] 백엔드 재질문 대기 중입니다. 예약어 없이 답변하세요."

    return "[READY] 계속 듣는 중입니다. '개구리 + 명령'으로 말하세요."


# ============================================================
# 메인 루프
# ============================================================

def main():
    global ignore_audio_until

    print("=" * 60)
    print(" Jetson Orin Nano 음성비서 + 예약어 + Backend process API 연동")
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
    print(f" Client ID: {CLIENT_ID}")
    print(f" Device ID: {DEVICE_ID}")
    print(" Process API: /api/v1/commands/process")
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