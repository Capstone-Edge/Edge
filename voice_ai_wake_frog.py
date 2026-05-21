#!/usr/bin/env python3
"""
Jetson Orin Nano - 음성비서 로컬 테스트 + 예약어 기능

기능:
- 예약어: "개구리"
- "개구리 + 명령" 형식일 때만 응답
- 예약어가 없으면 무시
- 말하면 자동 녹음
- Whisper로 STT
- 임시 AI 응답 생성
- gTTS로 블루투스 스피커 출력
- TTS 출력 중/직후에는 마이크 입력 무시
- 스피커 음성이 다시 마이크로 들어가 무한 반복되는 문제 방지
- VAD 녹음 파일을 last_vad_record.wav로 저장해서 디버깅 가능

실행:
    INPUT_DEVICE=24 WHISPER_MODEL=small python3 voice_ai_wake_frog.py

빠른 테스트:
    INPUT_DEVICE=24 WHISPER_MODEL=base python3 voice_ai_wake_frog.py

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

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import webrtcvad
import whisper
from gtts import gTTS


# ============================================================
# 설정
# ============================================================

# tiny / base / small 중 선택 가능
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

# sounddevice 입력 장치 번호.
# 현재 사용자 환경 기준 USB PnP Audio Device는 24번.
# 확인 명령:
#   python3 -c "import sounddevice as sd; print(sd.query_devices())"
INPUT_DEVICE_ENV = os.getenv("INPUT_DEVICE", "").strip()
INPUT_DEVICE = int(INPUT_DEVICE_ENV) if INPUT_DEVICE_ENV else 24

# USB PnP Audio Device가 16kHz를 직접 지원하지 않으므로 48kHz 사용
# WebRTC VAD는 8000/16000/32000/48000 지원
SAMPLE_RATE = 48000

CHANNELS = 1
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

# VAD 설정
VAD_AGGRESSIVENESS = 0       # 0~3. 높을수록 엄격. 인식 안정성 우선이면 0 추천
START_TRIGGER_FRAMES = 2     # 2 * 30ms = 60ms 연속 음성이면 녹음 시작
END_SILENCE_FRAMES = 80      # 80 * 30ms = 2.4초 침묵이면 녹음 종료
PRE_ROLL_FRAMES = 50         # 50 * 30ms = 1.5초 말 시작 전 보존

MIN_RECORD_SEC = 1.5
MAX_RECORD_SEC = 15.0

# TTS 출력 후 이 시간 동안 마이크 입력 무시
# 블루투스 스피커 잔향/지연 때문에 1.5~2.5초 권장
IGNORE_AFTER_TTS_SEC = 2.0

# VAD가 실제로 Whisper에 넣은 녹음 파일 저장 여부
SAVE_LAST_VAD_RECORD = True
LAST_VAD_RECORD_PATH = "last_vad_record.wav"

# 블루투스 스피커 PulseAudio sink
# 확인:
#   pactl list short sinks
PULSE_SINK = os.getenv(
    "PULSE_SINK",
    "bluez_sink.CB_81_E4_47_7B_56.handsfree_head_unit"
)

# 예약어 후보
# Whisper 오인식 가능성을 고려해 후보를 넓게 둠
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
# 전역 상태
# ============================================================

# 큐가 무한히 쌓이지 않도록 제한
audio_q = queue.Queue(maxsize=100)

is_tts_playing = False
ignore_audio_until = 0.0


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
        # 오래된 작업 처리 중 입력이 밀리면 버림
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
        .replace("　", "")
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
    "개구리 조명 켜줘" → True, "조명 켜줘", "개구리"
    "조명 켜줘" → False, "", ""
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
# 임시 AI 응답 함수
# ============================================================

def simple_ai_reply(user_text: str) -> str:
    """
    백엔드 구현 전 임시 AI 응답 함수.
    지금은 규칙 기반으로 동작.
    나중에 이 함수만 LLM API나 백엔드 호출로 교체하면 됨.
    """
    text = user_text.strip()
    compact = text.replace(" ", "")

    if not text:
        return "네, 말씀해 주세요."

    # 인사
    if any(word in compact for word in ["안녕", "하이", "반가워"]):
        return "안녕하세요. 음성비서 테스트를 시작합니다."

    # 상태 확인
    if any(word in compact for word in ["잘돼", "작동", "테스트"]):
        return "네, 현재 음성 인식과 음성 출력 테스트가 동작 중입니다."

    # 조명 제어 흉내
    if "불" in compact or "조명" in compact:
        if "켜" in compact:
            return "조명을 켰습니다."
        if "꺼" in compact:
            return "조명을 껐습니다."
        return "조명을 어떻게 할까요?"

    # 에어컨 제어 흉내
    if "에어컨" in compact or "냉방" in compact:
        if "켜" in compact:
            return "에어컨을 켰습니다. 원하는 온도를 말씀해 주세요."
        if "꺼" in compact:
            return "에어컨을 껐습니다."
        if "도" in compact:
            return "알겠습니다. 말씀하신 온도로 에어컨을 설정하겠습니다."
        return "에어컨을 어떻게 제어할까요?"

    # 청소기 제어 흉내
    if "청소" in compact or "청소기" in compact:
        if "시작" in compact or "켜" in compact:
            return "청소를 시작하겠습니다."
        if "멈춰" in compact or "중지" in compact or "꺼" in compact:
            return "청소를 중지하겠습니다."
        return "청소기를 어떻게 할까요?"

    # TV/미디어 제어 흉내
    if "티비" in compact or "tv" in compact.lower() or "영화" in compact or "보고싶" in compact:
        if "매드맥스" in compact or "매드 맥스" in text:
            return "매드맥스를 재생하겠습니다."
        return "원하시는 콘텐츠를 재생하겠습니다."

    # 시간 예약 흉내
    if "뒤" in compact or "분후" in compact or "시간후" in compact or "예약" in compact:
        return "예약 명령으로 인식했습니다. 백엔드가 연결되면 실제 예약 기능과 연동할 수 있습니다."

    # 일반 질문 흉내
    if "몇시" in compact or "시간" in compact:
        return "현재 시간 확인 기능은 아직 연결되지 않았습니다."

    if "날씨" in compact:
        return "날씨 조회 기능은 아직 연결되지 않았습니다."

    candidates = [
        f"제가 들은 명령은, {text}, 입니다.",
        f"{text}라고 말씀하셨습니다.",
        "좋습니다. 해당 명령을 정상적으로 인식했습니다.",
        "현재는 백엔드 없이 임시 응답 모드로 동작 중입니다.",
    ]
    return random.choice(candidates)


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
        audio_int16 = np.clip(audio_float32 * 32767, -32768, 32767).astype(np.int16)

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


def reset_runtime_state(pre_roll):
    """
    VAD 상태 초기화용.
    main 내부 지역 변수들은 직접 초기화해야 하므로,
    여기서는 pre_roll과 큐만 처리.
    """
    pre_roll.clear()
    clear_audio_queue()


# ============================================================
# 메인 루프
# ============================================================

def main():
    global ignore_audio_until

    print("=" * 60)
    print("  Jetson Orin Nano 음성비서 테스트 + 예약어")
    print(f"  Whisper 모델: {WHISPER_MODEL}")
    print(f"  Input Device: {INPUT_DEVICE}")
    print(f"  Sample Rate: {SAMPLE_RATE}")
    print(f"  Pulse Sink: {PULSE_SINK}")
    print(f"  VAD 민감도: {VAD_AGGRESSIVENESS}")
    print(f"  TTS 후 마이크 무시: {IGNORE_AFTER_TTS_SEC}초")
    print("  예약어 기능: ON")
    print(f"  예약어 후보: {', '.join(WAKE_WORDS)}")
    print("  사용 예: 개구리 조명 켜줘")
    print("  백엔드 연동: OFF")
    print("  AI 응답: simple_ai_reply() 임시 규칙 기반")
    print("  종료: Ctrl + C")
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

    print("[READY] 계속 듣는 중입니다. '개구리 + 명령'으로 말하세요.")

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
                        print("[READY] 계속 듣는 중입니다. '개구리 + 명령'으로 말하세요.")
                        continue

                    print(f"[AUDIO] 녹음 길이: {duration:.2f}초")

                    user_text = transcribe_whisper(model, audio_float32)

                    if not user_text:
                        print('[USER] ""')
                        print("[IGNORE] 인식된 문장이 없습니다.")
                        print("[READY] 계속 듣는 중입니다. '개구리 + 명령'으로 말하세요.")
                        continue

                    print(f'[USER] "{user_text}"')

                    has_wake, command, detected_wake = extract_command_with_wake_word(user_text)

                    if not has_wake:
                        print("[IGNORE] 예약어 없음. 무시합니다.")
                        print("[READY] 계속 듣는 중입니다. '개구리 + 명령'으로 말하세요.")
                        continue

                    print(f'[WAKE] 예약어 감지: "{detected_wake}"')

                    if not command:
                        reply = "네, 말씀해 주세요."
                    else:
                        print(f'[COMMAND] "{command}"')
                        reply = simple_ai_reply(command)

                    print(f'[AI] "{reply}"')
                    speak_tts(reply)

                    # TTS 후 VAD 상태 재초기화
                    triggered = False
                    speech_frames = []
                    pre_roll.clear()
                    silence_count = 0
                    voiced_count = 0
                    record_start_time = None
                    clear_audio_queue()

                    print("[READY] 계속 듣는 중입니다. '개구리 + 명령'으로 말하세요.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] 종료")
        sys.exit(0)
