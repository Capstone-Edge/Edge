# config.py

import os


# ============================================================
# Whisper / STT 설정
# ============================================================

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


# ============================================================
# 오디오 입력 설정
# ============================================================

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


# ============================================================
# VAD 설정
# ============================================================

VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "0"))
START_TRIGGER_FRAMES = int(os.getenv("START_TRIGGER_FRAMES", "2"))
END_SILENCE_FRAMES = int(os.getenv("END_SILENCE_FRAMES", "80"))
PRE_ROLL_FRAMES = int(os.getenv("PRE_ROLL_FRAMES", "50"))
MIN_RECORD_SEC = float(os.getenv("MIN_RECORD_SEC", "1.5"))
MAX_RECORD_SEC = float(os.getenv("MAX_RECORD_SEC", "15.0"))


# ============================================================
# TTS 설정
# ============================================================

# TTS 출력 후 이 시간 동안 마이크 입력 무시
IGNORE_AFTER_TTS_SEC = float(os.getenv("IGNORE_AFTER_TTS_SEC", "2.0"))

# 블루투스 스피커 PulseAudio sink
# 확인:
# pactl list short sinks
PULSE_SINK = os.getenv(
    "PULSE_SINK",
    "bluez_sink.CB_81_E4_47_7B_56.handsfree_head_unit",
)

# TTS 재생 속도
TTS_SPEED = float(os.getenv("TTS_SPEED", "0.15"))


# ============================================================
# 디버그 녹음 저장 설정
# ============================================================

SAVE_LAST_VAD_RECORD = os.getenv("SAVE_LAST_VAD_RECORD", "1") == "1"
LAST_VAD_RECORD_PATH = os.getenv("LAST_VAD_RECORD_PATH", "last_vad_record.wav")


# ============================================================
# 예약어 설정
# ============================================================

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
    "데구리",
]


# ============================================================
# Backend 설정
# ============================================================

BACKEND_URL = os.getenv("BACKEND_URL", "http://100.78.243.115:8000").rstrip("/")

CLIENT_ID = os.getenv("EDGE_CLIENT_ID", "edge-pi-01")
DEVICE_ID = os.getenv("EDGE_DEVICE_ID", "edge-pi-01")

PROCESS_API_PATH = "/api/v1/commands/process"

HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "10"))