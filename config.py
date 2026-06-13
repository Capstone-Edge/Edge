import os


# Whisper / STT
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")


# Audio input
# 장치 확인: python3 -c "import sounddevice as sd; print(sd.query_devices())"
INPUT_DEVICE_ENV = os.getenv("INPUT_DEVICE", "").strip()
INPUT_DEVICE = int(INPUT_DEVICE_ENV) if INPUT_DEVICE_ENV else 24

SAMPLE_RATE = 48000
CHANNELS = 1
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)


# VAD
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "3"))
START_TRIGGER_FRAMES = int(os.getenv("START_TRIGGER_FRAMES", "5"))
END_SILENCE_FRAMES = int(os.getenv("END_SILENCE_FRAMES", "50"))
PRE_ROLL_FRAMES = int(os.getenv("PRE_ROLL_FRAMES", "10"))
MIN_RECORD_SEC = float(os.getenv("MIN_RECORD_SEC", "0.8"))
MAX_RECORD_SEC = float(os.getenv("MAX_RECORD_SEC", "10.0"))


# TTS
IGNORE_AFTER_TTS_SEC = float(os.getenv("IGNORE_AFTER_TTS_SEC", "2.0"))

# sink 확인: pactl list short sinks
PULSE_SINK = os.getenv(
    "PULSE_SINK",
    "bluez_sink.CB_81_E4_47_7B_56.handsfree_head_unit",
)

TTS_SPEED = float(os.getenv("TTS_SPEED", "0.15"))


# Debug recording
SAVE_LAST_VAD_RECORD = os.getenv("SAVE_LAST_VAD_RECORD", "1") == "1"
LAST_VAD_RECORD_PATH = os.getenv("LAST_VAD_RECORD_PATH", "last_vad_record.wav")


# Wake words
WAKE_WORDS = [
    "나비",
    "나비야",
    "나비아",
    "나비여",
    "나비요",
    "나비어",
    "나비와",
    "나미",
    "나미야",
    "남이",
    "남이야",
    "라비",
    "라비야",
    "다비",
    "다비야",
    "답이",
    "답이야",
    "바비",
    "바비야",
    "랍이",
    "랍이야",
    "마비",
    "마비야",
]


# Backend
BACKEND_URL = os.getenv("BACKEND_URL", "http://100.78.243.115:8000").rstrip("/")
CLIENT_ID = os.getenv("EDGE_CLIENT_ID", "edge-pi-01")
DEVICE_ID = os.getenv("EDGE_DEVICE_ID", "edge-pi-01")
PROCESS_API_PATH = "/api/v1/commands/process"
HTTP_TIMEOUT_SEC = float(os.getenv("HTTP_TIMEOUT_SEC", "10"))
