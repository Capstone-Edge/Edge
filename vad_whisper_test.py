#!/usr/bin/env python3
"""
Jetson Orin Nano - WebRTC VAD + Whisper STT 테스트

목적:
- Vosk 제거
- Silero VAD / torchaudio 제거
- 버튼 / Enter 입력 제거
- 마이크를 계속 듣다가, 말소리가 감지된 구간만 녹음
- 말이 끝나면 Whisper로 STT 변환
- 결과를 화면에 출력
- "엣지/에지/엔지/edge"가 포함되면 COMMAND로 출력
- 예약어가 없으면 RAW만 보여주고 무시

실행:
    python3 vad_whisper_test.py

종료:
    Ctrl + C
"""

import os
import sys
import time
import queue
import tempfile
import collections

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import webrtcvad
import whisper


# ============================================================
# 설정
# ============================================================
WHISPER_MODEL = "small"      # tiny / base / small 중 선택 가능
SAMPLE_RATE = 16000         # WebRTC VAD는 8000/16000/32000/48000 지원
CHANNELS = 1
FRAME_MS = 30               # WebRTC VAD는 10/20/30ms만 지원
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

VAD_AGGRESSIVENESS = 2      # 0~3. 높을수록 엄격하게 음성만 잡음
START_TRIGGER_FRAMES = 3    # 연속 N프레임 음성 감지 시 녹음 시작
END_SILENCE_FRAMES = 25     # 연속 N프레임 무음 감지 시 녹음 종료. 25*30ms=0.75초
PRE_ROLL_FRAMES = 10        # 말 시작 직전 오디오도 약간 보존. 10*30ms=0.3초

MIN_RECORD_SEC = 0.5        # 너무 짧은 소리는 버림
MAX_RECORD_SEC = 10.0       # 한 번에 최대 녹음 시간

WAKE_WORDS = ["엣지", "에지", "엔지", "edge", "edgy"]


# ============================================================
# 오디오 큐
# ============================================================
audio_q = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    """마이크 콜백: int16 오디오를 큐에 넣음"""
    if status:
        print(f"[AUDIO] {status}", file=sys.stderr)
    audio_q.put(bytes(indata))


# ============================================================
# 텍스트 처리
# ============================================================
def normalize_text(text: str) -> str:
    return text.lower().replace(" ", "")


def extract_command(text: str):
    """
    Whisper 결과에서 예약어가 있는지 확인.
    있으면 (wake_word, command) 반환.
    없으면 (None, None) 반환.
    """
    original = text.strip()
    normalized = normalize_text(original)

    for word in WAKE_WORDS:
        nw = normalize_text(word)
        if nw in normalized:
            command = original

            # 원문에서 가능한 예약어 표현 제거
            for w in WAKE_WORDS:
                command = command.replace(w, "")
                command = command.replace(w.upper(), "")
                command = command.replace(w.capitalize(), "")

            command = command.strip()
            command = command.strip(" ,.!?~")
            return word, command

    return None, None


# ============================================================
# Whisper
# ============================================================
def transcribe_whisper(model, audio_float32: np.ndarray) -> str:
    """
    float32 오디오(-1.0~1.0)를 임시 wav로 저장한 뒤 Whisper STT 수행.
    """
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        audio_int16 = np.clip(audio_float32 * 32767, -32768, 32767).astype(np.int16)
        wav.write(tmp_path, SAMPLE_RATE, audio_int16)

        print("[STT] Whisper 변환 중...")
        result = model.transcribe(
            tmp_path,
            language="ko",
            fp16=False,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
        )

        return result.get("text", "").strip()

    except Exception as e:
        print(f"[ERROR] Whisper 실패: {e}", file=sys.stderr)
        return ""

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ============================================================
# VAD + 녹음 루프
# ============================================================
def bytes_to_float32_audio(audio_bytes: bytes) -> np.ndarray:
    """int16 bytes → float32 numpy"""
    audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
    return audio_int16.astype(np.float32) / 32768.0


def main():
    print("=" * 60)
    print("  WebRTC VAD + Whisper STT 테스트")
    print(f"  Whisper 모델: {WHISPER_MODEL}")
    print(f"  VAD 민감도: {VAD_AGGRESSIVENESS} / 3")
    print(f"  프레임: {FRAME_MS}ms")
    print(f"  예약어 후보: {', '.join(WAKE_WORDS)}")
    print("  종료: Ctrl + C")
    print("=" * 60)

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

    print("[READY] 계속 듣는 중입니다. 말하면 자동으로 녹음합니다.")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=CHANNELS,
        callback=audio_callback,
    ):
        while True:
            frame = audio_q.get()

            # WebRTC VAD는 정확히 10/20/30ms 길이의 mono int16 PCM만 허용
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
                    print("[VOICE] 말소리 종료 → Whisper 인식 시작")

                    audio_bytes = b"".join(speech_frames)
                    audio_float32 = bytes_to_float32_audio(audio_bytes)
                    duration = len(audio_float32) / SAMPLE_RATE

                    # 상태 초기화 먼저 해둠
                    triggered = False
                    speech_frames = []
                    pre_roll.clear()
                    silence_count = 0
                    voiced_count = 0
                    record_start_time = None

                    if duration < MIN_RECORD_SEC:
                        print(f"[SKIP] 녹음이 너무 짧음: {duration:.2f}초")
                        print("[READY] 계속 듣는 중입니다.")
                        continue

                    print(f"[AUDIO] 녹음 길이: {duration:.2f}초")
                    text = transcribe_whisper(model, audio_float32)

                    if not text:
                        print('[RAW] ""')
                        print("[RESULT] 인식된 문장이 없습니다.")
                        print("[READY] 계속 듣는 중입니다.")
                        continue

                    print(f'[RAW] "{text}"')

                    wake, command = extract_command(text)

                    if wake:
                        print(f'[WAKE] 예약어 감지: "{wake}"')
                        if command:
                            print(f'[COMMAND] "{command}"')
                        else:
                            print("[COMMAND] 예약어는 감지됐지만 명령어가 비어 있습니다.")
                    else:
                        print("[IGNORE] 예약어 없음. 무시합니다.")

                    print("[READY] 계속 듣는 중입니다.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] 종료")
        sys.exit(0)
