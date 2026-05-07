import Jetson.GPIO as GPIO
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import os
import time

# =========================
# 설정
# =========================
MODEL_NAME = "small"
SR = 16000
BTN_PIN = 7  # BOARD 물리 핀 7번 = GPIO09

# 버튼 상태
# 외부 풀업 회로 기준:
# 안 누름 = HIGH(1)
# 누름 = LOW(0)

print("Whisper 모델 로딩 중.")
model = whisper.load_model(MODEL_NAME)
print("Whisper 모델 로딩 완료!")

GPIO.setmode(GPIO.BOARD)

# Jetson.GPIO는 pull_up_down을 무시하므로 외부 10k 풀업 저항을 사용해야 함
GPIO.setup(BTN_PIN, GPIO.IN)


def record_while_pressed():
    frames = []

    try:
        stream = sd.InputStream(
            samplerate=SR,
            channels=1,
            dtype="float32",
            blocksize=1024
        )

        stream.start()
        print("🎙️ 녹음 중... 버튼을 떼면 종료됩니다.")

        while GPIO.input(BTN_PIN) == GPIO.LOW:
            data, overflowed = stream.read(1024)

            if overflowed:
                print("⚠️ 오디오 입력 버퍼 overflow 발생")

            frames.append(data.copy())

        stream.stop()
        stream.close()

    except Exception as e:
        print(f"마이크 녹음 오류: {e}")
        return None

    if not frames:
        return None

    return np.concatenate(frames).flatten()


def transcribe(audio):
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name

        # float32 [-1.0, 1.0] → int16 변환
        audio_int16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        wav.write(tmp_path, SR, audio_int16)

        print("🧠 Whisper 변환 중...")
        result = model.transcribe(tmp_path, language="ko", fp16=False)

        return result["text"].strip()

    except Exception as e:
        print(f"STT 변환 오류: {e}")
        return ""

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


print("=== 버튼 누르면 녹음 ===")
print("배선: 3.3V - 10kΩ - GPIO09(핀7), GPIO09(핀7) - 버튼 - GND")
print("안 누름: 1 / 누름: 0")
print("종료: Ctrl + C")

try:
    while True:
        # 버튼 누름 감지: HIGH(1) -> LOW(0)
        GPIO.wait_for_edge(BTN_PIN, GPIO.FALLING, bouncetime=200)

        # 짧은 디바운스
        time.sleep(0.05)

        # 혹시 바운스 때문에 이미 버튼이 떼어진 경우 무시
        if GPIO.input(BTN_PIN) != GPIO.LOW:
            continue

        audio = record_while_pressed()

        if audio is None:
            print("녹음된 오디오가 없습니다.")
            continue

        if len(audio) <= SR * 0.3:
            print("녹음 시간이 너무 짧습니다.")
            continue

        text = transcribe(audio)

        if text:
            print(f"📝 {text}")
        else:
            print("인식된 텍스트가 없습니다.")

        print("\n=== 다음 입력 대기 중 ===")

except KeyboardInterrupt:
    print("\n프로그램 종료")

finally:
    GPIO.cleanup()
    print("GPIO cleanup 완료")