#!/usr/bin/env python3
"""
TTS 테스트 프로그램

- 사용자가 터미널에 입력한 문장을 그대로 한국어 TTS로 변환
- 블루투스 스피커로 출력
- 종료하려면 q, quit, exit 입력
"""

import os
import subprocess
import tempfile
from gtts import gTTS


# 블루투스 스피커 PulseAudio sink 이름
BT_SINK = "bluez_sink.CB_81_E4_47_7B_56.handsfree_head_unit"


def speak(text: str):
    text = text.strip()
    if not text:
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        mp3_path = fp.name

    try:
        print(f"[TTS] 변환 중: {text}")

        tts = gTTS(text=text, lang="ko")
        tts.save(mp3_path)

        print("[TTS] 재생 중...")

        env = os.environ.copy()
        env["PULSE_SINK"] = BT_SINK

        subprocess.run(
            ["mpg123", "-q", mp3_path],
            env=env,
            check=False,
        )

    finally:
        if os.path.exists(mp3_path):
            os.remove(mp3_path)


def main():
    print("===================================")
    print(" Jetson Orin Nano TTS Test")
    print(" 입력한 문장을 블루투스 스피커로 출력합니다.")
    print(" 종료: q / quit / exit")
    print("===================================")

    while True:
        text = input("\n말할 문장 입력 > ").strip()

        if text.lower() in ["q", "quit", "exit"]:
            print("종료합니다.")
            break

        speak(text)


if __name__ == "__main__":
    main()