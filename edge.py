#!/usr/bin/env python3
"""
Jetson Orin Nano - 음성비서 + 예약어 + Backend process API 연동

기능:
- 예약어: "나비"
- 일반 상태에서는 "나비 + 명령" 형식일 때만 백엔드로 전송
- 재질문 대기 상태에서는 예약어 없이도 사용자 답변을 백엔드로 전송
- 말하면 자동 녹음
- Whisper로 STT
- Backend FastAPI /api/v1/commands/process 로 명령 전송
- Backend 응답을 gTTS로 블루투스 스피커 출력
- TTS 출력 중/직후에는 마이크 입력 무시
- VAD 녹음 파일을 last_vad_record.wav로 저장해서 디버깅 가능

실행 예:
BACKEND_URL=http://100.104.72.38:8000 INPUT_DEVICE=24 WHISPER_MODEL=small python3 edge.py

빠른 테스트:
BACKEND_URL=http://100.104.72.38:8000 INPUT_DEVICE=24 WHISPER_MODEL=base python3 edge.py

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

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import webrtcvad
import whisper
from gtts import gTTS

import config
from backend_client import BackendClient


# ============================================================
# 전역 상태
# ============================================================

audio_q = queue.Queue(maxsize=100)

is_tts_playing = False
ignore_audio_until = 0.0


class EdgeRuntime:
    """
    Edge 실행 상태 관리.

    여기서 관리하는 상태:
    - current_session_id
    - waiting_for_clarification

    즉, 백엔드가 재질문 중이면 session_id를 유지하고,
    executed / cancelled / expired면 세션을 초기화한다.
    """

    def __init__(self):
        self.backend_client = BackendClient(
            backend_url=config.BACKEND_URL,
            process_api_path=config.PROCESS_API_PATH,
            client_id=config.CLIENT_ID,
            device_id=config.DEVICE_ID,
            timeout_sec=config.HTTP_TIMEOUT_SEC,
        )

        self.current_session_id: str | None = None
        self.waiting_for_clarification = False

    def update_session_state_from_result(self, result: dict | None) -> None:
        """
        Backend /api/v1/commands/process 응답 status에 따라
        Edge의 current_session_id를 저장하거나 비운다.

        요구사항:
        1. waiting_clarification이면 session_id 저장
        2. executed / cancelled / expired이면 session_id 초기화
        """

        if result is None:
            print("[SESSION WARN] 백엔드 result 없음 → 기존 세션 상태 유지")
            return

        status = result.get("status")

        if status == "waiting_clarification":
            backend_session_id = result.get("session_id")

            if backend_session_id:
                self.current_session_id = backend_session_id
                self.waiting_for_clarification = True
                print(f"[SESSION] waiting_clarification → session_id 저장: {self.current_session_id}")
            else:
                self.current_session_id = None
                self.waiting_for_clarification = False
                print(
                    "[SESSION WARN] waiting_clarification 응답인데 session_id가 없습니다.",
                    file=sys.stderr,
                )

            return

        if status in ("executed", "cancelled", "expired"):
            print(f"[SESSION] status={status} → session_id 초기화")
            self.current_session_id = None
            self.waiting_for_clarification = False
            return

        # 백엔드가 다른 status를 주는 경우 안전하게 세션을 비운다.
        print(f"[SESSION WARN] 알 수 없는 status={status} → session_id 초기화")
        self.current_session_id = None
        self.waiting_for_clarification = False

    def backend_ai_reply(self, user_text: str) -> str:
        """
        엣지 STT 결과를 백엔드 /api/v1/commands/process 로 보내고,
        TTS로 읽을 response_text를 반환한다.
        """

        reply_text, result = self.backend_client.request_ai_reply(
            user_text=user_text,
            session_id=self.current_session_id,
        )

        self.update_session_state_from_result(result)

        return reply_text

    def ready_message(self) -> str:
        if self.waiting_for_clarification:
            return "[READY] 백엔드 재질문 대기 중입니다. 예약어 없이 답변하세요."

        return "[READY] 계속 듣는 중입니다. '나비 + 명령'으로 말하세요."


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

    for wake_word in sorted(config.WAKE_WORDS, key=len, reverse=True):
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
    "나비 조명 켜줘" -> True, "조명 켜줘", "나비"
    "조명 켜줘" -> False, "", ""
    """

    original = text.strip()
    normalized = normalize_text(original)

    for wake_word in config.WAKE_WORDS:
        normalized_wake_word = normalize_text(wake_word)

        if normalized_wake_word in normalized:
            command = remove_wake_words_from_text(original)
            return True, command, wake_word

    return False, "", ""


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
            ["paplay", f"--device={config.PULSE_SINK}", "/tmp/tts_warmup.wav"],
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
        env["PULSE_SINK"] = config.PULSE_SINK

        subprocess.run(
            ["mpg123", "-q", "--pitch", str(config.TTS_SPEED), mp3_path],
            env=env,
            check=False,
        )

    except Exception as e:
        print(f"[ERROR] TTS 실패: {e}", file=sys.stderr)

    finally:
        is_tts_playing = False
        ignore_audio_until = time.time() + config.IGNORE_AFTER_TTS_SEC
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

        if config.SAVE_LAST_VAD_RECORD:
            tmp_path = config.LAST_VAD_RECORD_PATH
            should_delete = False
        else:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                should_delete = True

        wav.write(tmp_path, config.SAMPLE_RATE, audio_int16)

        if config.SAVE_LAST_VAD_RECORD:
            print(f"[DEBUG] VAD 녹음 저장: {config.LAST_VAD_RECORD_PATH}")

        print("[STT] Whisper 변환 중...")

        result = model.transcribe(
            tmp_path,
            language="ko",
            task="transcribe",
            fp16=False,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
            logprob_threshold=-0.5,
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


def print_startup_info(runtime: EdgeRuntime):
    print("=" * 60)
    print(" Jetson Orin Nano 음성비서 + 예약어 + Backend process API 연동")
    print(f" Whisper 모델: {config.WHISPER_MODEL}")
    print(f" Input Device: {config.INPUT_DEVICE}")
    print(f" Sample Rate: {config.SAMPLE_RATE}")
    print(f" Pulse Sink: {config.PULSE_SINK}")
    print(f" VAD 민감도: {config.VAD_AGGRESSIVENESS}")
    print(f" TTS 후 마이크 무시: {config.IGNORE_AFTER_TTS_SEC}초")
    print(" 예약어 기능: ON")
    print(f" 예약어 후보: {', '.join(config.WAKE_WORDS)}")
    print(" 사용 예: 나비 조명 켜줘")
    print(" 백엔드 연동: ON")
    print(f" Backend URL: {config.BACKEND_URL}")
    print(f" Client ID: {config.CLIENT_ID}")
    print(f" Device ID: {config.DEVICE_ID}")
    print(f" Process API: {config.PROCESS_API_PATH}")
    print(" 종료: Ctrl + C")
    print("=" * 60)


# ============================================================
# 메인 루프
# ============================================================

def main():
    global ignore_audio_until

    runtime = EdgeRuntime()

    print_startup_info(runtime)
    print_audio_devices()

    print(f"[INIT] Whisper {config.WHISPER_MODEL} 모델 로딩 중...")
    model = whisper.load_model(config.WHISPER_MODEL)
    print("[INIT] Whisper 로딩 완료")

    vad = webrtcvad.Vad(config.VAD_AGGRESSIVENESS)

    pre_roll = collections.deque(maxlen=config.PRE_ROLL_FRAMES)
    speech_frames = []

    triggered = False
    voiced_count = 0
    silence_count = 0
    record_start_time = None

    clear_audio_queue()
    ignore_audio_until = time.time() + 0.5

    print(runtime.ready_message())

    stream_kwargs = dict(
        samplerate=config.SAMPLE_RATE,
        blocksize=config.FRAME_SAMPLES,
        dtype="int16",
        channels=config.CHANNELS,
        callback=audio_callback,
    )

    if config.INPUT_DEVICE is not None:
        stream_kwargs["device"] = config.INPUT_DEVICE

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

            expected_bytes = config.FRAME_SAMPLES * 2

            if len(frame) != expected_bytes:
                continue

            is_speech = vad.is_speech(frame, config.SAMPLE_RATE)

            if not triggered:
                pre_roll.append(frame)

                if is_speech:
                    voiced_count += 1
                else:
                    voiced_count = 0

                if voiced_count >= config.START_TRIGGER_FRAMES:
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

                should_stop_by_silence = silence_count >= config.END_SILENCE_FRAMES
                should_stop_by_max_time = elapsed >= config.MAX_RECORD_SEC

                if should_stop_by_silence or should_stop_by_max_time:
                    print("[VOICE] 말소리 종료 → STT 시작")

                    audio_bytes = b"".join(speech_frames)
                    audio_float32 = bytes_to_float32_audio(audio_bytes)
                    duration = len(audio_float32) / config.SAMPLE_RATE

                    # 녹음 상태 초기화
                    triggered = False
                    speech_frames = []
                    pre_roll.clear()
                    silence_count = 0
                    voiced_count = 0
                    record_start_time = None

                    if duration < config.MIN_RECORD_SEC:
                        print(f"[SKIP] 녹음이 너무 짧음: {duration:.2f}초")
                        print(runtime.ready_message())
                        continue

                    print(f"[AUDIO] 녹음 길이: {duration:.2f}초")

                    user_text = transcribe_whisper(model, audio_float32)

                    if not user_text:
                        print('[USER] ""')
                        print("[IGNORE] 인식된 문장이 없습니다.")
                        print(runtime.ready_message())
                        continue

                    print(f'[USER] "{user_text}"')

                    # ----------------------------------------------------
                    # 핵심 분기:
                    # 1. 재질문 대기 중이면 예약어 없이도 백엔드로 전송
                    # 2. 일반 상태면 예약어가 있어야 백엔드로 전송
                    # ----------------------------------------------------
                    if runtime.waiting_for_clarification:
                        command = user_text.strip()

                        print(f'[CLARIFY_ANSWER] "{command}"')

                        if not command:
                            reply = "다시 말씀해 주세요."
                        else:
                            reply = runtime.backend_ai_reply(command)

                    else:
                        has_wake, command, detected_wake = extract_command_with_wake_word(user_text)

                        if not has_wake:
                            print("[IGNORE] 예약어 없음. 무시합니다.")
                            print(runtime.ready_message())
                            continue

                        print(f'[WAKE] 예약어 감지: "{detected_wake}"')

                        if not command:
                            reply = "네, 말씀해 주세요."
                        else:
                            print(f'[COMMAND] "{command}"')
                            reply = runtime.backend_ai_reply(command)

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

                    print(runtime.ready_message())


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n[EXIT] 종료")
        sys.exit(0)
