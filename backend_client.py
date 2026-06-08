# backend_client.py

import sys
from typing import Any

import requests


class BackendClient:
    """
    Backend FastAPI /api/v1/commands/process 연동 전담 클래스.

    역할:
    - process API POST 요청
    - response_text / clarification_question / message 중 TTS 문장 추출
    - HTTP 예외를 사용자에게 읽어줄 문장으로 변환
    """

    def __init__(
        self,
        backend_url: str,
        process_api_path: str,
        client_id: str,
        device_id: str,
        timeout_sec: float,
    ):
        self.backend_url = backend_url.rstrip("/")
        self.process_api_path = process_api_path
        self.client_id = client_id
        self.device_id = device_id
        self.timeout_sec = timeout_sec

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.backend_url}{path}"

        print(f"[HTTP] POST {url}")
        print(f"[HTTP] payload={payload}")

        response = requests.post(
            url,
            json=payload,
            timeout=self.timeout_sec,
        )

        print(f"[HTTP] status={response.status_code}")

        if response.status_code >= 400:
            print(f"[HTTP ERROR] {response.text}", file=sys.stderr)
            response.raise_for_status()

        data = response.json()
        print(f"[HTTP] response={data}")

        return data

    def send_process(
        self,
        stt_text: str,
        session_id: str | None,
    ) -> dict[str, Any]:
        """
        STT 결과를 Backend의 단일 process API로 전송한다.

        규칙:
        - client_id는 항상 고정해서 보낸다.
        - device_id도 고정해서 보낸다.
        - session_id는 저장된 값이 있으면 포함하고, 없으면 None으로 보낸다.
        - stt_text는 실제 사용자 명령/답변 문장이다.
        """

        payload = {
            "client_id": self.client_id,
            "device_id": self.device_id,
            "session_id": session_id,
            "stt_text": stt_text,
            "source": "edge",
        }

        return self.post_json(self.process_api_path, payload)

    @staticmethod
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

    def request_ai_reply(
        self,
        user_text: str,
        session_id: str | None,
    ) -> tuple[str, dict[str, Any] | None]:
        """
        백엔드로 user_text를 보내고,
        TTS로 읽을 문장과 원본 result를 함께 반환한다.

        반환:
        - reply_text
        - result

        오류가 나면 result는 None.
        """

        try:
            result = self.send_process(
                stt_text=user_text,
                session_id=session_id,
            )
            reply_text = self.get_tts_text_from_result(result)
            return reply_text, result

        except requests.exceptions.ConnectionError:
            print("[BACKEND ERROR] 백엔드 서버에 연결할 수 없습니다.", file=sys.stderr)
            return "백엔드 서버에 연결할 수 없습니다.", None

        except requests.exceptions.Timeout:
            print("[BACKEND ERROR] 백엔드 응답 시간이 초과되었습니다.", file=sys.stderr)
            return "백엔드 응답 시간이 초과되었습니다.", None

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else "unknown"
            print(f"[BACKEND ERROR] HTTP {status_code}", file=sys.stderr)
            return f"백엔드 요청 중 오류가 발생했습니다. 상태 코드 {status_code}.", None

        except Exception as e:
            print(f"[BACKEND ERROR] {e}", file=sys.stderr)
            return "백엔드 처리 중 오류가 발생했습니다.", None