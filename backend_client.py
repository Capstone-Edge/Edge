import sys
from typing import Any

import requests


class BackendClient:
    """Backend process API 호출과 TTS 응답 추출을 담당한다."""

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
        self.session = requests.Session()

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.backend_url}{path}"

        print(f"[HTTP] POST {url}")
        print(f"[HTTP] payload={payload}")

        response = self.session.post(
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
