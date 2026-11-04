import requests
from urllib.parse import quote
import os
from dotenv import load_dotenv

class PRINT_START:
    def schedule(self, event_name, event_value,
                 octo_url, location, file_path, request_timeout):
        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path)
        api_key = os.getenv('API_KEY')

        # --- output events (3) ---
        init_o = None
        started_o = None
        start_failed_o = None

        # --- output vars (2) ---
        status = ""
        start_reply = ""

        # INIT event - pass-through
        if event_name == 'INIT':
            init_o = 1
            return [init_o, started_o, start_failed_o, status, start_reply]

        if event_name == 'START_PRINT':
            # validate inputs
            if not octo_url or not file_path:
                start_failed_o = 1
                status = "Bad configuration: OCTO_URL and FILE_PATH required"
                return [None, None, start_failed_o, status, start_reply]

            try:
                path_enc = quote(file_path, safe='')
                url = f"{octo_url.rstrip('/')}/api/files/{location}/{path_enc}"
            except Exception as e:
                start_failed_o = 1
                status = f"URL build error: {e}"
                return [None, None, start_failed_o, status, start_reply]

            headers = {}
            if api_key:
                headers["X-Api-Key"] = api_key

            payload = {"command": "select", "print": True}
            try:
                resp = requests.post(url, headers=headers, json=payload,
                                     timeout=(request_timeout or 10.0))
            except Exception as e:
                start_failed_o = 1
                status = f"Request error: {e}"
                return [None, None, start_failed_o, status, start_reply]

            if resp.ok:
                try:
                    start_reply = str(resp.json())
                except Exception:
                    start_reply = resp.text or "<no body>"
                started_o = 1
                status = "Started"
                return [None, started_o, None, status, start_reply]
            else:
                text = resp.text or f"HTTP {resp.status_code}"
                start_failed_o = 1
                status = f"Start failed: {resp.status_code} {text}"
                return [None, None, start_failed_o, status, start_reply]

        # unknown event
        return [None, None, None, status, start_reply]
