import requests
import os
import time
from dotenv import load_dotenv

class PRINT_CHECK:
    def schedule(self, event_name, event_value,
                 octo_url, request_timeout):

        env_path = os.path.join(os.path.dirname(__file__), '.env')
        load_dotenv(env_path)
        api_key = os.getenv('API_KEY')

        # --- output events (5) ---
        init_o = None
        read_o = None
        print_done_o = None
        print_error_o = None
        progress_update_o = None

        # --- output vars (3) ---
        job_state = ""
        completion = -1.0
        job_info = ""

        if event_name == 'INIT':
            init_o = 1
            return [
                init_o, read_o, print_done_o, print_error_o, progress_update_o,
                job_state, completion, job_info
            ]

        if event_name == 'READ':
            # 1-second delay to prevent hammering the OctoPrint API in the continuous loop
            time.sleep(1)
            
            # Setup HTTP timeout fallback
            timeout_val = request_timeout if request_timeout else 5.0

            if not octo_url:
                print_error_o = 1
                read_o = 1
                job_info = "OCTO_URL missing"
                return [
                    None, read_o, None, print_error_o, None,
                    job_state, completion, job_info
                ]

            url = f"{octo_url.rstrip('/')}/api/job"
            headers = {}
            if api_key:
                headers["X-Api-Key"] = api_key

            try:
                resp = requests.get(url, headers=headers, timeout=timeout_val)
            except Exception as e:
                print_error_o = 1
                read_o = 1
                job_info = str(e)
                return [
                    None, read_o, None, print_error_o, None,
                    job_state, completion, job_info
                ]

            if not resp.ok:
                print_error_o = 1
                read_o = 1
                job_info = f"HTTP {resp.status_code}: {resp.text}"
                return [
                    None, read_o, None, print_error_o, None,
                    job_state, completion, job_info
                ]

            js = resp.json()
            job_info = str(js)
            job_state = js.get("state", "")

            progress = js.get("progress") or {}
            completion_val = progress.get("completion")
            
            # Triggering READ_O signals to DINASORE that the loop iteration has successfully executed
            read_o = 1 

            if js.get("error"):
                print_error_o = 1
                job_info = js.get("error")
                return [
                    None, read_o, None, print_error_o, None,
                    job_state, completion, job_info
                ]

            if isinstance(completion_val, (int, float)):
                completion = float(completion_val)
                # OctoPrint reports completion in percent (0-100)
                if completion >= 99.9:
                    print_done_o = 1
                    return [
                        None, read_o, print_done_o, None, None,
                        job_state, completion, job_info
                    ]
                else:
                    progress_update_o = 1
                    return [
                        None, read_o, None, None, progress_update_o,
                        job_state, completion, job_info
                    ]

            # fallback: idle printer = done
            if job_state.lower() == "operational":
                print_done_o = 1
                return [
                    None, read_o, print_done_o, None, None,
                    job_state, completion, job_info
                ]

            # any other state without progress (Offline, Connecting, ...):
            # still signal READ_O so the polling loop keeps running
            return [
                None, read_o, None, None, None,
                job_state, completion, job_info
            ]

        # unknown event
        return [
            None, None, None, None, None,
            job_state, completion, job_info
        ]
