import subprocess
import os
import atexit

class IMU_STREAM:
    def __init__(self):
        self.process = None
        # atexit works from any thread; handles Ctrl+C and normal RTE shutdown
        atexit.register(self._cleanup)

    def _cleanup(self):
        if self.process:
            self._kill_process()

    def _kill_process(self):
        if not self.process:
            return True, "No process running"

        try:
            pgid = os.getpgid(self.process.pid)
        except ProcessLookupError:
            self.process = None
            return True, "Process was not running"

        subprocess.run(f"sudo kill -TERM -{pgid}", shell=True)

        try:
            self.process.wait(timeout=5)
            self.process = None
            return True, "Process stopped (SIGTERM)"
        except subprocess.TimeoutExpired:
            pass

        subprocess.run(f"sudo kill -KILL -{pgid}", shell=True)

        try:
            self.process.wait(timeout=3)
            self.process = None
            return True, "Process stopped (SIGKILL)"
        except subprocess.TimeoutExpired:
            self.process = None
            return False, "Process may still be running after SIGKILL"
        except Exception as e:
            self.process = None
            return False, f"Error during SIGKILL wait: {e}"

    def schedule(self, event_name, event_value, exec_path, save_file, duration, freq, target_ip, target_port, topic):
        init_o = None
        started = None
        start_failed = None
        stopped = None
        status = ""

        if event_name == 'INIT':
            init_o = 1
            status = "Initialized"
            return [init_o, started, start_failed, stopped, status]

        if event_name == 'START':
            if not exec_path:
                start_failed = 1
                status = "Bad configuration: EXEC_PATH is required"
                return [None, None, start_failed, None, status]

            try:
                duration_val = int(duration) if duration else 0
            except ValueError:
                duration_val = 0

            effective_duration = duration_val if duration_val > 0 else 2147483647

            cmd = f"sudo {exec_path}"

            if save_file:
                cmd += f" -s {save_file}"

            try:
                freq_val = int(freq) if freq else 0
            except ValueError:
                freq_val = 0

            if freq_val > 0:
                cmd += f" -f {freq_val}"

            cmd += f" -t {effective_duration}"

            if target_ip and target_port and not topic:
                start_failed = 1
                status = "Bad configuration: TOPIC is required to publish over MQTT"
                return [None, None, start_failed, None, status]

            if target_ip and target_port and topic:
                # -l: read stdin line by line, publish each line as one MQTT message.
                # Assumes EXEC_PATH emits one newline-terminated sample per line.
                # -q 0: fire-and-forget, matches the old nc behaviour and keeps up
                # with high-rate telemetry without broker/client ACK overhead.
                cmd += f" | mosquitto_pub -h {target_ip} -p {target_port} -t {topic} -l -q 0"

            try:
                self.process = subprocess.Popen(
                    cmd,
                    shell=True,
                    preexec_fn=os.setsid
                )
                started = 1
                status = f"Started: {cmd}"
            except Exception as e:
                start_failed = 1
                status = f"Process error: {e}"

            return [None, started, start_failed, None, status]

        if event_name == 'STOP':
            success, msg = self._kill_process()
            if success:
                stopped = 1
                status = msg
                return [None, None, None, stopped, status]
            else:
                start_failed = 1
                status = msg
                return [None, None, start_failed, None, status]

        return [None, None, None, None, status]
