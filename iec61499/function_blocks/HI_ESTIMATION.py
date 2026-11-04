import subprocess
import os
import sys
import csv
import time
import signal
import atexit
import threading
from datetime import datetime

class HI_ESTIMATION:
    # Force an fsync (actual write to the SD card, not just Python's/the OS's
    # buffer) after this many logged rows. At ~1 row/s this bounds how much
    # of the trial you could ever lose to a power cut / hard crash to ~this
    # many seconds, without fsync-ing on every single row.
    FSYNC_EVERY_N_ROWS = 10

    # If POLL keeps succeeding but the source file's mtime hasn't moved for
    # this long, the inference pipeline has gone silent (e.g. the python3
    # process died but mosquitto_sub/pv upstream in the shell pipe is still
    # alive, so self.process looks "running"). Flag it instead of silently
    # logging a flat line for hours.
    STALE_AFTER_S = 10.0

    def __init__(self):
        self.process  = None
        self.rul_file = None
        self._pgid    = None   # captured at launch; safe to use even if the shell exits early

        # ── Trial logging state ──
        self.log_fh             = None
        self._csv_writer        = None
        self._log_lock          = threading.Lock()
        self._session_start     = None
        self._last_mtime        = None
        self._last_update_wall  = None
        self._rows_since_fsync  = 0
        self._was_stale         = False

        # ── Ensure the subprocess is killed if the RTE exits for any reason ──
        # NOTE: signal.signal() requires the main thread — this FB runs in a worker
        # thread, so we use atexit only. atexit handlers always execute on interpreter
        # shutdown regardless of which thread registered them.
        atexit.register(self._cleanup)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _kill_group(self, sig=signal.SIGTERM):
        """Send sig to the entire process group. Silently ignore if already gone."""
        if self._pgid is None:
            return
        try:
            os.killpg(self._pgid, sig)
        except ProcessLookupError:
            pass   # already dead — that's fine

    def _close_log(self):
        """Flush + fsync + close the CSV log handle, if open. Never raises."""
        with self._log_lock:
            if self.log_fh is not None:
                try:
                    self.log_fh.flush()
                    os.fsync(self.log_fh.fileno())
                except OSError:
                    pass
                try:
                    self.log_fh.close()
                except OSError:
                    pass
            self.log_fh            = None
            self._csv_writer       = None
            self._rows_since_fsync = 0

    def _log_row(self, rul_val, status_val):
        """Append one row to the trial CSV. Never raises — a logging hiccup
        must never take down RUL polling / the DATA output."""
        if self.log_fh is None:
            return
        now = time.time()
        try:
            with self._log_lock:
                self._csv_writer.writerow([
                    datetime.fromtimestamp(now).isoformat(timespec="milliseconds"),
                    f"{now:.3f}",
                    f"{(now - self._session_start):.3f}" if self._session_start else "",
                    f"{rul_val:.6f}" if rul_val is not None else "",
                    status_val,
                ])
                self.log_fh.flush()                  # Python buffer -> OS
                self._rows_since_fsync += 1
                if self._rows_since_fsync >= self.FSYNC_EVERY_N_ROWS:
                    os.fsync(self.log_fh.fileno())   # OS cache -> physical SD card
                    self._rows_since_fsync = 0
        except OSError:
            pass  # disk hiccup: drop this row, keep the pipeline alive

    def _cleanup(self):
        """Graceful shutdown: SIGTERM first, then SIGKILL to be sure. Also
        flushes/closes the trial log so nothing buffered is lost."""
        if self._pgid is None and self.log_fh is None:
            return
        self._kill_group(signal.SIGTERM)
        # Give the pipeline up to 2 s to flush / exit cleanly, then force-kill.
        # We can't sleep here (called from atexit/signal), so we use waitpid with WNOHANG
        # in a short retry loop instead.
        if self.process is not None:
            for _ in range(20):          # ~2 s max (20 × 0.1 s)
                try:
                    pid, _ = os.waitpid(self.process.pid, os.WNOHANG)
                    if pid != 0:         # child has exited
                        break
                except ChildProcessError:
                    break
                import time; time.sleep(0.1)
            else:
                self._kill_group(signal.SIGKILL)   # escalate if still alive

        self._pgid    = None
        self.process  = None
        self.rul_file = None
        self._close_log()

    # ------------------------------------------------------------------
    # FB schedule
    # ------------------------------------------------------------------

    def schedule(self, event_name, event_value, env_path, script_path, port, mode, model_path, scaler_path, topic):
        # --- Output Events (5) ---
        init_o       = None
        started      = None
        start_failed = None
        stopped      = None
        data         = None

        # --- Output Variables (2) ---
        status = ""
        rul    = 0.0

        if event_name == 'INIT':
            init_o = 1
            status = "Initialized"
            return [init_o, started, start_failed, stopped, data, status, rul]

        if event_name == 'START':
            if not env_path or not script_path:
                start_failed = 1
                status = "Bad configuration: ENV_PATH and SCRIPT_PATH are required"
                return [None, None, start_failed, None, None, status, rul]

            base_path = os.path.dirname(sys.path[0])

            res_env    = os.path.join(base_path, env_path) if not os.path.isabs(env_path) else env_path
            res_script = os.path.join(base_path, 'resources', 'function_blocks', script_path) \
                         if not os.path.isabs(script_path) else script_path

            is_tcp = port and port.strip().isdigit()

            if is_tcp and not port:
                start_failed = 1
                status = "Bad configuration: PORT must be a number for MQTT mode"
                return [None, None, start_failed, None, None, status, rul]

            if is_tcp and not topic:
                start_failed = 1
                status = "Bad configuration: TOPIC is required for MQTT mode"
                return [None, None, start_failed, None, None, status, rul]

            safe_port     = port.replace('/', '_') if port else 'serial_default'
            self.rul_file = f"/tmp/online_rul_{safe_port}.txt"

            # ── Set up the persistent trial log ──────────────────────────
            # NOTE: deliberately NOT under /tmp — on Raspberry Pi OS /tmp is
            # commonly tmpfs (RAM-backed) and evaporates on reboot/power loss,
            # which is exactly what a multi-hour unattended trial can't risk.
            log_note = ""
            try:
                log_dir = os.path.join(base_path, 'resources', 'logs')
                os.makedirs(log_dir, exist_ok=True)
                session_ts = time.strftime('%Y%m%dT%H%M%S')
                log_path = os.path.join(log_dir, f"rul_log_{safe_port}_{session_ts}.csv")
                self.log_fh = open(log_path, 'a', newline='', buffering=1)
                self._csv_writer = csv.writer(self.log_fh)
                if self.log_fh.tell() == 0:
                    self._csv_writer.writerow(
                        ["timestamp_iso", "unix_time", "elapsed_s", "rul", "status"]
                    )
                    self.log_fh.flush()
                self._session_start    = time.time()
                self._last_mtime       = None
                self._last_update_wall = self._session_start
                self._rows_since_fsync = 0
                self._was_stale        = False
                log_note = f" | logging every estimate to {log_path}"
            except OSError as e:
                self.log_fh = None
                log_note = f" | WARNING: could not open trial log ({e}) — RUL values will NOT be saved to disk"
            # ───────────────────────────────────────────────────────────

            if is_tcp:
                # Broker is assumed to run locally on this Pi (the IMU_STREAM side
                # connects out to it as an mosquitto_pub client). -q 0: fire-and-forget,
                # matching the previous nc behaviour. The while-loop mirrors the old
                # nc restart-on-disconnect behaviour in case mosquitto_sub exits
                # (e.g. broker restart).
                cmd = (
                    f"source {res_env}/bin/activate && "
                    f"while true; do mosquitto_sub -h localhost -p {port} -t {topic} -q 0; done "
                    f"| pv -l -a "
                    f"| python3 {res_script}"
                    f" --rul-file {self.rul_file}"
                )
                if mode:
                    cmd += f" --mode {mode}"
            else:
                cmd = (
                    f"source {res_env}/bin/activate && "
                    f"python3 {res_script}"
                    f" --rul-file {self.rul_file}"
                )
                if port:
                    cmd += f" --port {port}"

            if model_path:
                res_model = os.path.join(base_path, 'resources', 'function_blocks', model_path) \
                            if not os.path.isabs(model_path) else model_path
                cmd += f" --model {res_model}"
            if scaler_path:
                res_scaler = os.path.join(base_path, 'resources', 'function_blocks', scaler_path) \
                             if not os.path.isabs(scaler_path) else scaler_path
                cmd += f" --scaler {res_scaler}"

            try:
                self.process = subprocess.Popen(
                    cmd, shell=True, executable='/bin/bash', preexec_fn=os.setsid
                )
                # ── Capture pgid immediately while the shell process is still alive ──
                # Doing this at STOP time is unreliable: the shell may have already
                # exited (while mosquitto_sub/python are still running), making getpgid() raise.
                self._pgid = os.getpgid(self.process.pid)
                started    = 1
                status     = f"Started (pgid={self._pgid}): {cmd}{log_note}"
            except Exception as e:
                start_failed = 1
                status = f"Process error: {e}"
                self._close_log()  # don't leave an empty log file open if launch failed

            return [None, started, start_failed, None, None, status, rul]

        if event_name == 'STOP':
            if self._pgid is not None or self.process is not None or self.log_fh is not None:
                self._cleanup()
                stopped = 1
                status  = "Process successfully stopped"
            else:
                stopped = 1
                status  = "No process currently running"

            return [None, None, None, stopped, None, status, rul]

        if event_name == 'POLL':
            if not self.rul_file or not os.path.isfile(self.rul_file):
                status = "No RUL data available yet"
                return [None, None, None, None, None, status, rul]

            try:
                mtime = os.path.getmtime(self.rul_file)
                with open(self.rul_file, 'r') as fh:
                    lines = fh.read().splitlines()
                rul    = float(lines[0])
                status = lines[1] if len(lines) > 1 else "OK"
                data   = 1

                now = time.time()
                if mtime != self._last_mtime:
                    # A genuinely new estimate was written since the last poll —
                    # this is the "every value" we log, once per real estimate
                    # rather than once per poll (avoids duplicate rows if POLL
                    # ever races ahead of the ~1 s inference cadence).
                    self._last_mtime       = mtime
                    self._last_update_wall = now
                    if self._was_stale:
                        status += " | RECOVERED"
                        self._was_stale = False
                    self._log_row(rul, status)
                else:
                    gap = now - (self._last_update_wall or now)
                    if gap > self.STALE_AFTER_S:
                        status += f" | STALE: no new estimate for {gap:.1f}s"
                        if not self._was_stale:
                            self._was_stale = True
                            self._log_row(rul, status)  # mark the moment it went stale

            except (ValueError, OSError) as e:
                status = f"Error reading RUL file: {e}"

            return [None, None, None, None, data, status, rul]

        return [None, None, None, None, None, status, rul]
