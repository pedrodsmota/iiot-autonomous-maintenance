import time
from collections import deque
from threading import Lock


class PRINT_QUEUE:
    """FIFO queue placed between READ_MQTT and PRINT_START for a shared printer.

    - REQUEST      : a new print request arrived (payload = file to print).
                     Printer available -> START_PRINT right away.
                     Printer busy      -> request is stored (QUEUED).
    - CHECK        : periodic printer status (JOB_STATE from PRINT_CHECK).
                     If the printer is available and the queue is not empty,
                     ONE request is popped and sent (START_PRINT). The next one
                     is only sent after the printer has gone busy and become
                     available again.
    - START_FAILED : PRINT_START could not start the job -> retry / drop.
    """

    MAX_RETRIES = 3               # start attempts per job before it is dropped
    DEFAULT_START_TIMEOUT = 30.0  # s waiting for the printer to go busy after START_PRINT

    def __init__(self):
        self.lock = Lock()
        self.queue = deque()      # entries: [file_path, attempts]
        self.printer_state = ''   # last JOB_STATE received ('' = unknown)
        self.dispatching = False  # START_PRINT sent, printer not seen busy yet
        self.dispatch_time = 0.0
        self.current = None       # entry currently being started

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _available(state):
        return str(state).strip().lower() == 'operational'

    @staticmethod
    def _busy(state):
        return str(state).strip().lower().startswith(('start', 'print', 'paus', 'resum'))

    def _can_dispatch(self):
        return bool(self.queue) and not self.dispatching and self._available(self.printer_state)

    def _dispatch_next(self):
        """Pops the head of the queue (this is the 'subtract') and arms the guard."""
        entry = self.queue.popleft()
        entry[1] += 1
        self.current = entry
        self.dispatching = True
        self.dispatch_time = time.time()
        return entry[0]

    # ------------------------------------------------------------ schedule
    def schedule(self, event_name, event_value, request_file, job_state, start_timeout):
        init_o = None
        queued_o = None
        start_print_o = None
        dropped_o = None

        file_path = ''
        status = ''

        with self.lock:
            if event_name == 'INIT':
                init_o = 1

            elif event_name == 'REQUEST':
                self.queue.append([request_file, 0])
                if self._can_dispatch():
                    file_path = self._dispatch_next()
                    start_print_o = 1
                else:
                    queued_o = 1
                    status = f"Queued (position {len(self.queue)})"

            elif event_name == 'CHECK':
                self.printer_state = job_state or ''
                timeout = start_timeout if start_timeout and start_timeout > 0 \
                    else self.DEFAULT_START_TIMEOUT

                # release the guard once the printer picked the job up (busy)
                # or if it never did within the timeout
                if self.dispatching:
                    if self._busy(self.printer_state) or \
                            time.time() - self.dispatch_time > timeout:
                        self.dispatching = False
                        self.current = None

                if self._can_dispatch():
                    file_path = self._dispatch_next()
                    start_print_o = 1

            elif event_name == 'START_FAILED':
                self.dispatching = False
                job = self.current
                self.current = None
                if job is not None:
                    if job[1] < self.MAX_RETRIES:
                        # back to the head; retried on the next CHECK
                        self.queue.appendleft(job)
                    else:
                        dropped_o = 1
                        status = f"Dropped '{job[0]}' after {job[1]} failed start attempts"

            queue_size = len(self.queue)

        return [init_o, queued_o, start_print_o, dropped_o,
                file_path, queue_size, status]
