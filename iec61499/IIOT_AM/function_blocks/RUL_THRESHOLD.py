class RUL_THRESHOLD:
    """
    Prescriptive rule on the RUL (dissertation, Sections 4.2 and 5.6): emits PRINT
    once when the estimated RUL (hours) falls below THRESHOLD, and only re-arms
    when the RUL rises above THRESHOLD + HYSTERESIS_H (for example after the part
    has been replaced, when the RUL returns to L).

    THRESHOLD is the target RUL: the time needed to have the part printed and
    installed, plus a safety margin derived from the RUL error on the test
    sequences (prognostics/10_choose_thresholds.py): 4.47 h, 3.93 h, and 4.63 h
    for Arms #1, #2, and #3.

    A negative VALUE (-1) means "no estimate yet": it is ignored and never fires.

    Interface: events INIT, RUN -> INIT_O, RUN_O, PRINT; data VALUE, THRESHOLD -> STATUS.
    """

    HYSTERESIS_H = 0.5

    def __init__(self):
        self.armed = True

    def schedule(self, event_name, event_value, value, threshold):
        init_o = None
        run_o = None
        print_ = None
        status = ""

        if event_name == 'INIT':
            self.armed = True
            init_o = 1
            status = "Initialized (armed)"
            return [init_o, run_o, print_, status]

        if event_name == 'RUN':
            if value is None or threshold is None:
                return [None, 1, None, "Error: VALUE and THRESHOLD are required"]
            try:
                rul, thr = float(value), float(threshold)
            except (TypeError, ValueError) as e:
                return [None, 1, None, f"Error: could not convert inputs to float — {e}"]

            if rul < 0:
                return [None, 1, None, "WAITING: no RUL estimate yet"]

            rearm = thr + self.HYSTERESIS_H
            if self.armed and rul < thr:
                self.armed = False
                print_ = 1
                status = f"PRINT ORDER: RUL {rul:.2f} h < threshold {thr:.2f} h"
            elif not self.armed and rul >= rearm:
                self.armed = True
                status = f"RE-ARMED: RUL {rul:.2f} h >= {rearm:.2f} h"
            elif not self.armed:
                status = f"ORDER ISSUED: RUL {rul:.2f} h (waiting for RUL >= {rearm:.2f} h to re-arm)"
            else:
                status = f"OK: RUL {rul:.2f} h >= threshold {thr:.2f} h"
            return [None, 1, print_, status]

        return [None, None, None, status]
