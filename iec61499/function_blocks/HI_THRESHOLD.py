class HI_THRESHOLD:
    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # FB schedule
    # ------------------------------------------------------------------

    def schedule(self, event_name, event_value, value, threshold):
        # --- Output Events ---
        init_o = None
        run_o  = None
        print_ = None   # PRINT event

        # --- Output Variables ---
        status = ""

        if event_name == 'INIT':
            init_o = 1
            status = "Initialized"
            return [init_o, run_o, print_, status]

        if event_name == 'RUN':
            if value is None or threshold is None:
                status = "Error: VALUE and THRESHOLD are required"
                run_o  = 1
                return [None, run_o, None, status]

            try:
                fval  = float(value)
                fthld = float(threshold)
            except (TypeError, ValueError) as e:
                status = f"Error: could not convert inputs to float — {e}"
                run_o  = 1
                return [None, run_o, None, status]

            if fval < fthld:
                status = f"ALERT: value {fval:.4f} is below threshold {fthld:.4f}"
                print_ = 1
            else:
                status = f"OK: value {fval:.4f} >= threshold {fthld:.4f}"

            run_o = 1
            return [None, run_o, print_, status]

        return [None, None, None, status]
