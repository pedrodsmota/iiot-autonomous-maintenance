"""
Step 10 - Calibrate the thresholds of the prescriptive logic from the RUL error
on the test sequences (dissertation, Sections 4.2 and 5.6, Table 5.6).

Logic
  RUL_THRESHOLD issues the print order when the estimated RUL falls below a
  target RUL. The risk that matters is an optimistic estimate (estimated RUL >
  real RUL): the order is issued late and the part may not be ready before the
  failure. A pessimistic estimate only anticipates the order. Hence

      target RUL = time to act + quantile q of the error (estimated - real)

  computed only on the test sequences of the decision zone (real RUL between
  ZONE_H[0] and ZONE_H[1] hours), with a minimum margin of MIN_MARGIN_H.
  The dissertation uses q = 0.95. The equivalent HI threshold is also reported.

  The test sequences were seen neither in training nor in tuning. Replaying a
  whole recorded session would mostly replay training windows, and the error
  would look smaller than it is, so the thresholds are derived from here.

Reads   data/processed/<sensor>_dataset.pkl, models/scaler_<sensor>.pkl,
        models/model_<sensor>.pt
Writes  results/thresholds.json
"""

import json
import pickle

import numpy as np
import torch

from pipeline_common import (MODEL_DIR, PROCESSED_DIR, RESULTS_DIR, SENSORS, get_device,
                             hi_to_rul, rul_to_hi)
from model_common import CNN_BiLSTM, SequenceDataset, predict

T_NEEDED_H = 3.6            # time to act: measured print (3 h 26 min) + ~10 min to replace the gripper
ZONE_H = (2.5, 7.0)         # real RUL range used for the error in the decision zone
QUANTILES = [0.80, 0.90, 0.95]
MIN_MARGIN_H = 0.25         # minimum margin, even if the model is pessimistic in the zone


def load_model(sensor, device):
    ck = torch.load(MODEL_DIR / f"model_{sensor}.pt", map_location="cpu", weights_only=False)
    hp = ck["hparams"]
    model = CNN_BiLSTM(ck["input_dim"], hp["cnn_filters"], hp["cnn_kernel"], hp["lstm_hidden"],
                       hp["lstm_layers"], hp["dropout_rate"])
    model.load_state_dict(ck["model_state_dict"])
    return model.to(device), ck


if __name__ == "__main__":
    device = get_device()
    out = {"t_needed_h": T_NEEDED_H, "zone_h": ZONE_H, "sensors": {}}
    for sensor in SENSORS:
        with open(PROCESSED_DIR / f"{sensor}_dataset.pkl", "rb") as f:
            ds = pickle.load(f)
        with open(MODEL_DIR / f"scaler_{sensor}.pkl", "rb") as f:
            scaler = pickle.load(f)
        model, ck = load_model(sensor, device)
        T = ds["time_steps"]
        L = ds["label_info"]["L_s"]
        Xs = scaler.transform(ds["X_raw"]).astype(np.float32)
        test = SequenceDataset(Xs, ds["y"], ds["starts"]["test"], T)
        pred, _ = predict(model, test, device)
        pred = np.clip(pred, 0.0, 1.0)

        last = ds["starts"]["test"] + T - 1
        rul_true = ds["rul_s"][last] / 3600
        rul_pred = hi_to_rul(pred, L) / 3600
        zone = (rul_true >= ZONE_H[0]) & (rul_true <= ZONE_H[1]) & (ds["y"][last] < 1.0)
        err = rul_pred[zone] - rul_true[zone]          # > 0: optimistic (late order)
        if len(err) == 0:
            print(f"\n{sensor.upper()}: no test sequence in the zone {ZONE_H} h")
            continue

        s = {"L_h": L / 3600, "n_zone": int(zone.sum()),
             "bias_h": float(err.mean()), "mae_h": float(np.abs(err).mean()),
             "rmse_h": float(np.sqrt(np.mean(err ** 2))), "worst_late_h": float(err.max()),
             "options": {}}
        print(f"\n{sensor.upper()}  (L = {L / 3600:.2f} h, {s['n_zone']} test sequences with "
              f"a real RUL between {ZONE_H[0]} and {ZONE_H[1]} h)")
        print(f"  RUL error in the zone: bias {s['bias_h']:+.2f} h | MAE {s['mae_h']:.2f} h | "
              f"RMSE {s['rmse_h']:.2f} h | worst delay {s['worst_late_h']:+.2f} h")
        for q in QUANTILES:
            margin = max(MIN_MARGIN_H, float(np.quantile(err, q)))
            target = T_NEEDED_H + margin
            hi = float(rul_to_hi(target * 3600, L))
            s["options"][f"q{int(q * 100)}"] = {"margin_h": margin, "rul_target_h": target,
                                                "hi_threshold": hi}
            print(f"  quantile {q:.2f}: margin {margin:.2f} h -> target RUL {target:.2f} h "
                  f"-> equivalent HI threshold {hi:.3f}")
        out["sensors"][sensor] = s

    RESULTS_DIR.mkdir(exist_ok=True)
    with open(RESULTS_DIR / "thresholds.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'thresholds.json'}")
