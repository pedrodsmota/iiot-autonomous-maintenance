"""
Step 4 - Health stages (GMM) and Health Index labels (dissertation, Section 5.4.2).

Reads   data/features/<sensor>_features.csv  (to build the session time line)
        data/features/<sensor>_features_pca.csv
Writes  data/features/session_timeline.json
        data/features/<sensor>_training_dataset_with_targets.csv  (HS, HI, RUL_s)
        data/features/<sensor>_label_info.json                    (t_FPT, t_EoL, L)
        data/features/<sensor>_hs_hi_targets.png

Method
  * Time axis: seconds of operation since the start of the data collection,
    from the real duration of each session, so that missing or discarded
    windows do not compress the time line of a sensor.
  * Health stages: a GMM with four components over the principal components,
    after a rolling median (15 min) and an EMA; the normalized operating time,
    weighted by a factor of 15, is added to the input so that the clustering
    respects chronological distance; the clusters are ordered in time and
    passed through a cumulative maximum (monotonic degradation).
  * First predicting time (FPT): first window assigned to stage >= 1.
  * End of life: start of session 22 (EOL_MODE = "failure_onset").
  * Health Index: 1 before the FPT, then HI(t') = 2 - 2^t' (Equation 5.3).
  * The "qualitative notes" stages (from the notes taken during the collection)
    are only used to print a comparison with the GMM stages.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture

from pipeline_common import (EOL_MODE, FAILURE_SESSION, FEAT_DIR, SEED, SENSORS,
                             build_session_timeline, elapsed_time, rul_to_hi)

RUL_TARGETS_H = [3.6, 4.0, 4.5, 5.0]   # 3.6 h = time to act (print + replacement), Section 5.6


def qualitative_hs(session):
    if session <= 9:
        return 0
    if session <= 16:
        return 1
    if session <= 21:
        return 2
    return 3


if __name__ == "__main__":
    timeline = build_session_timeline(FEAT_DIR)
    print("Session durations (s):", timeline["duration_s"])

    for sensor in SENSORS:
        print(f"\n================ {sensor.upper()} ================")
        df = pd.read_csv(FEAT_DIR / f"{sensor}_features_pca.csv")
        pca_cols = [c for c in df.columns if c.startswith("PC")]
        df["t"] = elapsed_time(df, timeline)
        df = df.sort_values("t").reset_index(drop=True)

        smoothed = df[pca_cols].rolling(window=900, min_periods=1, center=True).median()
        smoothed = smoothed.ewm(span=300).mean()
        X = smoothed.values
        t = df["t"].values.astype(float)
        time_norm = ((t - t[0]) / (t[-1] - t[0])).reshape(-1, 1)
        X_comb = np.hstack([X, time_norm * np.max(np.std(X, axis=0)) * 15.0])

        gmm = GaussianMixture(n_components=4, random_state=SEED)
        raw = gmm.fit_predict(X_comb)
        order = pd.Series(t).groupby(raw).mean().sort_values().index
        mapping = {old: new for new, old in enumerate(order)}
        df["Health_Stage_HS"] = pd.Series(raw).map(mapping).cummax().values
        df["Qualitative_Notes_HS"] = df["session"].map(qualitative_hs)

        t_fpt = float(t[np.argmax(df["Health_Stage_HS"].values >= 1)])
        if EOL_MODE == "failure_onset":
            t_eol = float(timeline["offset_s"][FAILURE_SESSION])
        else:
            t_eol = float(t[-1] + 1)
        L = t_eol - t_fpt

        tp = np.clip((t - t_fpt) / L, 0.0, 1.0)
        df["Health_Index_HI"] = np.where(t < t_fpt, 1.0, 2.0 - np.exp(np.log(2) * tp))
        df["RUL_s"] = np.clip(t_eol - t, 0.0, None)

        df.to_csv(FEAT_DIR / f"{sensor}_training_dataset_with_targets.csv", index=False)
        info = {"sensor": sensor, "eol_mode": EOL_MODE, "t_fpt_s": t_fpt, "t_eol_s": t_eol,
                "L_s": L, "fpt_fraction_of_life": t_fpt / t_eol, "n_pca": len(pca_cols)}
        with open(FEAT_DIR / f"{sensor}_label_info.json", "w") as f:
            json.dump(info, f, indent=2)

        print(f"  FPT: {t_fpt / 3600:.2f} h ({t_fpt / t_eol:.1%} of the life cycle, session "
              f"{int(df['session'].iloc[np.argmax(t >= t_fpt)])}) | EoL: {t_eol / 3600:.2f} h "
              f"({EOL_MODE}) | L = {L / 3600:.2f} h")
        print(f"  Agreement between GMM and notes: "
              f"{accuracy_score(df['Qualitative_Notes_HS'], df['Health_Stage_HS']):.1%}")
        print("  Confusion matrix (rows = notes, columns = GMM):")
        print(confusion_matrix(df["Qualitative_Notes_HS"], df["Health_Stage_HS"]))
        print("  HI threshold equivalent to each target RUL:")
        for r in RUL_TARGETS_H:
            print(f"    RUL = {r:.2f} h  ->  HI = {float(rul_to_hi(r * 3600, L)):.3f}")

        hours = t / 3600
        fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
        sc = ax[0].scatter(hours, smoothed["PC1"], c=df["Health_Stage_HS"], cmap="viridis", s=2)
        ax[0].set_title(f"[{sensor.upper()}] Time-aware GMM on median-smoothed PC1")
        plt.colorbar(sc, ax=ax[0], label="GMM stage")
        sc = ax[1].scatter(hours, smoothed["PC1"], c=df["Qualitative_Notes_HS"], cmap="viridis", s=2)
        ax[1].set_title(f"[{sensor.upper()}] Qualitative notes")
        plt.colorbar(sc, ax=ax[1], label="Notes stage")
        ax[2].plot(hours, df["Health_Index_HI"], color="red", lw=2)
        ax[2].axvline(t_eol / 3600, color="black", ls="--", lw=1, label="End of life")
        ax[2].set_title(f"[{sensor.upper()}] Health Index target")
        ax[2].set_xlabel("Operating time (h)")
        ax[2].legend()
        for a in ax:
            a.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FEAT_DIR / f"{sensor}_hs_hi_targets.png", dpi=200)
        plt.close(fig)
