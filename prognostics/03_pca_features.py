"""
Step 3 - Standardize the features of each sensor and reduce them by PCA,
retaining 95 % of the variance (dissertation, Section 5.4.2).

The principal components are used ONLY to construct the health labels
(step 4); the prognostic models are trained on the features themselves.

Reads   data/features/<sensor>_features.csv
Writes  data/features/<sensor>_features_pca.csv

Windows with a variance below ACTIVITY_THRESHOLD (no signal) are discarded,
and session 8 of the ADXL345, which has no data, is excluded.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pipeline_common import (ACTIVITY_THRESHOLD, EXCLUDED_SESSIONS, FEAT_DIR,
                             FEATURE_FILES, add_window_keys)

META = ["trial", "window_idx", "time_start_sec", "session", "part", "win"]
VAR_COLUMN = {"adxl": "x_var", "mpu": "Accel_X_var", "piezo": "Vibration_Val_var"}

if __name__ == "__main__":
    for name in ["adxl", "mpu", "piezo"]:
        df = add_window_keys(pd.read_csv(FEAT_DIR / FEATURE_FILES[name]))
        df = df[~df["session"].isin(EXCLUDED_SESSIONS.get(name, set()))]
        df = df[df[VAR_COLUMN[name]] > ACTIVITY_THRESHOLD].reset_index(drop=True)
        assert not df.duplicated(["session", "win"]).any(), "duplicated window keys"

        features = df.drop(columns=[c for c in META if c in df.columns])
        features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
        print(f"\nProcessing {name} with length {len(df)}...")
        scaled = StandardScaler().fit_transform(features)
        pca = PCA(n_components=0.95)
        pcs = pca.fit_transform(scaled)
        print(f"  {name}: {features.shape[1]} features -> {pcs.shape[1]} principal components")

        out = pd.DataFrame(pcs, columns=[f"PC{i + 1}" for i in range(pcs.shape[1])])
        for c in ["trial", "window_idx", "session", "part", "win"]:
            out[c] = df[c].values
        out.to_csv(FEAT_DIR / f"{name}_features_pca.csv", index=False)
        print(f"  Saved {FEAT_DIR / f'{name}_features_pca.csv'}")
