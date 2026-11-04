import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import re
import os

DATA_DIR = Path(os.environ.get("IIOT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

def get_base_trial_name(name):
    m = re.search(r'trial_(\d+)', name)
    if m:
        return f"trial_{m.group(1)}.csv"
    return name

if __name__ == '__main__':
    feat_dir = DATA_DIR / "features"
    
    print("Loading feature sets...")
    df_adxl = pd.read_csv(feat_dir / "adxl345_features.csv")
    df_mpu = pd.read_csv(feat_dir / "mpu9250_features.csv")
    df_piezo = pd.read_csv(feat_dir / "piezo_features.csv")

    df_piezo['trial_base'] = df_piezo['trial'].apply(get_base_trial_name)
    df_adxl['trial_base'] = df_adxl['trial'].apply(get_base_trial_name)
    df_mpu['trial_base'] = df_mpu['trial'].apply(get_base_trial_name)
    
    # REMOVE TRIAL 8 based on user's manual notes (only from ADXL)
    df_adxl = df_adxl[df_adxl['trial_base'] != 'trial_8.csv'] 

    # PURGE "DEAD" SENSOR ZONES
    ACTIVITY_THRESHOLD = 1e-6
    df_adxl = df_adxl[df_adxl['x_var'] > ACTIVITY_THRESHOLD]
    df_mpu = df_mpu[df_mpu['Accel_X_var'] > ACTIVITY_THRESHOLD]
    df_piezo = df_piezo[df_piezo['Vibration_Val_var'] > ACTIVITY_THRESHOLD]

    
    datasets = {
        'adxl': df_adxl.reset_index(drop=True),
        'mpu': df_mpu.reset_index(drop=True),
        'piezo': df_piezo.reset_index(drop=True)
    }
    
    for name, df_sensor in datasets.items():
        print(f"\nProcessing {name} with length {len(df_sensor)}...")
        
        metadata_cols = ['trial_base', 'window_idx', 'time_start_sec_adxl', 'time_start_sec_mpu', 'time_start_sec', 'trial_adxl', 'trial_mpu', 'trial']
        metadata_cols = [c for c in metadata_cols if c in df_sensor.columns]
        
        metadata = df_sensor[metadata_cols].copy()
        features = df_sensor.drop(columns=metadata_cols)
        
        features.replace([np.inf, -np.inf], np.nan, inplace=True)
        features.fillna(0, inplace=True)
        
        print(f"  Standardizing {name} features...")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(features)
        
        print(f"  Applying PCA Information Fusion for {name}...")
        pca = PCA(n_components=0.95)
        fused_pca = pca.fit_transform(scaled_features)
        
        print(f"  {name}: Dimensionality reduced from {features.shape[1]} raw features to {fused_pca.shape[1]} principle components!")
        
        pca_cols = [f"PC{i+1}" for i in range(fused_pca.shape[1])]
        df_final = pd.DataFrame(fused_pca, columns=pca_cols)
        
        df_final['trial'] = metadata['trial_base']
        df_final['window_idx'] = metadata['window_idx']
        
        out_file = feat_dir / f"{name}_features_pca.csv"
        df_final.to_csv(out_file, index=False)
        print(f"  Successfully saved {name} PCA features to {out_file}")
