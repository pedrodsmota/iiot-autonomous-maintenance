import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from pathlib import Path
import re
import gc
import os

DATA_DIR = Path(os.environ.get("IIOT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

def sort_key(filepath):
    match = re.search(r'trial_(\d+)(?:_(\d+))?\.csv', filepath.name)
    if match:
        main_num = int(match.group(1))
        sub_num = int(match.group(2)) if match.group(2) else 0
        return (main_num, sub_num)
    return (999, 0)

def extract_features_from_window(window_data, fs=2800):
    """
    Extracts time and frequency features for a 1D array (one window of one axis).
    """
    features = {}
    
    # Time-domain features
    features['rms'] = np.sqrt(np.mean(window_data**2))
    features['var'] = np.var(window_data)
    features['peak'] = np.max(np.abs(window_data))
    features['crest_factor'] = features['peak'] / (features['rms'] + 1e-8)
    features['skew'] = skew(window_data)
    features['kurtosis'] = kurtosis(window_data)
    
    # Frequency-domain features (FFT)
    # Perform Real FFT
    n = len(window_data)
    fft_vals = np.abs(np.fft.rfft(window_data))
    fft_freqs = np.fft.rfftfreq(n, d=1.0/fs)
    
    # Total spectral energy
    spectral_energy = np.sum(fft_vals**2) / n
    features['total_spectral_energy'] = spectral_energy
    
    # Spectral energy in 4 bands (Nyquist is 1400 Hz)
    # Band 1: 0-350 Hz, Band 2: 350-700 Hz, Band 3: 700-1050 Hz, Band 4: 1050-1400 Hz
    bands = [(0, 350), (350, 700), (700, 1050), (1050, 1400)]
    for i, (low, high) in enumerate(bands):
        idx = np.where((fft_freqs >= low) & (fft_freqs < high))[0]
        band_energy = np.sum(fft_vals[idx]**2) / n
        features[f'energy_band_{i+1}'] = band_energy

    return features

def process_sensor_features(sensor_name, input_dir, output_dir, columns, window_size=2800):
    print(f"Extracting features for {sensor_name}...")
    files = sorted(list(input_dir.glob('trial_*.csv')), key=sort_key)
    
    all_trials_features = []
    
    for f in files:
        print(f"  Reading {f.name}...")
        df = pd.read_csv(f, usecols=columns)
        
        # Determine number of windows
        n_samples = len(df)
        n_windows = n_samples // window_size
        
        # We will store the trial's features in a list of dicts
        trial_features_list = []
        
        for w in range(n_windows):
            start_idx = w * window_size
            end_idx = start_idx + window_size
            
            window_features = {
                'trial': f.name,
                'window_idx': w,
                'time_start_sec': w * (window_size / 2800.0)
            }
            
            # Extract features for each column/axis separately
            for col in columns:
                data_window = df[col].iloc[start_idx:end_idx].values.astype(np.float32)
                col_features = extract_features_from_window(data_window, fs=2800)
                
                # Append column name to feature key
                for k, v in col_features.items():
                    window_features[f"{col}_{k}"] = v
                    
            trial_features_list.append(window_features)
            
        all_trials_features.extend(trial_features_list)
        del df
        gc.collect()
        
    if all_trials_features:
        features_df = pd.DataFrame(all_trials_features)
        out_file = output_dir / f"{sensor_name}_features.csv"
        features_df.to_csv(out_file, index=False)
        print(f"Saved {len(features_df)} feature vectors to {out_file}")

if __name__ == '__main__':
    sync_dir = DATA_DIR / "synchronized"
    feat_dir = DATA_DIR / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    
    # Piezo
    if (sync_dir / 'piezo').exists():
        process_sensor_features('piezo', sync_dir / 'piezo', feat_dir, ['Vibration_Val'])
        
    # ADXL345
    if (sync_dir / 'adxl345').exists():
        process_sensor_features('adxl345', sync_dir / 'adxl345', feat_dir, ['x', 'y', 'z'])
        
    # MPU9250
    if (sync_dir / 'mpu9250').exists():
        process_sensor_features('mpu9250', sync_dir / 'mpu9250', feat_dir, ['Accel_X', 'Accel_Y', 'Accel_Z'])
        
    print("Feature Extraction Complete!")
