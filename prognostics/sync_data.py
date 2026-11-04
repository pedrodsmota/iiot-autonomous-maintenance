import numpy as np
import pandas as pd
from scipy import signal
import os
import math
import gc
from pathlib import Path

DATA_DIR = Path(os.environ.get("IIOT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

def process_file_adxl(filepath, target_sr=2800, original_sr=3200, hp_cutoff=1.0):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    time = df['time'].values
    
    # Cast to float32 to halve memory usage immediately
    data = df[['x', 'y', 'z']].values.astype(np.float32)
    del df # Free memory early
    
    # Filter
    nyq = 0.5 * original_sr
    b, a = signal.butter(4, hp_cutoff / nyq, btype='high')
    data_filtered = signal.filtfilt(b, a, data, axis=0).astype(np.float32)
    del data
    
    # Use resample_poly instead of resample (resample uses massive FFT -> Out Of Memory)
    gcd = math.gcd(target_sr, original_sr)
    up = target_sr // gcd
    down = original_sr // gcd
    data_resampled = signal.resample_poly(data_filtered, up, down, axis=0)
    del data_filtered
    
    # Reconstruct timeline
    time_resampled = np.linspace(time[0], time[-1], len(data_resampled), endpoint=False)
    
    df_out = pd.DataFrame(data_resampled, columns=['x', 'y', 'z'])
    df_out.insert(0, 'time', time_resampled)
    return df_out

def process_file_mpu(filepath, target_sr=2800, original_sr=4000, hp_cutoff=1.0):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    time = df['Sample'].values / original_sr
    
    data = df[['Accel_X', 'Accel_Y', 'Accel_Z']].values.astype(np.float32)
    del df
    
    nyq = 0.5 * original_sr
    b, a = signal.butter(4, hp_cutoff / nyq, btype='high')
    data_filtered = signal.filtfilt(b, a, data, axis=0).astype(np.float32)
    del data
    
    gcd = math.gcd(target_sr, original_sr)
    up = target_sr // gcd
    down = original_sr // gcd
    data_resampled = signal.resample_poly(data_filtered, up, down, axis=0)
    del data_filtered
    
    time_resampled = np.linspace(time[0], time[-1], len(data_resampled), endpoint=False)
    
    df_out = pd.DataFrame(data_resampled, columns=['Accel_X', 'Accel_Y', 'Accel_Z'])
    df_out.insert(0, 'time', time_resampled)
    return df_out

def process_file_piezo(filepath, target_sr=2800, hp_cutoff=1.0):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    time = df['Time_Sec'].values
    data = df['Vibration_Val'].values.astype(np.float32)
    del df
    
    time_uniform = np.arange(time[0], time[-1], 1.0 / target_sr)
    data_interp = np.interp(time_uniform, time, data).astype(np.float32)
    del time
    del data
    
    nyq = 0.5 * target_sr
    b, a = signal.butter(4, hp_cutoff / nyq, btype='high')
    data_filtered = signal.filtfilt(b, a, data_interp).astype(np.float32)
    del data_interp
    
    df_out = pd.DataFrame({'time': time_uniform, 'Vibration_Val': data_filtered})
    return df_out

if __name__ == '__main__':
    base_dir = DATA_DIR / "raw_csv"
    output_dir = DATA_DIR / "synchronized"
    
    sensor_dirs = {
        'adxl345': base_dir / "ARM1_adxl345",
        'piezo': base_dir / "ARM2_piezo",
        'mpu9250': base_dir / "ARM3_mpu9250"
    }

    for sensor in sensor_dirs.keys():
        (output_dir / sensor).mkdir(parents=True, exist_ok=True)

    print("Processing ADXL345...")
    if sensor_dirs['adxl345'].exists():
        for file_path in sensor_dirs['adxl345'].glob('trial_*.csv'):
            print(f"  {file_path.name}")
            df = process_file_adxl(file_path)
            file_out = output_dir / 'adxl345' / file_path.name
            df.to_csv(file_out, index=False)
            del df
            gc.collect()

    print("Processing Piezo...")
    if sensor_dirs['piezo'].exists():
        for file_path in sensor_dirs['piezo'].glob('trial_*.csv'):
            print(f"  {file_path.name}")
            df = process_file_piezo(file_path)
            file_out = output_dir / 'piezo' / file_path.name
            df.to_csv(file_out, index=False)
            del df
            gc.collect()

    print("Processing MPU9250...")
    if sensor_dirs['mpu9250'].exists():
        for file_path in sensor_dirs['mpu9250'].glob('trial_*.csv'):
            print(f"  {file_path.name}")
            df = process_file_mpu(file_path)
            file_out = output_dir / 'mpu9250' / file_path.name
            df.to_csv(file_out, index=False)
            del df
            gc.collect()
            
    print("Done!")