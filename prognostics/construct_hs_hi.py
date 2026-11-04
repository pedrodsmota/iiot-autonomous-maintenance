import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.mixture import GaussianMixture
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

DATA_DIR = Path(os.environ.get("IIOT_DATA_DIR", Path(__file__).resolve().parent.parent / "data"))

def get_qualitative_hs(trial_name):
    match = re.search(r'trial_(\d+)', str(trial_name))
    if not match: return 0
    t = int(match.group(1))
    if t <= 9: return 0
    elif t <= 15: return 1
    elif t <= 21: return 2
    else: return 3

if __name__ == '__main__':
    feat_dir = DATA_DIR / "features"
    
    sensors = ['adxl', 'mpu', 'piezo']
    
    for sensor in sensors:
        print(f"\n================ Processing {sensor.upper()} ================\n")
        fused_file = feat_dir / f"{sensor}_features_pca.csv"
        
        if not fused_file.exists():
            print(f"PCA file not found for {sensor}: {fused_file}")
            continue
            
        print(f"Loading {sensor} PCA features...")
        df = pd.read_csv(fused_file)
        pca_cols = [c for c in df.columns if c.startswith('PC')]
        df['global_time_idx'] = np.arange(len(df))
        
        # ISSUE 1: Extreme Outliers
        # A rolling mean is heavily distorted by extreme shocks (like the spike at ~23000).
        # We use a Rolling MEDIAN filter first. Medians completely ignore transient extreme outliers.
        print(f"[{sensor}] Applying Rolling Median filter to remove transient operational outliers...")
        window_size = 900 # 15-minute window to solidly filter out short kinetic tasks
        df_median = df[pca_cols].rolling(window=window_size, min_periods=1, center=True).median()
        
        # Smooth the result
        df_smoothed = df_median.ewm(span=300).mean()
        
        X_pca = df_smoothed.values
        
        # ISSUE 2: "Vertical" Clustering (Algorithm doesn't understand time)
        # GMM clusters points purely based on amplitude. So identical amplitude at hr 1 and hr 20
        # get put in the same cluster. 
        # FIX: We explicitly inject Normalized Time into the feature matrix given to the GMM,
        # forcing the AI to consider chronological distance!
        time_norm = (df['global_time_idx'].values / df['global_time_idx'].max()).reshape(-1, 1)
        
        # Give time a massive variance weight so chronological separation strictly overrides transient physical shocks
        pca_scale = np.max(np.std(X_pca, axis=0))
        time_feature = time_norm * pca_scale * 15.0  # Strongly enforces contiguous temporal clustering!
        
        X_combined = np.hstack([X_pca, time_feature])
        
        print(f"[{sensor}] Fitting Time-Aware GMM algorithm...")
        gmm = GaussianMixture(n_components=4, random_state=42)
        df['raw_cluster'] = gmm.fit_predict(X_combined)
        
        # Order clusters chronologically
        cluster_times = df.groupby('raw_cluster')['global_time_idx'].mean().sort_values()
        cluster_mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_times.index)}
        
        # Enforce strict monotonicity (Health States can only go 0 -> 1 -> 2 -> 3, they cannot heal)
        df['Health_Stage_HS'] = df['raw_cluster'].map(cluster_mapping).cummax()
        
        df['Qualitative_Notes_HS'] = df['trial'].apply(get_qualitative_hs)
        
        # --- 2. Health Index (HI) via Analytical Exponential Function ---
        # According to guidelines: Identify First Predicting Threshold (FPT) and Failure Point
        
        # Find when degradation explicitly starts (FPT)
        fpt_idx = df[df['Health_Stage_HS'] >= 1]['global_time_idx'].min()
        
        # Base target formula: HI = 1.0 during Normal Operation, and decays to 0.0 at the end
        hi_array = np.ones(len(df))
        
        # We fit the exponential only from FPT to the end of the lifecycle
        t_decay = df['global_time_idx'].values[fpt_idx:]
        t_decay_norm = (t_decay - t_decay[0]) / (t_decay[-1] - t_decay[0]) # 0.0 to 1.0
        
        # Exponential decay function: HI(t) = 2 - e^(ln(2)*t_norm) ensures boundary from 1.0 to 0.0
        decay_curve = 2.0 - np.exp(np.log(2) * t_decay_norm)
        hi_array[fpt_idx:] = np.clip(decay_curve, 0.0, 1.0)
        
        df['Health_Index_HI'] = hi_array
        
        out_file = feat_dir / f"{sensor}_training_dataset_with_targets.csv"
        df.to_csv(out_file, index=False)
        print(f"[{sensor}] Saved HS and HI targets to {out_file}")
        
        plt.figure(figsize=(15, 10))
        
        plt.subplot(3, 1, 1)
        scatter1 = plt.scatter(df['global_time_idx'], df_smoothed['PC1'], c=df['Health_Stage_HS'], cmap='viridis', alpha=0.5, s=2)
        plt.title(f"[{sensor.upper()}] Time-Aware GMM Clustering on MEDIAN-SMOOTHED PC1")
        plt.ylabel("Filtered PC1")
        plt.colorbar(scatter1, label="GMM Stage")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 2)
        scatter2 = plt.scatter(df['global_time_idx'], df_smoothed['PC1'], c=df['Qualitative_Notes_HS'], cmap='viridis', alpha=0.5, s=2)
        plt.title(f"[{sensor.upper()}] Qualitative Notes Ground Truth")
        plt.ylabel("Filtered PC1")
        plt.colorbar(scatter2, label="Manual Stage Notes")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(3, 1, 3)
        plt.plot(df['global_time_idx'], df['Health_Index_HI'], color='red', linewidth=2)
        plt.title(f"[{sensor.upper()}] Exponential Health Index Target (HI)")
        plt.xlabel("Global Time (1-Second Windows)")
        plt.ylabel("HI [1.0 -> 0.0]")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        out_img = feat_dir / f"{sensor}_hs_hi_targets.png"
        plt.savefig(out_img, dpi=300)
        plt.close()
        print(f"[{sensor}] Plot saved to {out_img}")

        print(f"[{sensor}] HS Clustering Evaluation vs Manual Ground Truth:")
        print(f"Accuracy: {accuracy_score(df['Qualitative_Notes_HS'], df['Health_Stage_HS']):.4f}")
        print(classification_report(df['Qualitative_Notes_HS'], df['Health_Stage_HS'],
                                    target_names=['Stage 0', 'Stage 1', 'Stage 2', 'Stage 3']))
        print("Confusion Matrix:")
        print(confusion_matrix(df['Qualitative_Notes_HS'], df['Health_Stage_HS']))
