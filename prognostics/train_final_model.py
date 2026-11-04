import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# WHAT CHANGED AND WHY
#
# Previously: one model was trained on fused PCA features from all three arms
#             combined, validated with Leave-One-Trial-Out by time segment.
#
# Now: three independent models are trained, one per physical arm/sensor.
#      Each model learns to map its own sensor's raw features to the shared
#      HI target. In real-time deployment, each arm runs its own inference
#      loop independently — no cross-arm synchronization required.
#
#      Evaluation uses the final 25% of the lifecycle (the failure region)
#      as the test set, matching the chronological split in dataset_builder.py.
#      This is the standard PHM single-trajectory evaluation strategy.
# ─────────────────────────────────────────────────────────────────────────────


# ── Architecture (unchanged from tuning phase) ────────────────────────────────
class CNN_BiLSTM(nn.Module):
    """
    1D-CNN (auxiliary) + Bidirectional LSTM (primary).
    CNN compresses local spectral patterns from the sensor's feature window.
    BiLSTM captures the temporal degradation trajectory and produces HI regression.
    """
    def __init__(self, input_dim, cnn_filters, cnn_kernel, lstm_hidden, lstm_layers, dropout):
        super(CNN_BiLSTM, self).__init__()
        self.cnn = nn.Conv1d(in_channels=input_dim,
                             out_channels=cnn_filters,
                             kernel_size=cnn_kernel,
                             padding='same')
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(input_size=cnn_filters,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0,
                            bidirectional=True)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, 1)  # ×2 because bidirectional

    def forward(self, x):
        # x: (batch, time_steps, features)
        x = x.transpose(1, 2)          # → (batch, features, time_steps) for Conv1d
        x = self.relu(self.cnn(x))
        x = self.pool(x)
        x = x.transpose(1, 2)          # → (batch, time_steps/2, cnn_filters)
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])   # last time step
        return self.fc(out).squeeze()


# ── PHM 2012 scoring (Eq. 3 from the guidelines paper) ───────────────────────
def phm2012_score(y_true, y_pred):
    """
    Penalises late predictions more heavily than early ones.
    Target: maximise towards 1.0.
    """
    E = (y_true - y_pred) / (np.abs(y_true) + 1e-8)
    scores = np.where(E > 0,
                      np.exp(-np.log(0.5) * E / 20),
                      np.exp(-np.log(0.5) * E / 5))
    return float(np.mean(scores))


# ── Per-sensor training routine ───────────────────────────────────────────────
def train_sensor_model(sensor_name, dataset, hparams, device, results_dir):
    print(f"\n{'='*60}")
    print(f"  Sensor: {sensor_name.upper()}")
    print(f"  Input features : {dataset['input_dim']}")
    print(f"  Train sequences: {len(dataset['X_train'])}")
    print(f"  Test  sequences: {len(dataset['X_test'])}")
    print(f"{'='*60}")

    # ── Dataloaders ───────────────────────────────────────────────────────────
    X_train = torch.tensor(dataset['X_train'], dtype=torch.float32)
    y_train = torch.tensor(dataset['y_train'], dtype=torch.float32)
    X_test  = torch.tensor(dataset['X_test'],  dtype=torch.float32)
    y_test  = torch.tensor(dataset['y_test'],  dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train),
                              batch_size=256, shuffle=True,  drop_last=False)
    test_loader  = DataLoader(TensorDataset(X_test,  y_test),
                              batch_size=256, shuffle=False, drop_last=False)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = CNN_BiLSTM(
        input_dim   = dataset['input_dim'],
        cnn_filters = hparams['cnn_filters'],
        cnn_kernel  = hparams['cnn_kernel'],
        lstm_hidden = hparams['lstm_hidden'],
        lstm_layers = hparams['lstm_layers'],
        dropout     = hparams['dropout_rate'],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hparams['learning_rate'],
                                  weight_decay=hparams['weight_decay'])

    # ── Training loop ─────────────────────────────────────────────────────────
    EPOCHS = 50
    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_x.size(0)

        avg = epoch_loss / len(X_train)
        train_losses.append(avg)
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch [{epoch+1:>2}/{EPOCHS}]  Train MSE: {avg:.6f}")

    # ── Evaluation on held-out failure region (final 25%) ────────────────────
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds.extend(model(batch_x.to(device)).cpu().numpy())
            trues.extend(batch_y.numpy())

    preds = np.clip(np.array(preds), 0.0, 1.0)
    trues = np.array(trues)

    mse   = float(np.mean((preds - trues) ** 2))
    rmse  = float(np.sqrt(mse))
    mae   = float(np.mean(np.abs(preds - trues)))
    score = phm2012_score(trues, preds)

    print(f"\n  ── Test Metrics (final 25% — failure region) ──")
    print(f"  MSE          : {mse:.6f}")
    print(f"  RMSE         : {rmse:.6f}")
    print(f"  MAE          : {mae:.6f}")
    print(f"  PHM-2012 Score: {score:.4f}  (target → 1.0)")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(13, 12))

    # HI prediction
    axes[0].plot(trues, color='black', linewidth=2,   label='True HI (target)')
    axes[0].plot(preds, color='red',   linewidth=1.5,
                 linestyle='--', label=f'Predicted HI  (RMSE={rmse:.4f})')
    axes[0].set_title(f'{sensor_name.upper()} — Health Index Prediction\n'
                      f'Test set: 20% Random Block Shuffle',
                      fontsize=12)
    axes[0].set_xlabel('Time Sequence (Windows)')
    axes[0].set_ylabel('Health Index  [1.0 → 0.0]')
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Training loss
    axes[1].plot(train_losses, color='steelblue', linewidth=1.5)
    axes[1].set_title(f'{sensor_name.upper()} — Training Loss Curve')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].grid(True, alpha=0.3)

    # Third subplot: PHM-2012 error distribution
    E = (trues - preds) / (np.abs(trues) + 1e-8)
    early = E[E > 0]
    late  = E[E <= 0]

    axes[2].hist(late,  bins=60, color='orange', alpha=0.7, label='Late predictions (E≤0)')
    axes[2].hist(early, bins=60, color='red',    alpha=0.7, label='Early predictions (E>0)')
    axes[2].axvline(0, color='black', linewidth=1.5, linestyle='--')
    axes[2].set_title(f'{sensor_name.upper()} — PHM-2012 Error Distribution')
    axes[2].set_xlabel('Percentage Error  E = (y_true - y_pred) / y_true')
    axes[2].set_ylabel('Count')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = results_dir / f"prediction_{sensor_name}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {plot_path}")

    # ── Save model checkpoint ─────────────────────────────────────────────────
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'hparams':    hparams,
        'input_dim':  dataset['input_dim'],
        'time_steps': dataset['time_steps'],
        'sensor':     sensor_name,
        'metrics':    {'mse': mse, 'rmse': rmse, 'mae': mae, 'phm_score': score},
    }
    model_path = results_dir / f"model_{sensor_name}.pt"
    torch.save(checkpoint, model_path)
    print(f"  Model saved  → {model_path}")

    return checkpoint['metrics']


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    base_dir    = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
    results_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Train one model per physical arm/sensor
    SENSORS = ['adxl']
    all_metrics = {}

    for sensor_name in SENSORS:
        # Load the independent best hyperparameters for THIS specific sensor
        hparams_path = base_dir / f"best_hyperparameters_{sensor_name}.pkl"
        if not hparams_path.exists():
            print(f"WARNING: {hparams_path} not found. Did you tune {sensor_name}?")
            continue
            
        with open(hparams_path, "rb") as f:
            hparams = pickle.load(f)
        print(f"\nLoaded hyperparameters for {sensor_name.upper()}:", hparams)

        dataset_path = base_dir / f"{sensor_name}_dataset.pkl"
        if not dataset_path.exists():
            print(f"WARNING: {dataset_path} not found — skipping {sensor_name}.")
            continue
            
        with open(dataset_path, "rb") as f:
            dataset = pickle.load(f)

        all_metrics[sensor_name] = train_sensor_model(
            sensor_name, dataset, hparams, device, results_dir
        )

    # ── Final summary table ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  FINAL SUMMARY — all sensors")
    print(f"{'='*60}")
    print(f"  {'Sensor':<8}  {'RMSE':>10}  {'MAE':>10}  {'PHM-2012':>10}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")
    for s, m in all_metrics.items():
        print(f"  {s.upper():<8}  {m['rmse']:>10.6f}  {m['mae']:>10.6f}  {m['phm_score']:>10.4f}")
    print()

