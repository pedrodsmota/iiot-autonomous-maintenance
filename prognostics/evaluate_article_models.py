"""
Evaluate the trained models of the article on the test sequences (Table 4, CNN-BiLSTM rows).

The models were not retrained for the article's final evaluation. The metrics stored inside
the checkpoints were computed during training, on an earlier build of the dataset, so they
differ from the article for the piezoelectric and MPU9250 models.

Reads   models/model_<sensor>.pt, prognostics/<sensor>_dataset.pkl (from dataset_builder.py)
Prints  RMSE and MAE per sensor

Usage (from the prognostics folder):  python evaluate_article_models.py
"""

import pickle
from pathlib import Path

import numpy as np
import torch

from train_final_model import CNN_BiLSTM

HERE = Path(__file__).resolve().parent
DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

for sensor in ("adxl", "piezo", "mpu"):
    with open(HERE / f"{sensor}_dataset.pkl", "rb") as f:
        dataset = pickle.load(f)
    checkpoint = torch.load(HERE.parent / "models" / f"model_{sensor}.pt",
                            map_location="cpu", weights_only=False)
    h = checkpoint["hparams"]
    model = CNN_BiLSTM(checkpoint["input_dim"], h["cnn_filters"], h["cnn_kernel"],
                       h["lstm_hidden"], h["lstm_layers"], h["dropout_rate"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()

    X, y = dataset["X_test"], np.asarray(dataset["y_test"]).ravel()
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            batch = torch.tensor(X[i:i + 512], dtype=torch.float32).to(DEVICE)
            predictions.append(model(batch).cpu().numpy().ravel())
    p = np.clip(np.concatenate(predictions), 0.0, 1.0)   # the HI is defined between 0 and 1
    print(f"{sensor:<6} {len(y):>6} sequences | RMSE {np.sqrt(((p - y) ** 2).mean()):.4f} "
          f"| MAE {np.abs(p - y).mean():.4f}")
