"""
Model and sequence loading shared by the tuning, the training, and the evaluation
(dissertation, Section 5.4.3).

CNN_BiLSTM: one-dimensional convolution (same padding) + ReLU + max pooling that
halves the sequence, followed by a bidirectional LSTM whose output at the last
time step goes through dropout and a linear layer that outputs the HI of the
last window. The layer names are those of the checkpoints in models/, which the
online scripts also load.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CNN_BiLSTM(nn.Module):
    def __init__(self, input_dim, cnn_filters, cnn_kernel, lstm_hidden, lstm_layers, dropout):
        super().__init__()
        self.cnn = nn.Conv1d(input_dim, cnn_filters, cnn_kernel, padding="same")
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):                      # x: (batch, time_steps, features)
        x = self.relu(self.cnn(x.transpose(1, 2)))
        x = self.pool(x).transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :])).squeeze(-1)


class SequenceDataset(Dataset):
    """Sequences of T windows built from their start indices; the target is the
    HI of the last window. The sequences are not materialized in memory."""

    def __init__(self, X_scaled, y, starts, T):
        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.starts = np.asarray(starts, dtype=np.int64)
        self.T = T

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        return self.X[s:s + self.T], self.y[s + self.T - 1]


def predict(model, dataset, device, batch_size=1024):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for bx, by in loader:
            preds.append(model(bx.to(device)).cpu().numpy())
            trues.append(by.numpy())
    return np.concatenate(preds), np.concatenate(trues)
