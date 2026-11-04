# models/

Trained models of the three sensors (dissertation, Section 5.4.3).

| File | Content |
|---|---|
| `model_<sensor>.pt` | PyTorch checkpoint: `model_state_dict`, `hparams`, `input_dim`, `time_steps` (60), ordered `feature_cols`, `label_info` (t_FPT, t_EoL, L in seconds), test `metrics` |
| `scaler_<sensor>.pkl` | scikit-learn `StandardScaler` fitted on the training blocks |
| `best_hyperparameters_<sensor>.pkl` | best Optuna trial (Table 5.4) |

Sensors: `adxl` (ADXL345, 33 features), `piezo` (piezoelectric, 11), `mpu` (MPU9250, 33).

```python
import torch, pickle, sys
sys.path.insert(0, "prognostics")
from model_common import CNN_BiLSTM
ck = torch.load("models/model_piezo.pt", map_location="cpu", weights_only=False)
hp = ck["hparams"]
model = CNN_BiLSTM(ck["input_dim"], hp["cnn_filters"], hp["cnn_kernel"],
                   hp["lstm_hidden"], hp["lstm_layers"], hp["dropout_rate"])
model.load_state_dict(ck["model_state_dict"]); model.eval()
scaler = pickle.load(open("models/scaler_piezo.pkl", "rb"))
# input: (batch, 60, n_features) of scaled features -> HI of the last window
```
