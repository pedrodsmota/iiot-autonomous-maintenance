# models/

The models deployed in the end-to-end trials of the article, and their scalers.

| File | Content |
|---|---|
| `model_<sensor>.pt` | PyTorch checkpoint: `model_state_dict`, `hparams` (the hyperparameters used), `input_dim`, `time_steps`, `sensor`, `metrics` |
| `scaler_<sensor>.pkl` | scikit-learn `StandardScaler` |

**The `metrics` inside the checkpoints are not the values in the article.** They were computed during training, on an earlier build of the dataset: RMSE 0.0334 / 0.0456 / 0.0838 for the ADXL345, piezoelectric, and MPU9250 models. See [`../results/`](../results/).
