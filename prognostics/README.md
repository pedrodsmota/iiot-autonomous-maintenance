# prognostics/

Offline pipeline of the article, in order of execution:

| Step | Script | Writes |
|---|---|---|
| 1 | `sync_data.py` | `data/synchronized/` |
| 2 | `extract_features.py` | `data/features/<sensor>_features.csv` |
| 3 | `fuse_features.py` | `data/features/<sensor>_features_pca.csv` (per-sensor PCA) |
| 4 | `construct_hs_hi.py` | `data/features/<sensor>_training_dataset_with_targets.csv` |
| 5 | `dataset_builder.py` | `<sensor>_dataset.pkl`, `scaler_<sensor>.pkl` |
| 6 | `tune_hyperparameters.py` | `best_hyperparameters_<sensor>.pkl` |
| 7 | `train_final_model.py` | `results/model_<sensor>.pt` |
| - | `evaluate_article_models.py` | RMSE and MAE of the published models (prints) |

Set `IIOT_DATA_DIR` to change the data location (default `../data`).

## Running it from the published dataset

`sync_data.py` reads CSV files from `data/raw_csv/<sensor folder>/`, with the piezoelectric recordings **already multiplied** by their calibration factor. The dataset stores them as recorded, in Parquet, so before step 1:

1. Convert them to CSV with [`parquet_to_csv.py`](https://github.com/pedrodsmota/iiot-autonomous-maintenance/blob/main/dataset/parquet_to_csv.py).
2. Multiply `Vibration_Val` of sessions 15-22 by `piezo_calibration_factor` from `metadata/sessions.csv`.

`tune_hyperparameters.py` and `train_final_model.py` have the list of sensors of their last run in the code (`sensors`, `SENSORS`): edit it to run the others. `train_final_model.py` needs a `best_hyperparameters_<sensor>.pkl`; those of the article are stored in the `hparams` of each checkpoint in `../models/`.
