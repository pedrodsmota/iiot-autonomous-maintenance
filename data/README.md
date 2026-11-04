# data/

Local data folder (not tracked by git). Download the dataset ([data card](../dataset/README.md)), from this folder:

```bash
# Zenodo (3.8 GB; each zip can be downloaded on its own)
for f in ARM1_adxl345 ARM2_piezo ARM3_mpu9250 features; do
  curl -L -o $f.zip "https://zenodo.org/records/23900129/files/$f.zip?download=1" && unzip $f.zip && rm $f.zip
done
mkdir -p metadata && curl -L -o metadata/sessions.csv "https://zenodo.org/records/23900129/files/sessions.csv?download=1"

# or Kaggle (needs a Kaggle API token)
kaggle datasets download pedrodesousamota/run-to-failure-vibration-of-3d-printed-grippers --unzip
```

It should look like this (if Kaggle places the files in a subfolder, move its contents here):

```
data/
├── raw/
│   ├── ARM1_adxl345/trial_<n>.parquet
│   ├── ARM2_piezo/trial_<n>[_<part>].parquet
│   └── ARM3_mpu9250/trial_<n>.parquet
├── features/<sensor>_features.csv   (lets you skip steps 1 and 2 of the pipeline)
└── metadata/sessions.csv
```

A copy of `sessions.csv` is also in `../dataset/metadata/`, which the pipeline uses if
`data/metadata/` is missing.

The pipeline in `../prognostics/` writes its intermediate outputs here too
(`synchronized/`, `features/`, `processed/`). Set `IIOT_DATA_DIR` to use another location.
