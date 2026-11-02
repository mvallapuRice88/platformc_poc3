# Platform C PoC3 telemetry anomaly benchmark

This repository contains two offline research workflows for the PoC3 COSMOS telemetry files. It does not issue spacecraft commands, automate mitigation, or classify confirmed cyberattacks.

## Workflows

### Multi-model pipeline: `run.py`

Reads the files in `data/`, preprocesses telemetry, and trains/evaluates:

- Isolation Forest
- Gradient Boosting
- XGBoost
- LSTM
- MSGMamba-inspired GRU baseline (not a reproduction of the published MSGMamba architecture)

Run from the project directory:

```powershell
python run.py --data-dir data --output-dir outputs
```

Useful options include `--epochs`, `--window`, `--fpr`, and `--seed`. Results are written to `outputs/model_metrics.csv`.

This pipeline creates chronological 60/20/20 train/calibration/test splits for each data group. It fits preprocessing on normal rows in the training splits. Neural models use the configured number of epochs and run on CUDA when available, otherwise CPU.

### Statistical baseline: `src/run.py`

This standalone workflow scores telemetry using Robust Z-score, Isolation Forest, and PCA reconstruction error. It optionally includes a dense autoencoder and LSTM autoencoder with `--deep`.

```powershell
python src/run.py `
  --nominal data\2026_10_23_02_07_13_tlm_poc3_nominal.txt `
  --set1 data\2026_10_24_20_57_44_tlm_poc3_anomaly_set1.txt `
  --set2 data\2026_10_25_13_08_33_tlm_poc3_anomaly_set2.txt `
  --out outputs
```

Use `--deep` to include both neural autoencoders, and `--epochs` to set their training duration. The default startup cutoff is 320 seconds; change it with `--start`.

This workflow trains on the first 70% of nominal telemetry and reserves the remaining 30% as a nominal holdout. It writes `metrics.csv`, `scores.csv`, and `metadata.json` under the selected output directory.

## Setup on Windows

Run these commands from the project directory:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The requirements file includes the dependencies for both workflows. The supplied `requirements-deep.txt` contains PyTorch only; PyTorch is already listed in `requirements.txt`.

## Input data and labels

Place the three COSMOS TSV exports in `data/`:

- `2026_10_23_02_07_13_tlm_poc3_nominal.txt`
- `2026_10_24_20_57_44_tlm_poc3_anomaly_set1.txt`
- `2026_10_25_13_08_33_tlm_poc3_anomaly_set2.txt`

The parser handles the source `.bin` path at the start of the file, an optional blank line before the tab-separated header, repeated packet field names, and sparse packet columns. It uses `CCSDS_SECONDS` to align telemetry packet streams.

These exports do not contain explicit row-level ground-truth labels. For the anomaly sets, both workflows use **any EPS `SWITCH_n_STATE` equal to ON as a proxy anomaly label**. The nominal file is treated as normal. These are switch-state proxy labels, not verified event annotations or evidence of an attack.

## Interpretation limits

- An anomaly score or proxy label does not establish attacker intent, attack origin, or cyber classification.
- The 98th-percentile score thresholds are research settings, not validated operational limits.
- Independent runs are not matched counterfactuals; telemetry differences alone do not prove causation.
- The results do not provide calibrated confidence, command provenance, SPARTA mapping, root-cause analysis, operator review, or ODC integration.
- Evaluate event-level recall and false alerts per nominal hour before drawing further conclusions.
- Use only authorized offline test data.
