# Alternative imbalance-handling losses

Compares the Weighted Cross-Entropy (WCE) baseline against two alternative
imbalance-handling losses by training ConvNeXt-Tiny over the six LOSO folds with
**Focal loss** (alpha-balanced, gamma = 2) and **plain Cross-Entropy**, then
comparing both against WCE. Uses the base seed (42) and the same configuration as
the domain-generalization baselines (`exp2`).

## Components

The following files provide this experiment (the rest of the package is unchanged):

- `engine.py` — adds `FocalLoss` and a backward-compatible `loss=` switch; the
  `torch < 2.4` GradScaler compatibility patch is included. The default remains
  `loss="wce"`, so `exp1`–`exp6` behave identically.
- `exp7_imbalance_loss.py`
- `aggregate_imbalance.py`

## Run

Run from the repository root (the folder that contains `train.py`, `data/`, and
`Processed_Data_PNG/`); image paths in the metadata are relative to it.

```bash
cd <repo-root>
python experiment_package/exp7_imbalance_loss.py   # 12 runs: {focal, ce} x 6 LOSO folds
python experiment_package/aggregate_results.py      # rebuilds all aggregated CSVs (incl. imbalance)
python experiment_package/aggregate_imbalance.py    # optional: print just this table
```

Approximate wall-clock: ~3-5 h on the reference NVIDIA RTX 6000 Ada Generation
(12 ConvNeXt-Tiny LOSO trainings with early stopping). Runs are resumable — a fold
that already has `final_metrics.csv` is skipped.

## Output

- Predictions: `results_revision/imbalance/{focal,ce}/<name>/test_predictions.csv`
- Comparison table: `results_revision/aggregated/imbalance_loss_LOSO.csv`

`aggregate_imbalance.py` reports, for each loss, the mean LOSO accuracy and
Macro-F1 over the six held-out domains and the change relative to the WCE baseline.
