# Experiment package (automated pipeline)

This folder orchestrates the full experimental suite of the study. It is placed
inside the repository root (the folder that contains `train.py`,
`data/diatom_metadata.csv`, `code/`, and `results/`) and executed with a single
command. All new outputs are written under `results_revision/`; the original
`results/` directory is not modified.

```bash
# 0) optional sanity check (~1-2 min, uses a throwaway synthetic set)
python experiment_package/smoke_test.py      # prints: SMOKE TEST PASSED

# 1) run the full suite (logs to results_revision/logs/)
bash experiment_package/run_all.sh
```

The driver does not abort the whole run if a single step fails, so one pass
produces the maximum amount of output. Re-running skips any experiment whose
`final_metrics.csv` already exists, so the suite is safe to resume.

## Components

| Script | Purpose | Output |
|---|---|---|
| `exp4_inference_metrics.py` | Inference efficiency: GPU memory, latency, throughput, parameters, FLOPs, model size | `results_revision/efficiency/efficiency_metrics.csv` |
| `exp1_multiseed.py` | Multi-seed (≥3) stability of metrics and rankings | `results_revision/multiseed/seed_*/` |
| `exp2_dg_baselines.py` | Domain-generalization baselines (CORAL, MixStyle) versus ERM | `results_revision/dg/{coral,mixstyle}/` |
| `exp3_ablation.py` | Augmentation-component ablation | `results_revision/ablation/<variant>/` |
| `exp5_sensitivity.py` | One-factor sensitivity (resolution / batch / learning rate / augmentation) | `results_revision/sensitivity/<factor>/` |
| `exp6_posthoc_analysis.py` | Post-hoc error analysis (no training) | `results_revision/aggregated/posthoc/` |
| `exp7_imbalance_loss.py` | Imbalance-handling losses (Weighted CE, Focal, plain CE) | `results_revision/imbalance/` |
| `capture_env.py` | Software/hardware environment, `pip freeze`, git commit | `results_revision/environment/` |
| `aggregate_results.py` | Consolidates all analyses into publication tables | `results_revision/aggregated/` |
| `make_revision_figures.py` | Publication figures | `results_revision/aggregated/figures/` |

The base seed (42) is trained by `train.py` at the repository root; the aggregator
reuses those outputs automatically from `results/per_experiment/`, and only the
additional seeds are trained here.

## Consolidated outputs

`results_revision/aggregated/REVISION_RESULTS_SUMMARY.md` collects every headline
number as Markdown tables; `revision_numbers.json` holds the same data in
machine-readable form. Per-analysis CSVs are written alongside them.

## Configuration

All scope is controlled from `config_revision.py`, including:

- `MULTISEED_SEEDS = [42, 202, 1337]` and `MULTISEED_SCENARIOS = ["LOSO", "SINGLE", "STANDARD"]`;
- sensitivity grids `SENS_BATCH_SIZES = [64, 128, 256]` and `SENS_LEARNING_RATES = [1e-4, 3e-4]`;
- ablation and sensitivity run on the primary backbone (`convnext_tiny`), sensitivity on two representative held-out folds.

## Reference runtime

Approximate wall-clock on a single 48 GB CUDA GPU (reference:
NVIDIA RTX 6000 Ada Generation), training to ~500 epochs with early stopping:

| Step | Trainings | Approx. time |
|---|---|---|
| efficiency | 0 (inference) | minutes |
| multi-seed | ~80 | ~12-20 h |
| DG baselines | ~42 | ~7-12 h |
| ablation | ~30 | ~5-9 h |
| sensitivity | ~20 | ~3-6 h |

The full suite is roughly 1.5-2 days of GPU time and resumes safely, so it can be
run in chunks.

## Requirements

The same PyTorch/torchvision environment as the main training run, plus optional
`fvcore` **or** `thop` for FLOPs (efficiency metrics degrade gracefully to `NaN`
if neither is installed):

```bash
pip install fvcore    # or: pip install thop
```

## Domain-generalization baselines

- **CORAL** (Sun & Saenko, 2016): aligns second-order feature statistics across
  the source domains present in each mini-batch (no target access); evaluated on
  all five backbones, on a single GPU so that per-batch domain grouping is exact.
- **MixStyle** (Zhou et al., 2021): mixes instance-level feature statistics in the
  early stages; defined for the convolutional backbones (ResNet-50, ConvNeXt-Tiny).

Both are trained under the identical LOSO protocol and budget as the ERM baseline.

---

## Dataset layout and single-GPU execution

### 1. Folder layout (run all commands from the repository root)

The metadata uses **relative** image paths
(`Processed_Data_PNG/<Domain>/<Genus>/<file>.png`), so no path editing is
required; the `Processed_Data_PNG/` tree must sit at the repository root:

```
<repo-root>/
├── train.py
├── data/diatom_metadata.csv     (12,353 rows; filepaths relative to this folder)
├── Processed_Data_PNG/
│   ├── ADIAC_Database/<Genus>/*.png
│   ├── AFD_Database/ ...
│   ├── DIA_Database/ ...  DONA_Database/ ...  FCE_LTER_Database/ ...  LOIR_Database/ ...
├── code/   results/   experiment_package/
```

Integrity check before launching:

```bash
python - <<'PY'
import pandas as pd, os
d = pd.read_csv("data/diatom_metadata.csv")
missing = [p for p in d["filepath"] if not os.path.exists(p)]
print(f"{len(d)-len(missing)}/{len(d)} images found; missing={len(missing)}")
PY
```

### 2. Environment

Use the pinned PyTorch/torchvision versions from `results/environment/`. A single
GPU is sufficient; the code auto-detects the device and does not use DataParallel.

### 3. Batch size and learning rate

The reference configuration uses **batch size 128** on a single
NVIDIA RTX 6000 Ada Generation (48 GB). The learning rate is scaled with the batch
size (`1e-4 × 128/32 = 4e-4`). To keep the additional seeds directly comparable to
the base seed, use the same batch size:

```bash
export DIATOM_BATCH=128
export DIATOM_WORKERS=8
```

On memory-limited hardware, reduce `DIATOM_BATCH` (e.g. 96 or 64) for the largest
configurations (EfficientNetV2-S at 384 px, MaxViT); note that this changes the
scaled learning rate and reduces exact comparability across seeds.

### 4. Launch (long job)

```bash
cd <repo-root>
export DIATOM_BATCH=128 DIATOM_WORKERS=8
python experiment_package/smoke_test.py          # ~1-2 min -> SMOKE TEST PASSED
tmux new -s diatom                               # or: nohup ... &
bash experiment_package/run_all.sh 2>&1 | tee run_all.console.log
```

All outputs land in `results_revision/` and the run is resume-safe if interrupted.

---

## Full-provenance export (regenerate any table or figure offline)

The suite is run from scratch and saves everything needed to reproduce any
downstream table or figure without retraining:

- **Image-traceable predictions** — every run writes `test_index.csv`
  (`filepath, domain, label, True_Label, Predicted_Label, MaxProb, Correct`), so
  each prediction maps back to the exact image (failure cases, Grad-CAM candidates).
- **Post-hoc analyses** (`exp6_posthoc_analysis.py`, no training) →
  `results_revision/aggregated/posthoc/`: per-genus precision/recall/F1,
  most-confused genus pairs (global and per-domain), failure buckets with example
  image paths, and extended summary metrics (balanced accuracy, micro-F1).
- **Single export archive** (`export_all.py`) → `results_revision_export.tar.gz` at
  the repository root: all metrics, predictions, indices, confusion matrices,
  per-epoch logs, aggregated tables, post-hoc CSVs, figures, and the environment
  capture. Model checkpoints (`*.pth`) are excluded to keep the archive small.
  This archive is sufficient to regenerate every manuscript table and figure
  without GPU access.

`run_all.sh` invokes the post-hoc and export steps automatically.
