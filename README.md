# Cross-Domain Evaluation of Modern Deep Learning Architectures for Microscopic Diatom Classification

Reproducible benchmark and analysis code for a leave-one-source-out (LOSO) study
of cross-laboratory generalization in microscopic diatom classification. The
benchmark aggregates six heterogeneous public source databases and evaluates five
representative backbones (ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny,
Swin V2-Tiny, MaxViT-Tiny) under three protocols — Single-Source, Standard-Mixed,
and LOSO — with three-seed averaging, statistical testing, domain-generalization
baselines, ablations, and inference-efficiency measurements.

This repository releases the curated metadata, deterministic fold definitions,
per-experiment prediction files, statistical-analysis code, and figure-generation
code required to reproduce every reported number without image re-acquisition or
GPU access. The source images are redistributed from their original providers and
are reconstructed locally from the six public databases (see **Dataset**).

## Headline LOSO results (seed-averaged over {42, 202, 1337})

| Backbone | LOSO accuracy | LOSO Macro-F1 |
|---|---|---|
| ConvNeXt-Tiny | 0.632 ± 0.015 | 0.503 ± 0.013 |
| MaxViT-Tiny | 0.622 ± 0.009 | 0.515 ± 0.006 |
| EfficientNetV2-S | 0.607 ± 0.011 | 0.485 ± 0.014 |
| Swin V2-Tiny | 0.549 ± 0.002 | 0.431 ± 0.004 |
| ResNet-50 | 0.484 ± 0.012 | 0.377 ± 0.019 |

Values are means ± standard deviation across the six held-out source domains.
Full statistics (bootstrap confidence intervals, Friedman/Nemenyi, Wilcoxon with
Holm correction) are in `results/aggregated/`.

## Repository structure

```
.
├── README.md                     This file
├── LICENSE                       MIT
├── code/
│   ├── train.py                  Single-seed training over all (backbone, scenario) runs
│   ├── regen_figs_bigfont.py     Regenerates the manuscript plot figures from result CSVs
│   ├── curation_and_analysis/    Dataset curation, metadata, statistics, leakage audit
│   └── experiment_package/       Automated multi-seed / DG / ablation / sensitivity suite
├── data/                         Curated metadata, MD5 and pHash manifests, fold definitions
├── figures/                      Manuscript figures (vector PDF + PNG preview)
└── results/
    ├── aggregated/               Seed-averaged tables, statistics, post-hoc analyses
    ├── environment/              Captured software/hardware environment and pip freeze
    └── per_experiment_predictions.tar.gz   Per-run predictions, indices, confusion matrices
```

## Dataset

The Core Dataset comprises 12,353 curated images spanning 46 genera, drawn from
six source databases. The 27 genera present in at least five of the six sources
form the LOSO core used in the cross-domain protocol.

| Source | Images | Genera | Imaging modality |
|---|---|---|---|
| ADIAC | 2,594 | 38 | Bright-field |
| AFD | 1,148 | 31 | Bright-field / DIC mixed |
| DIA | 863 | 31 | Bright-field / DIC |
| DONA | 5,486 | 45 | Bright-field / DIC subset |
| FCE_LTER | 509 | 37 | Bright-field / occasional DIC |
| LOIR | 1,753 | 35 | Phase-contrast / DIC |

Images are not shipped with this repository. Reconstruct the
`Processed_Data_PNG/<Domain>/<Genus>/<file>.png` tree from the six public sources
and place it at the repository root; `data/diatom_metadata.csv` references those
relative paths and includes the deterministic `split` assignment. `data/md5_manifest.csv`
allows exact verification of every file.

## Environment

Install the pinned versions recorded under `results/environment/`
(`environment.json`, `pip_freeze.txt`). A single CUDA GPU is sufficient; the code
auto-detects the device and does not use DataParallel.

## Reproducing the results

All headline numbers are already provided in `results/aggregated/`
(`revision_numbers.json`, `REVISION_RESULTS_SUMMARY.md`, and the per-analysis
CSVs), so most tables and figures can be regenerated without retraining. The
per-experiment prediction files are shipped compressed; the analysis and
figure scripts read them from `results_revision/`, so extract the archive once
before running those scripts:

```bash
mkdir -p results_revision
tar xzf results/per_experiment_predictions.tar.gz -C results_revision   # -> results_revision/{multiseed,ablation,dg}/
cp -r results/aggregated results_revision/aggregated
```

The data-reading scripts locate the repository root automatically, so they can
be run from any directory (or given explicit `--metadata` / `--data-root`).

1. **Metadata and integrity** — `code/curation_and_analysis/diatom_metadata.py`
   builds the metadata and the stratified `(source, genus)` split; strata with
   fewer than five images are assigned entirely to training.
   `compute_md5_manifest.py` and `verify_dataset_integrity.py` verify the files.
2. **Base seed (42)** — run `code/train.py` from the repository root to train every
   (backbone, scenario) configuration for the base seed.
3. **Full suite** — `code/experiment_package/` orchestrates the additional seeds,
   the domain-generalization baselines, ablations, sensitivity analysis,
   inference-efficiency measurements, and post-hoc error analysis (see its README).
4. **Statistics** — `code/curation_and_analysis/compute_statistics.py` and
   `compute_matched_gap_and_stats.py` produce the Friedman/Nemenyi/Wilcoxon results
   and the label-matched cross-domain gap.
5. **Figures** — `code/regen_figs_bigfont.py` regenerates the plot figures from the
   aggregated CSVs.

## Hardware and training configuration

The reference configuration uses a single NVIDIA RTX 6000 Ada Generation (48 GB),
batch size 128, and the AdamW optimizer. The base learning rate of 1e-4 is scaled
linearly with the batch size (`batch/32`), giving an effective 4e-4 at batch 128.
Training uses `ReduceLROnPlateau` with early stopping. Each backbone is fed at its
native ImageNet-1K input resolution (224 px for ResNet-50, ConvNeXt-Tiny and
MaxViT-Tiny; 256 px for Swin V2-Tiny; 384 px for EfficientNetV2-S).

## Duplicate and leakage audit

`code/curation_and_analysis/phash_dedup_check.py` computes a 64-bit perceptual hash
for every image and reports exact and near-duplicate collisions. There are no
cross-source exact duplicates, so the source-level LOSO evaluation cannot share
images between training and test folds.

`code/curation_and_analysis/check_dup_split_leakage.py` extends this to the
image-level splits of the in-distribution Single-Source and Standard-Mixed
scenarios: it cross-references the residual within-source duplicate groups against
the `split` assignment and re-scores the affected test sets with duplicates
removed. The impact is quantified in `results/aggregated/leakage_audit_impact.csv`
(accuracy and Macro-F1 shift by at most 0.5 percentage points, leaving all
rankings unchanged). The per-image hashes and the affected groups are provided in
`data/phash_manifest.csv` and `data/dup_split_leakage_report.csv`.

## Figures

`figures/` contains the manuscript figures as vector PDF with a PNG preview.
Plot figures are regenerated deterministically from the result CSVs by
`code/regen_figs_bigfont.py`; the Grad-CAM panels are in `figures/figure6a-c.png`.

## Citation

Ulaş Yurtsever, "Cross-Domain Evaluation of Modern Deep Learning Architectures for
Microscopic Diatom Classification," *IEEE Access*. A BibTeX entry will be added on
publication.

## License

Released under the MIT License (see `LICENSE`). The source images remain subject
to the terms of their original providers.
