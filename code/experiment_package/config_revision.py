"""
Central configuration for the IEEE Access R1 revision experiment package.

Everything is controlled from here so the whole package can be run
non-interactively with a single command and produce every output the
manuscript revision requires. Edit the SCOPE lists to trade runtime for
coverage; defaults match the reported configuration.
"""

RESULTS_SUBDIR = "results_revision"   # outputs under <REPO_ROOT>/results_revision/

# Models (identical names/order to train.py)
ALL_MODELS = ["resnet50", "efficientnet_v2_s", "convnext_tiny",
              "swin_v2_t", "maxvit_t"]
PRIMARY_MODEL = "convnext_tiny"       # most robust model, used for lighter analyses

# EXP 1 — Multi-seed stability
MULTISEED_SEEDS = [42, 202, 1337]                     # >=3 seeds total (incl. 42)
MULTISEED_SCENARIOS = ["LOSO", "SINGLE", "STANDARD"]  # LOSO is the headline
MULTISEED_MODELS = ALL_MODELS

# EXP 2 — Dedicated DG baselines
DG_METHODS = ["coral", "mixstyle"]
DG_SCENARIO = "LOSO"
CORAL_MODELS = ALL_MODELS
MIXSTYLE_MODELS = ["resnet50", "convnext_tiny"]       # well-defined for CNNs
CORAL_LAMBDA = 0.5
MIXSTYLE_P = 0.5
MIXSTYLE_ALPHA = 0.1
DG_SEED = 42

# EXP 3 — Augmentation ablation
ABLATION_MODEL = PRIMARY_MODEL
ABLATION_SCENARIO = "LOSO"
ABLATION_SEED = 42
ABLATION_VARIANTS = {
    "full":             dict(flip=True,  rotation=True,  jitter=True,  erase=True),
    "no_augmentation":  dict(flip=False, rotation=False, jitter=False, erase=False),
    "geometric_only":   dict(flip=True,  rotation=True,  jitter=False, erase=False),
    "photometric_only": dict(flip=False, rotation=False, jitter=True,  erase=False),
    "no_random_erase":  dict(flip=True,  rotation=True,  jitter=True,  erase=False),
}

# EXP 4 — Inference / efficiency metrics  [no training]
INFER_MODELS = ALL_MODELS
INFER_BATCH_FOR_THROUGHPUT = 32
INFER_WARMUP = 20
INFER_ITERS = 100

# EXP 5 — Hyper-parameter sensitivity (one-factor-at-a-time)
SENS_MODEL = PRIMARY_MODEL
SENS_SCENARIO = "LOSO"
SENS_SEED = 42
SENS_FOLDS = ["LOIR_Database", "ADIAC_Database"]   # hardest + easiest target
SENS_RESOLUTIONS = [224, None]                     # None = backbone native size
SENS_BATCH_SIZES = [64, 128, 256]
SENS_LEARNING_RATES = [1e-4, 3e-4]
SENS_AUG_INTENSITY = {
    "light":   dict(jitter=0.1, rotation=7,  erase_p=0.25),
    "default": dict(jitter=0.2, rotation=15, erase_p=0.50),
    "strong":  dict(jitter=0.4, rotation=25, erase_p=0.75),
}

# Shared training budget (identical to original submission)
EPOCHS = 500
PATIENCE = 20
SMOKE_EPOCHS = None    # set to e.g. 2 ONLY for a quick smoke test

# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------
# * Per-run model checkpoints (best_model_*.pth) are saved for every run so you
#   can regenerate Grad-CAM / do inference later WITHOUT retraining. They are the
#   biggest disk consumer (~0.1-0.35 GB each; the full suite can reach tens of GB).
#   They are kept on the server and are NOT included in results_revision_export.tar.gz.
#   If disk is tight, you may delete checkpoints for the non-seed-42 runs after the
#   aggregation step; keep the seed-42 LOSO checkpoints if you might redo Grad-CAM.
