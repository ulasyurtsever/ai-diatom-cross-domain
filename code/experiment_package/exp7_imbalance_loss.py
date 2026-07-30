"""EXP7 - alternative imbalance-handling losses.
ConvNeXt-Tiny under LOSO with Focal loss (alpha-balanced, gamma=2) and plain CE,
compared against the existing Weighted-CE baseline (already in results_revision).
Seed 42 to match the DG baselines (exp2). Outputs -> results_revision/imbalance/<loss>/.
Run AFTER exp1 (the WCE baseline predictions must already exist).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine, config_revision as C

BASE = os.path.join(engine.REPO_ROOT, C.RESULTS_SUBDIR, "imbalance")
ARCH = C.PRIMARY_MODEL          # convnext_tiny
LOSSES = ["focal", "ce"]        # WCE baseline is reused from exp1 (multiseed seed_42)


def run():
    domains = engine.loso_domains()
    for loss in LOSSES:
        for t in domains:
            out = os.path.join(BASE, loss, f"{loss.upper()}_{ARCH}_{t}")
            engine.train_one(out_dir=out, arch=ARCH, mode="LOSO", target=t,
                             seed=C.DG_SEED, method="erm", loss=loss,
                             epochs=C.SMOKE_EPOCHS or C.EPOCHS, patience=C.PATIENCE,
                             extra_cols={"Group": f"imbalance_{loss}"})


if __name__ == "__main__":
    run()
