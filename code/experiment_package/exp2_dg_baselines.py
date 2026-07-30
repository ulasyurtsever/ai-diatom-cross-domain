"""EXP2 — dedicated domain-generalization baselines.
CORAL (all backbones) + MixStyle (CNN backbones) under LOSO.
Outputs -> results_revision/dg/<method>/<name>/.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine, config_revision as C

BASE = os.path.join(engine.REPO_ROOT, C.RESULTS_SUBDIR, "dg")


def run():
    domains = engine.loso_domains()
    if "coral" in C.DG_METHODS:
        for arch in C.CORAL_MODELS:
            for t in domains:
                out = os.path.join(BASE, "coral", f"CORAL_{arch}_{t}")
                engine.train_one(out_dir=out, arch=arch, mode="LOSO", target=t,
                                 seed=C.DG_SEED, method="coral",
                                 coral_lambda=C.CORAL_LAMBDA,
                                 epochs=C.SMOKE_EPOCHS or C.EPOCHS, patience=C.PATIENCE,
                                 extra_cols={"Group": "dg_coral"})
    if "mixstyle" in C.DG_METHODS:
        for arch in C.MIXSTYLE_MODELS:
            for t in domains:
                out = os.path.join(BASE, "mixstyle", f"MixStyle_{arch}_{t}")
                engine.train_one(out_dir=out, arch=arch, mode="LOSO", target=t,
                                 seed=C.DG_SEED, method="mixstyle",
                                 mixstyle_p=C.MIXSTYLE_P, mixstyle_alpha=C.MIXSTYLE_ALPHA,
                                 epochs=C.SMOKE_EPOCHS or C.EPOCHS, patience=C.PATIENCE,
                                 extra_cols={"Group": "dg_mixstyle"})


if __name__ == "__main__":
    run()
