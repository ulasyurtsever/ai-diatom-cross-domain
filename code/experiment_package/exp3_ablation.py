"""EXP3 — augmentation ablation.
PRIMARY_MODEL under LOSO with one augmentation family toggled at a time.
Outputs -> results_revision/ablation/<variant>/<name>/.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine, config_revision as C

BASE = os.path.join(engine.REPO_ROOT, C.RESULTS_SUBDIR, "ablation")


def run():
    domains = engine.loso_domains()
    for variant, flags in C.ABLATION_VARIANTS.items():
        for t in domains:
            out = os.path.join(BASE, variant, f"{variant}_{C.ABLATION_MODEL}_{t}")
            engine.train_one(out_dir=out, arch=C.ABLATION_MODEL, mode=C.ABLATION_SCENARIO,
                             target=t, seed=C.ABLATION_SEED, method="erm", aug=flags,
                             epochs=C.SMOKE_EPOCHS or C.EPOCHS, patience=C.PATIENCE,
                             extra_cols={"Group": "ablation", "Variant": variant})


if __name__ == "__main__":
    run()
