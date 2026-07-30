"""EXP5 — one-factor-at-a-time hyper-parameter sensitivity.
PRIMARY_MODEL on a small set of representative LOSO folds. Each sweep varies a
single factor; all others stay at the submission defaults.
Outputs -> results_revision/sensitivity/<factor>/<name>/.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine, config_revision as C

BASE = os.path.join(engine.REPO_ROOT, C.RESULTS_SUBDIR, "sensitivity")
M, SCEN, SEED = C.SENS_MODEL, C.SENS_SCENARIO, C.SENS_SEED


def _run(factor, tag, value, t, **kw):
    out = os.path.join(BASE, factor, f"{factor}_{tag}_{M}_{t}")
    engine.train_one(out_dir=out, arch=M, mode=SCEN, target=t, seed=SEED, method="erm",
                     epochs=C.SMOKE_EPOCHS or C.EPOCHS, patience=C.PATIENCE,
                     extra_cols={"Group": "sensitivity", "Factor": factor, "Value": value},
                     **kw)


def run():
    for t in C.SENS_FOLDS:
        for res in C.SENS_RESOLUTIONS:
            _run("resolution", str(res or "native"), res or "native", t, crop=res)
        for b in C.SENS_BATCH_SIZES:
            _run("batch", str(b), b, t, batch=b)
        for lr in C.SENS_LEARNING_RATES:
            _run("lr", f"{lr:.0e}", lr, t, lr_base=lr)
        for name, p in C.SENS_AUG_INTENSITY.items():
            aug = dict(flip=True, rotation=True, jitter=True, erase=True,
                       jitter_mag=p["jitter"], rotation_deg=p["rotation"], erase_p=p["erase_p"])
            _run("aug_intensity", name, name, t, aug=aug)


if __name__ == "__main__":
    run()
