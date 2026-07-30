"""EXP1 — multi-seed stability.
Seed 42 = original submission (reused by the aggregator from results/per_experiment).
Here we run the ADDITIONAL seeds only. Outputs -> results_revision/multiseed/seed_<s>/.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine, config_revision as C

BASE = os.path.join(engine.REPO_ROOT, C.RESULTS_SUBDIR, "multiseed")
NEW_SEEDS = list(C.MULTISEED_SEEDS)   # run ALL seeds (incl. 42) on THIS machine
# (to instead reuse the original single RTX 6000 Ada Generation seed-42, set MULTISEED_SEEDS=[202,1337] in config)


def run():
    domains = engine.loso_domains()
    for seed in NEW_SEEDS:
        for arch in C.MULTISEED_MODELS:
            for scen in C.MULTISEED_SCENARIOS:
                if scen == "SINGLE":
                    targets = ["ADIAC_Database"]; name = lambda t: f"S0_Single_{arch}_ADIAC_Database"
                elif scen == "STANDARD":
                    targets = [None]; name = lambda t: f"S1_Standard_{arch}"
                else:
                    targets = domains; name = lambda t: f"S2_LOSO_{arch}_{t}"
                for t in targets:
                    out = os.path.join(BASE, f"seed_{seed}", name(t))
                    engine.train_one(out_dir=out, arch=arch, mode=scen, target=t,
                                     seed=seed, method="erm",
                                     epochs=C.SMOKE_EPOCHS or C.EPOCHS, patience=C.PATIENCE,
                                     extra_cols={"Group": "multiseed"})


if __name__ == "__main__":
    run()
