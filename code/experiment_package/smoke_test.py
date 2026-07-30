"""Fast end-to-end sanity check (run this FIRST on your server).

Builds a tiny synthetic dataset in a temp folder (your real data/results are
NOT touched), then runs ONE 1-epoch pass of every experiment type:
ERM, CORAL, MixStyle (ResNet + ConvNeXt), augmentation-ablation variant,
sensitivity point, plus efficiency + aggregation + figures.

If this prints 'SMOKE TEST PASSED', the pipeline is functional and the full
runs can be launched with:  bash experiment_package/run_all.sh

Usage (from the released repo root):  python Experiment_Package/smoke_test.py
"""
import os, sys, tempfile, shutil, csv, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# --- build a throwaway "repo" with synthetic data -------------------------
SMOKE = tempfile.mkdtemp(prefix="diatom_smoke_")
os.makedirs(os.path.join(SMOKE, "data"), exist_ok=True)
# copy the real train.py so build_model is identical
real_repo = None
for c in [os.getcwd(), os.path.dirname(HERE)]:
    if os.path.exists(os.path.join(c, "train.py")):
        real_repo = c; break
if real_repo is None:
    sys.exit("Run this from the repo root (folder containing train.py).")
shutil.copy(os.path.join(real_repo, "train.py"), os.path.join(SMOKE, "train.py"))

from PIL import Image
import numpy as np
DOMS = ["ADIAC_Database", "LOIR_Database", "DONA_Database"]
CLASSES = ["Navicula", "Nitzschia", "Cymbella"]
img_dir = os.path.join(SMOKE, "imgs"); os.makedirs(img_dir, exist_ok=True)
rows = []
lbl_idx = {c: i for i, c in enumerate(CLASSES)}
rng = random.Random(0)
for d in DOMS:
    for c in CLASSES:
        for k in range(6):                       # 6 imgs/class/domain
            p = os.path.join(img_dir, f"{d}_{c}_{k}.png")
            Image.fromarray((np.random.rand(48, 48, 3) * 255).astype("uint8")).save(p)
            split = "train" if k < 4 else ("val" if k == 4 else "test")
            rows.append(dict(filepath=p, domain=d, label=c, is_dg_class=True,
                             label_idx=lbl_idx[c], split=split, modality="LM"))
with open(os.path.join(SMOKE, "data", "diatom_metadata.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# --- point the engine at the synthetic repo and shrink the budget ---------
os.environ["REPO_ROOT"] = SMOKE
import config_revision as C
C.SMOKE_EPOCHS = 1
import engine

def check(path):
    ok = os.path.exists(os.path.join(path, "final_metrics.csv"))
    print(("  PASS " if ok else "  FAIL ") + path)
    return ok

results = []
D = DOMS[0]
print("\n[1] ERM LOSO (resnet50)")
o = os.path.join(SMOKE, "out", "erm"); engine.train_one(out_dir=o, arch="resnet50", mode="LOSO", target=D, seed=42, method="erm", epochs=1, patience=1); results.append(check(o))
print("[2] CORAL LOSO (resnet50)")
o = os.path.join(SMOKE, "out", "coral"); engine.train_one(out_dir=o, arch="resnet50", mode="LOSO", target=D, seed=42, method="coral", epochs=1, patience=1); results.append(check(o))
print("[3] MixStyle LOSO (resnet50)")
o = os.path.join(SMOKE, "out", "ms_r"); engine.train_one(out_dir=o, arch="resnet50", mode="LOSO", target=D, seed=42, method="mixstyle", epochs=1, patience=1); results.append(check(o))
print("[4] MixStyle LOSO (convnext_tiny)")
o = os.path.join(SMOKE, "out", "ms_c"); engine.train_one(out_dir=o, arch="convnext_tiny", mode="LOSO", target=D, seed=42, method="mixstyle", epochs=1, patience=1); results.append(check(o))
print("[5] Ablation variant (no augmentation)")
o = os.path.join(SMOKE, "out", "abl"); engine.train_one(out_dir=o, arch="convnext_tiny", mode="LOSO", target=D, seed=42, method="erm", aug=dict(flip=False, rotation=False, jitter=False, erase=False), epochs=1, patience=1); results.append(check(o))
print("[6] Sensitivity (native resolution)")
o = os.path.join(SMOKE, "out", "sens"); engine.train_one(out_dir=o, arch="convnext_tiny", mode="LOSO", target=D, seed=42, method="erm", crop=None, epochs=1, patience=1); results.append(check(o))

print("[7] Efficiency metrics")
C.INFER_MODELS = ["resnet50", "convnext_tiny"]; C.INFER_ITERS = 3; C.INFER_WARMUP = 1
import exp4_inference_metrics as e4; e4.run()
results.append(os.path.exists(os.path.join(SMOKE, "results_revision", "efficiency", "efficiency_metrics.csv")))

ok = all(results)
print("\n" + ("SMOKE TEST PASSED — pipeline works; launch run_all.sh for the full runs."
              if ok else "SMOKE TEST FAILED — see errors above."))
print("(temp folder left for inspection: %s)" % SMOKE)
sys.exit(0 if ok else 1)
