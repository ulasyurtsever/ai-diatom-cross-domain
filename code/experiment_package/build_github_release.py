"""Assemble a COMPLETE, revision-consistent GitHub release folder so that every
number/figure in the revised manuscript is reproducible and nothing is stale.

Run AFTER: exp1-7 + aggregate_results.py + exp6_posthoc + regen_figs_bigfont.py.
It is safe to run at any time (idempotent) - missing pieces are simply skipped
and reported, so you can see exactly what is present vs still pending.

Output: 02_Revision_R1/github_release/  (replaces the stale GitHub_Repo_Ready).
"""
import os, sys, glob, shutil, tarfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, config_revision as C

PKG   = os.path.dirname(os.path.abspath(__file__))          # .../Experiment_Package
REV   = os.path.dirname(PKG)                                # .../02_Revision_R1
PROJ  = os.path.dirname(REV)                                # project root
RR    = os.path.join(REV, C.RESULTS_SUBDIR)                 # results_revision
OLD   = os.path.join(PROJ, "GitHub_Repo_Ready")            # legacy scripts to reuse
SUB   = os.path.join(REV, "submission_v22")
OUT   = os.path.join(REV, "github_release")

FIG_NAMES = ["figure1.png","figure2.png","figure3.png","figure4.png","figure5.png",
             "figure6a.png","figure6b.png","figure6c.png","figure7.png"]
report = []

def _log(msg): report.append(msg); print(msg)
def _ensure(d): os.makedirs(d, exist_ok=True)

def copy_file(src, dst_dir, rename=None):
    if not os.path.exists(src):
        _log(f"  [skip - missing] {os.path.relpath(src, PROJ)}"); return False
    _ensure(dst_dir)
    shutil.copy2(src, os.path.join(dst_dir, rename or os.path.basename(src)))
    return True

def _iglob(pattern):
    """glob that is safe when the path contains [ ] (our folder name has them)."""
    d, pat = os.path.split(pattern)
    return glob.glob(os.path.join(glob.escape(d), pat))

def copy_glob(pattern, dst_dir, label):
    n = 0
    for f in _iglob(pattern):
        if os.path.isfile(f):
            _ensure(dst_dir); shutil.copy2(f, os.path.join(dst_dir, os.path.basename(f))); n += 1
    _log(f"  {label}: {n} files")
    return n

def copy_tree(src, dst, label):
    if not os.path.isdir(src):
        _log(f"  [skip - missing] {label} ({os.path.relpath(src, PROJ)})"); return
    if os.path.isdir(dst): shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.pth", ".DS_Store"))
    n = sum(len(fs) for _, _, fs in os.walk(dst))
    _log(f"  {label}: {n} files")

def main():
    if os.path.isdir(OUT): shutil.rmtree(OUT)
    _ensure(OUT)
    _log("== building github_release ==")

    # 1) CODE ---------------------------------------------------------------
    _log("[code]")
    copy_glob(os.path.join(PKG, "*.py"),  os.path.join(OUT, "code", "experiment_package"), "experiment_package/*.py")
    copy_glob(os.path.join(PKG, "*.md"),  os.path.join(OUT, "code", "experiment_package"), "experiment_package/*.md")
    copy_glob(os.path.join(PKG, "*.sh"),  os.path.join(OUT, "code", "experiment_package"), "experiment_package/*.sh")
    copy_file(os.path.join(REV, "code", "regen_figs_bigfont.py"), os.path.join(OUT, "code"))
    copy_file(os.path.join(REV, "train.py"), os.path.join(OUT, "code"))
    # legacy curation / statistics / grad-cam scripts (still valid, dataset-level)
    copy_tree(os.path.join(OLD, "code"), os.path.join(OUT, "code", "curation_and_analysis"), "curation_and_analysis")

    # 2) DATA ---------------------------------------------------------------
    _log("[data]")
    copy_glob(os.path.join(REV, "data", "*.csv"), os.path.join(OUT, "data"), "data/*.csv")

    # 3) FIGURES (manuscript-consistent, big-font) --------------------------
    _log("[figures]")
    got = [copy_file(os.path.join(SUB, fn), os.path.join(OUT, "figures")) for fn in FIG_NAMES]
    _log(f"  manuscript figures: {sum(got)}/{len(FIG_NAMES)}")

    # 4) RESULTS ------------------------------------------------------------
    _log("[results]")
    copy_tree(os.path.join(RR, "aggregated"),   os.path.join(OUT, "results", "aggregated"),   "aggregated")
    copy_tree(os.path.join(RR, "environment"),  os.path.join(OUT, "results", "environment"),  "environment")
    # MANIFEST of every run
    rows = []
    for f in glob.glob(os.path.join(glob.escape(RR), "**", "final_metrics.csv"), recursive=True):
        try:
            r = pd.read_csv(f).iloc[0].to_dict(); r["path"] = os.path.relpath(f, RR); rows.append(r)
        except Exception:
            pass
    if rows:
        _ensure(os.path.join(OUT, "results"))
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "results", "MANIFEST_all_runs.csv"), index=False)
        _log(f"  MANIFEST_all_runs.csv: {len(rows)} runs")
    # bulky per-experiment prediction CSVs -> one archive
    groups = ["multiseed", "dg", "ablation", "sensitivity", "efficiency", "imbalance"]
    arch = os.path.join(OUT, "results", "per_experiment_predictions.tar.gz")
    _ensure(os.path.dirname(arch)); n = 0
    with tarfile.open(arch, "w:gz") as tar:
        for g in groups:
            gdir = os.path.join(RR, g)
            if not os.path.isdir(gdir): continue
            for root, _, files in os.walk(gdir):
                for fn in files:
                    if fn.endswith(".pth"): continue
                    fp = os.path.join(root, fn)
                    tar.add(fp, arcname=os.path.relpath(fp, RR)); n += 1
    _log(f"  per_experiment_predictions.tar.gz: {n} files ({os.path.getsize(arch)/1048576:.1f} MB)")

    # 5) ROOT FILES ---------------------------------------------------------
    _log("[root]")
    copy_file(os.path.join(REV, "requirements.txt"), OUT)
    copy_file(os.path.join(OLD, "LICENSE"), OUT)
    write_readme()

    # 6) COMPLETENESS CHECK -------------------------------------------------
    _log("\n== completeness check ==")
    checks = {
        "aggregated/multiseed_LOSO_mean_std.csv": os.path.join(OUT,"results","aggregated","multiseed_LOSO_mean_std.csv"),
        "aggregated/dg_comparison_LOSO_F1.csv":  os.path.join(OUT,"results","aggregated","dg_comparison_LOSO_F1.csv"),
        "aggregated/ablation_LOSO.csv":          os.path.join(OUT,"results","aggregated","ablation_LOSO.csv"),
        "aggregated/sensitivity.csv":            os.path.join(OUT,"results","aggregated","sensitivity.csv"),
        "aggregated/efficiency_metrics.csv":     os.path.join(OUT,"results","aggregated","efficiency_metrics.csv"),
        "aggregated/imbalance_loss_LOSO.csv":    os.path.join(OUT,"results","aggregated","imbalance_loss_LOSO.csv"),
        "aggregated/revision_numbers.json":      os.path.join(OUT,"results","aggregated","revision_numbers.json"),
        "posthoc (per-genus/confusions)":        os.path.join(OUT,"results","aggregated","posthoc"),
        "all 9 manuscript figures":              (sum(got)==len(FIG_NAMES)),
    }
    for k, v in checks.items():
        ok = v if isinstance(v, bool) else os.path.exists(v)
        _log(f"  [{'OK ' if ok else 'PENDING'}] {k}")
    open(os.path.join(OUT, "BUILD_REPORT.txt"), "w").write("\n".join(report))
    _log(f"\n-> {OUT}")

def write_readme():
    txt = """# Cross-Domain Evaluation of Modern Deep Learning Architectures for Microscopic Diatom Classification

Reproducibility package for the IEEE Access manuscript (revision).
Every number, table and figure in the manuscript is reproducible from the
artifacts here **without retraining** (trained checkpoints stay on the server;
only per-fold predictions are needed for all downstream tables/figures).

## Layout
- `code/experiment_package/` - the full revision pipeline: `engine.py`,
  `exp1_multiseed.py` ... `exp7_imbalance_loss.py`, `aggregate_results.py`,
  `aggregate_imbalance.py`, `make_revision_figures.py`, `config_revision.py`,
  `capture_env.py`, `run_all.sh`, `smoke_test.py`, setup notes.
- `code/regen_figs_bigfont.py` - regenerates manuscript Figures 2-5,7 (large fonts).
- `code/curation_and_analysis/` - dataset curation, pHash leakage audit, MD5
  manifest, statistics (Friedman/Nemenyi/Wilcoxon), Grad-CAM.
- `data/` - curated metadata (harmonized genus labels), LOSO fold definitions,
  per-image MD5 manifest, per-domain counts, domain-overlap matrix.
- `figures/` - the nine figures exactly as they appear in the manuscript.
- `results/aggregated/` - consolidated CSVs + `revision_numbers.json`
  (machine-readable headline numbers) + `posthoc/` (per-genus F1, confusions).
- `results/per_experiment_predictions.tar.gz` - every run's `test_predictions.csv`,
  `test_index.csv`, `confusion_matrix.csv`, `log.csv`, `final_metrics.csv`
  (multiseed x3, DG, ablation, sensitivity, efficiency, imbalance).
- `results/MANIFEST_all_runs.csv` - one row per run.
- `results/environment/` - `environment.json`, `pip_freeze.txt`.

## Reproduce the numbers (no GPU needed)
```bash
pip install -r requirements.txt
tar xzf results/per_experiment_predictions.tar.gz -C results/
python code/experiment_package/aggregate_results.py   # -> aggregated CSVs + revision_numbers.json
python code/regen_figs_bigfont.py                      # -> figures 2-5,7
```

## Retrain from scratch (GPU)
```bash
cd code/experiment_package
bash run_all.sh        # exp1-7 + aggregation + figures + export
```
Seeds {42, 202, 1337}; single NVIDIA RTX 6000 Ada (48 GB); torch 2.1.2/cu121.
"""
    open(os.path.join(OUT, "README.md"), "w").write(txt)

if __name__ == "__main__":
    main()
