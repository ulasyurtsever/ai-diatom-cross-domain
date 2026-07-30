"""Collect ALL lightweight artifacts into one archive so every table/figure can
be produced later WITHOUT retraining. Model checkpoints (*.pth) are intentionally
excluded (they stay on the server); everything else is included:
metrics, predictions, image-traceable indices, confusion matrices, per-epoch
logs, aggregated tables, post-hoc analyses, figures, environment capture.
Writes:  <repo>/results_revision_export.tar.gz  +  a MANIFEST.csv
"""
import os, sys, glob, tarfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, config_revision as C
from aggregate_results import REPO_ROOT

RR = os.path.join(REPO_ROOT, C.RESULTS_SUBDIR)


def manifest():
    rows = []
    for f in glob.glob(os.path.join(RR, "**", "final_metrics.csv"), recursive=True):
        try:
            r = pd.read_csv(f).iloc[0].to_dict(); r["path"] = os.path.relpath(f, REPO_ROOT)
            rows.append(r)
        except Exception:
            pass
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(RR, "MANIFEST_all_runs.csv"), index=False)
        print(f"  MANIFEST: {len(rows)} runs")


def make_archive():
    out = os.path.join(REPO_ROOT, "results_revision_export.tar.gz")
    n = 0
    with tarfile.open(out, "w:gz") as tar:
        for root, _, files in os.walk(RR):
            for fn in files:
                if fn.endswith(".pth"):     # skip heavy checkpoints
                    continue
                fp = os.path.join(root, fn)
                tar.add(fp, arcname=os.path.relpath(fp, REPO_ROOT))
                n += 1
    mb = os.path.getsize(out) / 1024**2
    print(f"  archive: {out}  ({n} files, {mb:.1f} MB)")
    print("  NOTE: model checkpoints (*.pth) were NOT archived; they remain under "
          "results_revision/ on the server (needed only if you regenerate Grad-CAM).")


def main():
    manifest()
    make_archive()


if __name__ == "__main__":
    main()
