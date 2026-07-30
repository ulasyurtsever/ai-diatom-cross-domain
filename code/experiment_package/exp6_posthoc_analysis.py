"""EXP6 — post-hoc analyses from saved predictions (NO training).

Uses the seed-42 LOSO runs (prefers the freshly re-run ones under
results_revision/multiseed/seed_42, else the original results/per_experiment).
Produces, per model and aggregated over the six held-out domains:
  * per-genus precision / recall / F1               -> posthoc/per_genus_<model>.csv
  * most-confused genus pairs (global + LOIR)        -> posthoc/top_confusions*.csv
  * failure buckets w/ example image paths           -> posthoc/failure_buckets*.csv
  * richer summary metrics (balanced acc, micro-F1)  -> posthoc/summary_metrics.csv
Everything is image-traceable via test_index.csv, so figures can be made later
without retraining.
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import config_revision as C
from aggregate_results import REPO_ROOT   # reuse repo resolution (torch-free)
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_recall_fscore_support

RR = os.path.join(REPO_ROOT, C.RESULTS_SUBDIR)
OUT = os.path.join(RR, "aggregated", "posthoc"); os.makedirs(OUT, exist_ok=True)
MODELS = C.ALL_MODELS
DISPLAY = {"resnet50": "ResNet-50", "efficientnet_v2_s": "EfficientNetV2-S",
           "convnext_tiny": "ConvNeXt-Tiny", "swin_v2_t": "Swin V2-Tiny", "maxvit_t": "MaxViT-Tiny"}


def loso_dir(model, domain):
    for cand in [os.path.join(RR, "multiseed", "seed_42", f"S2_LOSO_{model}_{domain}"),
                 os.path.join(REPO_ROOT, "results", "per_experiment", f"S2_LOSO_{model}_{domain}")]:
        if os.path.exists(os.path.join(cand, "test_predictions.csv")):
            return cand
    return None


def domains():
    md = pd.read_csv(os.path.join(REPO_ROOT, "data", "diatom_metadata.csv"))
    return sorted(md[md["is_dg_class"]]["domain"].unique())


def class_names(pred_csv):
    cols = pd.read_csv(pred_csv, nrows=0).columns.tolist()
    return cols[2:]   # after True_Label, Predicted_Label


def per_genus_and_summary(doms):
    summ = []
    for m in MODELS:
        frames = []
        for d in doms:
            dd = loso_dir(m, d)
            if dd:
                frames.append(pd.read_csv(os.path.join(dd, "test_predictions.csv")))
        if not frames:
            continue
        cn = class_names(os.path.join(loso_dir(m, doms[0]), "test_predictions.csv"))
        P = pd.concat(frames, ignore_index=True)
        y, yp = P["True_Label"].values, P["Predicted_Label"].values
        pr, rc, f1, sup = precision_recall_fscore_support(
            y, yp, labels=range(len(cn)), zero_division=0)
        pd.DataFrame({"genus": cn, "precision": pr.round(4), "recall": rc.round(4),
                      "f1": f1.round(4), "support": sup}).to_csv(
            os.path.join(OUT, f"per_genus_{m}.csv"), index=False)
        summ.append({"Model": m, "MacroF1": round(f1_score(y, yp, average="macro"), 4),
                     "MicroF1": round(f1_score(y, yp, average="micro"), 4),
                     "BalancedAcc": round(balanced_accuracy_score(y, yp), 4),
                     "N_test": len(P)})
    if summ:
        pd.DataFrame(summ).to_csv(os.path.join(OUT, "summary_metrics.csv"), index=False)


def confusions(doms):
    # global + LOIR-specific top confusions from confusion_matrix.csv
    for scope, dsel in [("global", doms), ("LOIR", [d for d in doms if "LOIR" in d])]:
        agg = {}
        for m in MODELS:
            tot = None
            for d in dsel:
                dd = loso_dir(m, d)
                cmp = os.path.join(dd, "confusion_matrix.csv") if dd else None
                if cmp and os.path.exists(cmp):
                    cm = pd.read_csv(cmp, index_col=0)
                    tot = cm if tot is None else tot.add(cm, fill_value=0)
            if tot is None:
                continue
            pairs = []
            g = tot.index.tolist()
            for i in g:
                for j in tot.columns:
                    if i != j and tot.loc[i, j] > 0:
                        pairs.append((i, j, int(tot.loc[i, j])))
            pd.DataFrame(sorted(pairs, key=lambda x: -x[2])[:20],
                         columns=["true_genus", "predicted_as", "count"]).to_csv(
                os.path.join(OUT, f"top_confusions_{scope}_{m}.csv"), index=False)


def failure_buckets(doms):
    # align models by image (needs test_index.csv). Buckets per held-out domain.
    rows, examples = [], []
    for d in doms:
        idx = {}
        for m in MODELS:
            dd = loso_dir(m, d)
            ti = os.path.join(dd, "test_index.csv") if dd else None
            if ti and os.path.exists(ti):
                t = pd.read_csv(ti)[["filepath", "label", "Correct"]].rename(columns={"Correct": m})
                idx[m] = t.set_index("filepath")
        if len(idx) < len(MODELS):
            continue
        merged = None
        for m in MODELS:
            col = idx[m][[m]]
            merged = col if merged is None else merged.join(col, how="inner")
        lab = idx[MODELS[0]]["label"]
        merged = merged.join(lab, how="left")
        allc = merged[MODELS].all(axis=1)
        allf = ~merged[MODELS].any(axis=1)
        modern = merged["convnext_tiny"] & merged.get("maxvit_t", False) & ~merged["resnet50"]
        rows.append({"domain": d, "n": len(merged), "all_correct": int(allc.sum()),
                     "all_fail": int(allf.sum()), "modern_wins": int(modern.sum())})
        for name, mask in [("all_fail", allf), ("modern_wins", modern)]:
            for fp in merged[mask].index[:15]:
                examples.append({"domain": d, "bucket": name,
                                 "genus": merged.loc[fp, "label"], "filepath": fp})
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "failure_buckets_counts.csv"), index=False)
    if examples:
        pd.DataFrame(examples).to_csv(os.path.join(OUT, "failure_example_images.csv"), index=False)


def main():
    doms = domains()
    for fn in (per_genus_and_summary, confusions, failure_buckets):
        try:
            fn(doms)
        except Exception as e:
            print(f"  ({fn.__name__} skipped: {e})")
    print("post-hoc analyses ->", OUT)


if __name__ == "__main__":
    main()
