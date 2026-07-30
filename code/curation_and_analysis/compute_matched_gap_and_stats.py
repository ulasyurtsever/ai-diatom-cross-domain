#!/usr/bin/env python3
"""
compute_matched_gap_and_stats.py
--------------------------------
Two robustness analyses computed directly from the released per-experiment
prediction CSVs (no GPU, no retraining):

  (A) Label-matched in-distribution -> LOSO gap.
      The Standard-Mixed (Scenario B, 46-class) model is re-evaluated on the
      subset of its test set whose true label is one of the 27 LOSO core
      genera, giving an in-distribution accuracy / Macro-F1 on the *same*
      label space as Scenario C (LOSO). The reported gap is B(core-27) - LOSO,
      which isolates domain shift from the change in label space.

  (B) Omnibus architecture comparison on Macro-F1 (the primary metric):
      Friedman test, Kendall's W, Nemenyi critical difference, and pairwise
      Wilcoxon signed-rank with Holm step-down, on the 6 (held-out source)
      x 5 (architecture) Macro-F1 matrix.

These reproduce the values reported in Sections 3.3 / 4.1 of the manuscript.

Repository layout assumed (script lives in code/):
    ../data/diatom_metadata.csv
    ../results/per_experiment/<experiment>/{final_metrics.csv,test_predictions.csv}

Requires: numpy, pandas, scipy, scikit-learn.
Usage:    python compute_matched_gap_and_stats.py
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon
from sklearn.metrics import accuracy_score, f1_score

HERE = Path(__file__).resolve().parent


def _find_repo_root(start):
    """Locate the repo root (containing data/diatom_metadata.csv) by walking up
    from the script or the CWD, so the script works regardless of nesting."""
    for base in [start, *start.parents, Path.cwd(), *Path.cwd().parents]:
        if (base / "data" / "diatom_metadata.csv").exists():
            return base
    return start.parent


REPO = _find_repo_root(HERE)
META = REPO / "data" / "diatom_metadata.csv"
RES = REPO / "results" / "per_experiment"

MODELS = {                       # folder token -> display name
    "resnet50": "ResNet-50",
    "efficientnet_v2_s": "EfficientNetV2-S",
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_v2_t": "Swin V2-Tiny",
    "maxvit_t": "MaxViT-Tiny",
}
DOMAINS = ["ADIAC_Database", "AFD_Database", "DIA_Database",
           "DONA_Database", "FCE_LTER_Database", "LOIR_Database"]


def core27():
    core = sorted(pd.read_csv(META).query("is_dg_class").label.unique())
    assert len(core) == 27, f"expected 27 core genera, found {len(core)}"
    return core


def loso_matrices():
    """6x5 accuracy and Macro-F1 matrices from S2_LOSO_*/final_metrics.csv."""
    acc = pd.DataFrame(index=DOMAINS, columns=list(MODELS), dtype=float)
    f1 = pd.DataFrame(index=DOMAINS, columns=list(MODELS), dtype=float)
    for m in MODELS:
        for d in DOMAINS:
            row = pd.read_csv(RES / f"S2_LOSO_{m}_{d}" / "final_metrics.csv").iloc[0]
            acc.loc[d, m] = float(row["Acc"])
            f1.loc[d, m] = float(row["F1"])
    return acc, f1


def matched_gap(core):
    acc_loso, f1_loso = loso_matrices()
    print("=== (A) Label-matched in-distribution(27 core) -> LOSO gap ===")
    print(f"{'Model':16}{'B_acc(core27)':>14}{'LOSO_acc':>10}{'gap_acc':>9}"
          f"{'B_f1(core27)':>14}{'LOSO_f1':>9}{'gap_f1':>8}")
    for m in MODELS:
        d = pd.read_csv(RES / f"S1_Standard_{m}" / "test_predictions.csv")
        names = list(d.columns[2:])                       # class names in index order
        tn = d["True_Label"].map(lambda i: names[i])
        pn = d["Predicted_Label"].map(lambda i: names[i])
        mask = tn.isin(core)
        b_acc = accuracy_score(tn[mask], pn[mask])
        b_f1 = f1_score(tn[mask], pn[mask], labels=core, average="macro", zero_division=0)
        la, lf = acc_loso[m].mean(), f1_loso[m].mean()
        print(f"{MODELS[m]:16}{b_acc:14.3f}{la:10.3f}{b_acc-la:9.3f}"
              f"{b_f1:14.3f}{lf:9.3f}{b_f1-lf:8.3f}")


def omnibus_macro_f1():
    _, f1 = loso_matrices()
    order = list(MODELS)
    M = f1[order].values.astype(float)        # 6 folds x 5 models
    K, m = M.shape
    chi, p = friedmanchisquare(*[M[:, j] for j in range(m)])
    W = chi / (K * (m - 1))                    # Kendall's W
    CD = 2.728 * np.sqrt(m * (m + 1) / (6 * K))   # Nemenyi, q at alpha=0.05, k=5
    mean_rank = np.vstack([rankdata(-M[i]) for i in range(K)]).mean(0)

    print("\n=== (B) Macro-F1 omnibus (6 folds x 5 architectures) ===")
    print(f"Friedman chi2 = {chi:.3f}  p = {p:.4f}  Kendall W = {W:.3f}  Nemenyi CD = {CD:.3f}")
    for j in np.argsort(mean_rank):
        print(f"  {MODELS[order[j]]:16} mean rank = {mean_rank[j]:.3f}")
    print("Nemenyi-significant pairs (|dRank| > CD):")
    for i, j in itertools.combinations(range(m), 2):
        if abs(mean_rank[i] - mean_rank[j]) > CD:
            print(f"  {MODELS[order[i]]} vs {MODELS[order[j]]}: "
                  f"dRank = {abs(mean_rank[i]-mean_rank[j]):.2f}")

    pairs = []
    for i, j in itertools.combinations(range(m), 2):
        try:
            _, pw = wilcoxon(M[:, i], M[:, j])
        except ValueError:
            pw = 1.0
        pairs.append((order[i], order[j], pw))
    pairs.sort(key=lambda x: x[2])
    print("Pairwise Wilcoxon signed-rank + Holm step-down:")
    for k, (a, b, pw) in enumerate(pairs):
        print(f"  {MODELS[a]:16} vs {MODELS[b]:16}  "
              f"p_unc={pw:.3f}  p_holm={min(1.0, pw*(len(pairs)-k)):.3f}")


if __name__ == "__main__":
    c = core27()
    matched_gap(c)
    omnibus_macro_f1()
