"""Aggregate EXP7 (imbalance-loss) results into one comparison table.
WCE baseline is read from the existing exp1 seed_42 LOSO predictions; Focal and
CE from results_revision/imbalance/. Prints a ready-to-paste table and writes
results_revision/aggregated/imbalance_loss_LOSO.csv.
"""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import engine, config_revision as C

ROOT = engine.REPO_ROOT
RR = os.path.join(ROOT, C.RESULTS_SUBDIR)
ARCH = C.PRIMARY_MODEL
DOMAINS = engine.loso_domains()

SOURCES = {
    "WCE (baseline)": os.path.join(RR, "multiseed", "seed_42", "S2_LOSO_%s_%%s" % ARCH),
    "Focal (a-bal, g=2)": os.path.join(RR, "imbalance", "focal", "FOCAL_%s_%%s" % ARCH),
    "Plain CE": os.path.join(RR, "imbalance", "ce", "CE_%s_%%s" % ARCH),
}


def metrics(pattern):
    accs, f1s = [], []
    for d in DOMAINS:
        p = os.path.join(pattern % d, "test_predictions.csv")
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p)
        yt, yp = df["True_Label"].values, df["Predicted_Label"].values
        accs.append(accuracy_score(yt, yp))
        f1s.append(f1_score(yt, yp, average="macro", zero_division=0))
    return np.mean(accs), np.std(accs, ddof=1), np.mean(f1s), np.std(f1s, ddof=1)


def main():
    rows = []
    base_f1 = None
    for name, pat in SOURCES.items():
        m = metrics(pat)
        if m is None:
            print(f"  [missing] {name} -> run exp7 first"); continue
        acc, acc_sd, f1, f1_sd = m
        if "WCE" in name:
            base_f1 = f1
        rows.append({"Loss": name, "Acc": round(acc, 4), "Acc_SD": round(acc_sd, 4),
                     "MacroF1": round(f1, 4), "F1_SD": round(f1_sd, 4)})
    out = pd.DataFrame(rows)
    if base_f1 is not None:
        out["dF1_vs_WCE"] = (out["MacroF1"] - base_f1).round(4)
    os.makedirs(os.path.join(RR, "aggregated"), exist_ok=True)
    dst = os.path.join(RR, "aggregated", "imbalance_loss_LOSO.csv")
    out.to_csv(dst, index=False)
    print("\n=== ConvNeXt-Tiny LOSO, imbalance-loss comparison (seed 42) ===")
    print(out.to_string(index=False))
    print("\nwrote", dst)


if __name__ == "__main__":
    main()
