
import argparse
import glob
import os

import pandas as pd


METRIC_COLUMNS = ['Acc', 'F1', 'AUC', 'Time_Sec']
PRESENT_ORDER = ['Mode', 'Model', 'Domain'] + METRIC_COLUMNS


def load_master(results_dir):
    files = glob.glob(os.path.join(results_dir, "**/final_metrics.csv"),
                      recursive=True)
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except (IOError, pd.errors.ParserError) as e:
            print(f"  could not read {f}: {e}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='../results/per_experiment')
    parser.add_argument('--output-dir', default='Final_Reports')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_master(args.results_dir)
    if df.empty:
        print(f"No results found under {args.results_dir!r}.")
        return

    # Reorder columns for readability
    keep = [c for c in PRESENT_ORDER if c in df.columns]
    df = df[keep]

    # S0 — Single-Source
    s0 = df[df['Mode'] == 'SINGLE'].sort_values('F1', ascending=False)
    if not s0.empty:
        s0.to_csv(os.path.join(args.output_dir, "Report_S0_Single.csv"), index=False)
        print(f"  S0 written ({len(s0)} rows)")

    # S1 — Standard Mixed
    s1 = df[df['Mode'] == 'STANDARD'].sort_values('F1', ascending=False)
    if not s1.empty:
        s1.to_csv(os.path.join(args.output_dir, "Report_S1_Standard.csv"), index=False)
        print(f"  S1 written ({len(s1)} rows)")

    # S2 — LOSO detailed (one row per held-out domain)
    s2 = df[df['Mode'] == 'LOSO']
    if not s2.empty:
        s2_det = s2.sort_values(['Domain', 'F1'], ascending=[True, False])
        s2_det.to_csv(os.path.join(args.output_dir, "Report_S2_LOSO_Detailed.csv"),
                      index=False)
        print(f"  S2 detailed written ({len(s2_det)} rows)")

        s2_avg = (s2.groupby('Model')[METRIC_COLUMNS]
                  .mean()
                  .reset_index()
                  .sort_values('F1', ascending=False))
        s2_avg.to_csv(os.path.join(args.output_dir, "Report_S2_LOSO_Average_Ranking.csv"),
                      index=False)
        print(f"  S2 ranking written")
        print("\nLOSO ranking (mean over 6 held-out domains):")
        print(s2_avg[['Model', 'Acc', 'F1']].to_string(index=False))

    print(f"\nDone. See {args.output_dir}/")


if __name__ == "__main__":
    main()
