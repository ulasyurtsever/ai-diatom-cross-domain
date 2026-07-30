
import argparse
import glob
import json
import os

import pandas as pd


def _ddir():
    """Return the repository data/ directory (with trailing slash), located by
    walking up from this script or the CWD; falls back to ../data/."""
    from pathlib import Path as _P
    for b in [_P(__file__).resolve().parent, *_P(__file__).resolve().parents, _P.cwd(), *_P.cwd().parents]:
        if (b / "data" / "diatom_metadata.csv").exists():
            return str(b / "data") + "/"
    return "../data/"



TARGET_DOMAIN = "ADIAC_Database"
MAX_PER_SCENARIO = 3


def best_and_worst(master_df, scenario_mode):
    if scenario_mode == 'LOSO':
        sub = master_df[(master_df['Mode'] == 'LOSO') &
                        (master_df['Domain'] == TARGET_DOMAIN)]
    elif scenario_mode == 'SINGLE':
        sub = master_df[(master_df['Mode'] == 'SINGLE') &
                        (master_df['Experiment'].str.contains(TARGET_DOMAIN))]
    else:
        sub = master_df[master_df['Mode'] == 'STANDARD']

    if sub.empty:
        return None

    ordered = sub.sort_values('F1', ascending=False)
    best_row = ordered.iloc[0]
    worst_row = ordered.iloc[-1]
    return {
        'best_model': best_row['Model'],
        'worst_model': worst_row['Model'],
        'best_dir': glob.glob(f"../results/per_experiment/{best_row['Experiment']}*")[0],
        'worst_dir': glob.glob(f"../results/per_experiment/{worst_row['Experiment']}*")[0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='../results/per_experiment')
    parser.add_argument('--metadata', default=_ddir()+'diatom_metadata.csv')
    parser.add_argument('--output', default='evidence_list.json')
    args = parser.parse_args()

    metrics_files = glob.glob(os.path.join(args.results_dir, "**/final_metrics.csv"),
                              recursive=True)
    if not metrics_files:
        print(f"No results under {args.results_dir!r}")
        return

    frames = []
    for f in metrics_files:
        try:
            frames.append(pd.read_csv(f))
        except (IOError, pd.errors.ParserError):
            continue
    master = pd.concat(frames, ignore_index=True)
    meta = pd.read_csv(args.metadata)

    selected = {}
    for scenario in ('SINGLE', 'STANDARD', 'LOSO'):
        print(f"\n[{scenario}]")
        info = best_and_worst(master, scenario)
        if info is None:
            print("  no completed experiments")
            continue
        print(f"  best  : {info['best_model']}")
        print(f"  worst : {info['worst_model']}")

        try:
            best_pred = pd.read_csv(os.path.join(info['best_dir'], "test_predictions.csv"))
            worst_pred = pd.read_csv(os.path.join(info['worst_dir'], "test_predictions.csv"))
        except FileNotFoundError as e:
            print(f"  missing prediction file: {e}")
            continue

        if not (best_pred['True_Label'] == worst_pred['True_Label']).all():
            print("  best/worst prediction rows do not align — skipping")
            continue

        # Indices where best is right and worst is wrong (or fall back to common-correct)
        diff_idx = best_pred[
            (best_pred['Predicted_Label'] == best_pred['True_Label']) &
            (worst_pred['Predicted_Label'] != worst_pred['True_Label'])
        ].index.tolist()
        if not diff_idx:
            diff_idx = best_pred[
                best_pred['Predicted_Label'] == best_pred['True_Label']
            ].index.tolist()

        # Locate the corresponding test-split rows in metadata
        if scenario == 'STANDARD':
            test_meta = meta[meta['split'] == 'test']
        else:
            test_meta = meta[(meta['domain'] == TARGET_DOMAIN) &
                             (meta['split'] == 'test')]

        picked = []
        for i in diff_idx:
            if len(picked) >= MAX_PER_SCENARIO:
                break
            if i >= len(test_meta):
                continue
            row = test_meta.iloc[i]
            picked.append({
                'index': int(i),
                'filepath': row['filepath'],
                'label': row['label'],
                'best_model': info['best_model'],
                'worst_model': info['worst_model'],
            })
        selected[scenario] = picked
        print(f"  picked {len(picked)} samples")

    with open(args.output, 'w') as f:
        json.dump(selected, f, indent=4)
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
