#!/usr/bin/env python3
"""
verify_dataset_integrity.py
---------------------------
Cross-checks the reconstructed image folder against the curated metadata
(../data/diatom_metadata.csv) used for all experiments.

Reports:
  * total images / domains / genera in the raw image folder,
  * the same for the curated metadata (the experimental Core Dataset),
  * which genera are dropped between the raw pool and the 46 curated classes,
  * whether the per-(domain, class) image counts of the 46 curated genera
    match the metadata exactly.

The image folder is not shipped with the repository; reconstruct it from the
six public sources first (see README / md5_manifest.csv), then point this
script at it with --data-root.

Usage:
    python verify_dataset_integrity.py --data-root /path/to/Processed_Data_PNG
"""
import argparse
from pathlib import Path

import pandas as pd

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


def find_data_root(explicit=None):
    if explicit:
        return Path(explicit)
    for name in ("Processed_Data_PNG", "+ Processed_Data_PNG"):
        for base in (REPO, REPO / "data", HERE):
            if (base / name).is_dir():
                return base / name
    raise FileNotFoundError(
        "Image folder not found. Reconstruct it from the public sources and "
        "pass it with --data-root.")


def scan_folder(root):
    rows = []
    for dom in sorted(p for p in root.iterdir() if p.is_dir()):
        for cls in sorted(p for p in dom.iterdir() if p.is_dir()):
            n = sum(1 for f in cls.iterdir() if f.suffix.lower() == ".png")
            rows.append({"domain": dom.name, "label": cls.name, "n": n})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    root = find_data_root(args.data_root)
    fdf = scan_folder(root)
    md = pd.read_csv(META)

    print(f"[i] image folder : {root}")
    print("\n=== RAW IMAGE FOLDER ===")
    print(f"images={int(fdf.n.sum())}  domains={fdf.domain.nunique()}  genera={fdf.label.nunique()}")
    print("=== CURATED METADATA (Core Dataset) ===")
    print(f"images={len(md)}  domains={md.domain.nunique()}  genera={md.label.nunique()}")

    kept = sorted(md.label.unique())
    dropped = sorted(set(fdf.label) - set(kept))
    print(f"\nGenera dropped from raw pool ({len(dropped)}): " + ", ".join(dropped))
    missing = sorted(set(kept) - set(fdf.label))
    print(f"Curated genera missing from folder: {missing if missing else 'none'}")

    fk = fdf[fdf.label.isin(kept)].set_index(["domain", "label"]).n.sort_index()
    mk = md[md.label.isin(kept)].groupby(["domain", "label"]).size().sort_index()
    j = pd.DataFrame({"folder": fk, "meta": mk}).fillna(0).astype(int)
    n_match = int((j.folder == j.meta).sum())
    print(f"\nPer-(domain,label) cells for the 46 curated genera: {len(j)}")
    print(f"  matching metadata exactly : {n_match}")
    print(f"  mismatching               : {len(j) - n_match}")
    if n_match != len(j):
        print(j[j.folder != j.meta])
    print(f"\nCurated-subset image total: folder={int(fk.sum())}  meta={len(md)}")
    print("OK" if (n_match == len(j) and int(fk.sum()) == len(md)) else "MISMATCH")


if __name__ == "__main__":
    main()
