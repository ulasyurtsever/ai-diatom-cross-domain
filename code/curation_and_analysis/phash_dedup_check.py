#!/usr/bin/env python3
"""
phash_dedup_check.py
--------------------
Perceptual-hash (pHash) duplicate / leakage audit of the Core Dataset.

For every image listed in ../data/diatom_metadata.csv it computes a 64-bit
pHash and reports:
  * exact-duplicate groups (Hamming distance 0), split into within-source,
    cross-source (= LOSO leakage) and cross-label groups;
  * cross-source near-duplicate pairs at Hamming distance <= 2 and <= 4.

This reproduces the leakage check reported in Section 3.3 of the manuscript:
no cross-source exact duplicates, a single cross-source pair within Hamming 4,
and a small residual of within-source exact duplicates (~0.6 % of images).

The image folder is not shipped with the repository; reconstruct it from the
six public sources first, then point this script at it with --data-root.

Requires: pillow, imagehash, numpy, pandas.
Usage:    python phash_dedup_check.py --data-root /path/to/Processed_Data_PNG
"""
import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import imagehash

HERE = Path(__file__).resolve().parent


def _find_repo_root(start):
    """Walk up from the script (and CWD) until data/diatom_metadata.csv is found,
    so the script works regardless of how deeply it is nested in the repository."""
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
    raise FileNotFoundError("Pass the image folder with --data-root.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--out", default=str(REPO / "data" / "phash_manifest.csv"))
    ap.add_argument("--near-thresholds", default="2,4")
    args = ap.parse_args()

    root = find_data_root(args.data_root)
    md = pd.read_csv(META)
    print(f"[i] hashing {len(md)} images from {root} ...")

    rows, bad = [], 0
    for _, r in md.iterrows():
        p = root / r["domain"] / r["label"] / Path(r["filepath"]).name
        try:
            h = imagehash.phash(Image.open(p).convert("L"))
            rows.append((r["domain"], r["label"], p.name, int(str(h), 16)))
        except Exception:
            bad += 1
    print(f"[i] hashed={len(rows)} unreadable={bad}")
    pd.DataFrame(rows, columns=["domain", "label", "name", "phash"]).to_csv(args.out, index=False)
    print(f"[i] manifest written to {args.out}")

    dom = np.array([r[0] for r in rows])
    lab = np.array([r[1] for r in rows])
    H = np.array([r[3] for r in rows], dtype=np.uint64)

    groups = collections.defaultdict(list)
    for i, r in enumerate(rows):
        groups[r[3]].append(i)
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    cross_dom = sum(1 for v in dups.values() if len({dom[i] for i in v}) > 1)
    cross_lab = sum(1 for v in dups.values() if len({lab[i] for i in v}) > 1)
    print("\n=== EXACT pHash duplicates ===")
    print(f"groups={len(dups)}  images={sum(len(v) for v in dups.values())}")
    print(f"  cross-source groups (LOSO leakage) : {cross_dom}")
    print(f"  cross-label groups                 : {cross_lab}")

    thrs = [int(t) for t in args.near_thresholds.split(",")]
    lut = np.array([bin(x).count("1") for x in range(256)], dtype=np.uint16)
    B = H.view(np.uint8).reshape(-1, 8)
    uniq = sorted(set(dom.tolist()))
    counts = {t: 0 for t in thrs}
    for a in range(len(uniq)):
        Ba = B[dom == uniq[a]]
        for b in range(a + 1, len(uniq)):
            Bb = B[dom == uniq[b]]
            for s in range(0, len(Ba), 300):
                d = lut[(Ba[s:s + 300][:, None, :] ^ Bb[None, :, :])].sum(axis=2)
                for t in thrs:
                    counts[t] += int((d <= t).sum())
    print("\n=== CROSS-SOURCE near-duplicate pairs ===")
    for t in thrs:
        print(f"  Hamming <= {t} : {counts[t]}")


if __name__ == "__main__":
    main()
