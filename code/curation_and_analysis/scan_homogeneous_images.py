"""
Scan the Core Dataset for images whose content is suspiciously homogeneous
(near-uniform colour, mostly blank, or fully black/white).  Reports any
image whose per-channel pixel standard deviation falls below a small
threshold.  These are candidates for being broken / empty rasters that
slipped through the curation pipeline.

Run locally on the workstation that holds the rebuilt image tree:

  python scan_homogeneous_images.py --data-root /path/to/Processed_Data_PNG

Outputs (next to md5_manifest.csv):

  data/homogeneity_scan.csv          one row per image, sorted ascending by std
  data/homogeneity_flagged.csv       subset with std < threshold (default 5.0)
  code/homogeneous_samples/          flagged files copied for visual review
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _ddir():
    """Return the repository data/ directory (with trailing slash), located by
    walking up from this script or the CWD; falls back to ../data/."""
    from pathlib import Path as _P
    for b in [_P(__file__).resolve().parent, *_P(__file__).resolve().parents, _P.cwd(), *_P.cwd().parents]:
        if (b / "data" / "diatom_metadata.csv").exists():
            return str(b / "data") + "/"
    return "../data/"



def load_paths(metadata_csv):
    df = pd.read_csv(metadata_csv)
    return df["filepath"].tolist()


def compute_std(full_path):
    """Return (std, n_unique, w, h, mode) for an image.  Faster than full
    entropy; std is robust to near-uniform content."""
    try:
        with Image.open(full_path) as img:
            mode = img.mode
            w, h = img.size
            arr = np.asarray(img, dtype=np.uint8)
    except Exception as e:
        return None, None, None, None, f"ERROR: {type(e).__name__}: {e}"
    std = float(arr.std())
    # n_unique on a flattened view; cap at 256 because uint8 has only 256 values per channel
    try:
        n_unique = int(np.unique(arr).size)
    except Exception:
        n_unique = -1
    return std, n_unique, w, h, mode


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default="./Processed_Data_PNG",
                   help="Root of the rebuilt image tree (default: %(default)s)")
    p.add_argument("--metadata", default=_ddir()+"diatom_metadata.csv",
                   help="Metadata CSV listing the curated 12,353 filepaths")
    p.add_argument("--out-dir", default=_ddir(),
                   help="Where to write scan CSVs")
    p.add_argument("--threshold", type=float, default=5.0,
                   help="Flag images with pixel std below this value (default: %(default)s)")
    p.add_argument("--copy-dir", default="./homogeneous_samples",
                   help="Where to copy flagged files for visual review")
    p.add_argument("--copy-cap", type=int, default=50,
                   help="Stop copying after this many flagged files (default: %(default)s)")
    args = p.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_dir = Path(args.copy_dir)
    copy_dir.mkdir(parents=True, exist_ok=True)

    paths = load_paths(args.metadata)
    print(f"Scanning {len(paths)} images for homogeneous content (threshold std < {args.threshold}) ...")

    rows = []
    n_err = 0
    for i, rel in enumerate(paths, 1):
        rel_strip = rel
        if rel_strip.startswith("Processed_Data_PNG/"):
            rel_strip = rel_strip[len("Processed_Data_PNG/"):]
        full = data_root / rel_strip
        if not full.exists():
            rows.append({"filepath": rel, "std": -1.0, "n_unique": -1,
                         "width": -1, "height": -1, "mode": "MISSING",
                         "file_size_bytes": 0})
            n_err += 1
            continue

        std, n_unique, w, h, mode = compute_std(full)
        size = full.stat().st_size
        if std is None:
            rows.append({"filepath": rel, "std": -1.0, "n_unique": -1,
                         "width": -1, "height": -1, "mode": mode,
                         "file_size_bytes": size})
            n_err += 1
        else:
            rows.append({"filepath": rel, "std": std, "n_unique": n_unique,
                         "width": w, "height": h, "mode": mode,
                         "file_size_bytes": size})
        if i % 1000 == 0:
            print(f"  {i}/{len(paths)} ...")

    scan_df = pd.DataFrame(rows).sort_values("std").reset_index(drop=True)
    scan_path = out_dir / "homogeneity_scan.csv"
    scan_df.to_csv(scan_path, index=False)
    print(f"\nWrote {scan_path} ({len(scan_df)} rows; {n_err} read errors)")

    flagged = scan_df[scan_df["std"] < args.threshold].copy()
    flagged_path = out_dir / "homogeneity_flagged.csv"
    flagged.to_csv(flagged_path, index=False)
    print(f"Wrote {flagged_path}: {len(flagged)} flagged rows (std < {args.threshold})")

    # Distribution stats
    valid = scan_df[scan_df["std"] >= 0]
    print(f"\nStd distribution over {len(valid)} valid images:")
    for pct in [0.1, 1, 5, 25, 50, 75, 95, 99]:
        v = np.percentile(valid["std"], pct)
        print(f"  P{pct:>5}: {v:8.2f}")

    # Copy a sample of flagged files for visual review
    print(f"\nCopying up to {args.copy_cap} flagged files to {copy_dir} ...")
    copied = 0
    for _, row in flagged.iterrows():
        if copied >= args.copy_cap:
            break
        rel = row["filepath"]
        rel_strip = rel[len("Processed_Data_PNG/"):] if rel.startswith("Processed_Data_PNG/") else rel
        src = data_root / rel_strip
        if not src.exists():
            continue
        flat = rel.replace("/", "__")
        try:
            shutil.copy2(src, copy_dir / flat)
            copied += 1
        except Exception as e:
            print(f"  copy failed for {flat}: {e}")
    print(f"  copied {copied} files")

    # Per-domain breakdown of flagged
    if len(flagged) > 0:
        flagged["domain"] = flagged["filepath"].str.split("/").str[1]
        print("\nFlagged by domain:")
        for dom, n in flagged["domain"].value_counts().items():
            print(f"  {dom:25s}  {n:5d}")

    print("\nDone.  Send `data/homogeneity_flagged.csv` and the contents of "
          f"`{copy_dir}` to inspect them.")


if __name__ == "__main__":
    main()
