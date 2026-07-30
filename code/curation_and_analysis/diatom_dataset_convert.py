
import argparse
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm


VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

# Directories we don't want to walk into
IGNORED_DIRS = {
    'Results', 'Results_Eski', 'Processed_Data_PNG', 'Processed_Data_New_PNG',
    '__pycache__', '.git', '.DS_Store', '.idea', '__MACOSX',
    'ImageDatabase.zip', 'Archive.zip',
}


def convert_one(args):
    src_path, dest_folder = args
    try:
        os.makedirs(dest_folder, exist_ok=True)
        safe_name = Path(src_path).stem.replace(" ", "_")
        dest_path = os.path.join(dest_folder, f"{safe_name}.png")
        if os.path.exists(dest_path):
            return True
        with Image.open(src_path) as img:
            # Normalise odd modes (CMYK from JPEGs, RGBA from PNGs, I;16 from TIFFs)
            img.convert("RGB").save(dest_path, "PNG", optimize=True)
        return True
    except (IOError, OSError, Image.DecompressionBombError) as e:
        print(f"  skipped (could not read): {src_path}  [{type(e).__name__}]")
        return False


def is_domain_dir(name):
    return not (name.startswith('.') or name in IGNORED_DIRS)


def scan_dataset(root_path):
    """Returns a list of dicts {domain, class, path} and a per-extension count."""
    out = []
    ext_counts = defaultdict(int)
    root = Path(root_path)

    domains = [d for d in root.iterdir() if d.is_dir() and is_domain_dir(d.name)]
    print(f"Domains to scan ({len(domains)}): {[d.name for d in domains]}")

    for domain in domains:
        print(f"  scanning {domain.name} ...")
        for current_root, dirs, files in os.walk(domain):
            if '__MACOSX' in current_root:
                continue
            valid = [f for f in files if Path(f).suffix.lower() in VALID_EXTENSIONS]
            if not valid:
                continue
            class_name = Path(current_root).name
            if class_name == domain.name:
                class_name = "Unlabeled_or_Root"
            for f in valid:
                ext_counts[Path(f).suffix.lower()] += 1
                out.append({'domain': domain.name,
                            'class': class_name,
                            'path': os.path.join(current_root, f)})

    print("\nExtension counts:")
    for ext, n in sorted(ext_counts.items()):
        print(f"  {ext}: {n}")

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', default='./Database/',
                        help='Root of the raw image database (default: %(default)s)')
    parser.add_argument('--dest', default='./Processed_Data_PNG',
                        help='Where to write the curated PNGs (default: %(default)s)')
    parser.add_argument('--min-count', type=int, default=30,
                        help='Minimum total images per class to keep (default: %(default)s)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Parallel workers (default: %(default)s)')
    args = parser.parse_args()

    dataset = scan_dataset(args.source)
    if not dataset:
        print("No images found. Check --source.")
        return

    global_counts = defaultdict(int)
    for item in dataset:
        global_counts[item['class']] += 1
    valid_classes = {k for k, v in global_counts.items() if v >= args.min_count}

    print()
    print(f"Total images           : {len(dataset)}")
    print(f"Total classes (raw)    : {len(global_counts)}")
    print(f"Classes kept (>= {args.min_count})    : {len(valid_classes)}")

    domain_stats = defaultdict(int)
    for item in dataset:
        domain_stats[item['domain']] += 1
    print("Per-domain image count:")
    for d, c in sorted(domain_stats.items()):
        print(f"  {d}: {c}")

    tasks = [(item['path'], os.path.join(args.dest, item['domain'], item['class']))
             for item in dataset if item['class'] in valid_classes]

    print(f"\nConverting {len(tasks)} images to {args.dest} ...")
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        results = list(tqdm(ex.map(convert_one, tasks), total=len(tasks)))

    print(f"Done. Success: {sum(results)} / {len(results)}")


if __name__ == "__main__":
    main()
