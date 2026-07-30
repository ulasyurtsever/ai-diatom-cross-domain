
import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from train import build_model  # noqa: E402


def _ddir():
    """Return the repository data/ directory (with trailing slash), located by
    walking up from this script or the CWD; falls back to ../data/."""
    from pathlib import Path as _P
    for b in [_P(__file__).resolve().parent, *_P(__file__).resolve().parents, _P.cwd(), *_P.cwd().parents]:
        if (b / "data" / "diatom_metadata.csv").exists():
            return str(b / "data") + "/"
    return "../data/"



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODELS = ['convnext_tiny', 'maxvit_t', 'resnet50']
DISPLAY = {'convnext_tiny': 'ConvNeXt-T',
           'maxvit_t': 'MaxViT-T',
           'resnet50': 'ResNet-50'}


def draw_checkmark(img, cx, cy, size=10, color=(0, 200, 0), thickness=2):
    cv2.line(img, (cx - size, cy), (cx - 3, cy + size), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx - 3, cy + size), (cx + size, cy - size - 5), color, thickness, cv2.LINE_AA)


def draw_cross(img, cx, cy, size=8, color=(0, 0, 200), thickness=2):
    cv2.line(img, (cx - size, cy - size), (cx + size, cy + size), color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx + size, cy - size), (cx - size, cy + size), color, thickness, cv2.LINE_AA)


def add_header(image_rgb, title, status=None):
    """Adds a white strip above the image with the title; if status is a
    bool, also draws a tick (True) or cross (False) next to the title."""
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w, _ = bgr.shape
    padded = cv2.copyMakeBorder(bgr, 40, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 0.6, 1
    tw, th = cv2.getTextSize(title, font, scale, thickness)[0]
    icon_space = 30 if status is not None else 0
    start_x = (w - tw - icon_space) // 2
    text_y = 25

    cv2.putText(padded, title, (start_x, text_y), font, scale, (0, 0, 0),
                thickness, cv2.LINE_AA)
    if status is True:
        draw_checkmark(padded, start_x + tw + 15, text_y - 7)
    elif status is False:
        draw_cross(padded, start_x + tw + 15, text_y - 7)

    return cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)


def target_layers_for(arch, model):
    if "resnet" in arch:
        return [model.layer4[-1]]
    if "convnext" in arch:
        return [model.features[-1]]
    if "maxvit" in arch:
        return [model.blocks[-1]]
    raise ValueError(f"No target layer rule for: {arch}")


def label_map_dg(df):
    sub = df[df['is_dg_class']] if 'is_dg_class' in df.columns else df
    unique = sorted(sub['label'].unique())
    return {n: i for i, n in enumerate(unique)}


def classify_test_set(df_meta, results_dir, target_domain, scenario_prefix, n_per_bucket):
    """Return (selected, test_meta) where selected[bucket] is a list of
    {idx, status} dicts and bucket is one of:
        all_correct, modern_wins, all_fail
    """
    if 'is_dg_class' in df_meta.columns:
        test_meta = df_meta[(df_meta['domain'] == target_domain) &
                            df_meta['is_dg_class']].reset_index(drop=True)
    else:
        test_meta = df_meta[df_meta['domain'] == target_domain].reset_index(drop=True)
    print(f"  test set size: {len(test_meta)}")

    label_idx = label_map_dg(df_meta)

    preds = {}
    for m in MODELS:
        pattern = f"{results_dir}/*{scenario_prefix}_{m}_{target_domain}*/test_predictions.csv"
        files = glob.glob(pattern)
        if not files:
            print(f"  prediction file missing for {m}")
            continue
        df_pred = pd.read_csv(files[0])
        if len(df_pred) != len(test_meta):
            # Tolerate length mismatch by truncating to the shorter
            n = min(len(df_pred), len(test_meta))
            df_pred = df_pred.iloc[:n]
        preds[m] = df_pred

    if len(preds) < 3:
        print("  not all three architectures present; aborting")
        return None, None

    limit = min(len(test_meta), *(len(preds[m]) for m in MODELS))
    buckets = {'all_correct': [], 'modern_wins': [], 'all_fail': []}

    for i in range(limit):
        true_idx_meta = label_idx.get(test_meta.iloc[i]['label'], -1)
        status = {}
        for m in MODELS:
            row = preds[m].iloc[i]
            if row['True_Label'] != true_idx_meta:
                status[m] = False
            else:
                status[m] = (row['Predicted_Label'] == true_idx_meta)

        c_ok, m_ok, r_ok = (status.get('convnext_tiny', False),
                            status.get('maxvit_t', False),
                            status.get('resnet50', False))

        if c_ok and m_ok and r_ok:
            buckets['all_correct'].append({'idx': i, 'status': status})
        elif c_ok and m_ok and not r_ok:
            buckets['modern_wins'].append({'idx': i, 'status': status})
        elif not (c_ok or m_ok or r_ok):
            buckets['all_fail'].append({'idx': i, 'status': status})

    for k, v in buckets.items():
        print(f"  bucket {k}: {len(v)} candidates")

    rnd = random.Random(42)
    selection = {k: (rnd.sample(v, n_per_bucket) if len(v) > n_per_bucket else v)
                 for k, v in buckets.items()}
    return selection, test_meta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='../results/per_experiment')
    parser.add_argument('--metadata', default=_ddir()+'diatom_metadata.csv')
    parser.add_argument('--output-dir', default='Final_GradCAM_Images')
    parser.add_argument('--target-domain', default='ADIAC_Database')
    parser.add_argument('--scenario-prefix', default='S2_LOSO')
    parser.add_argument('--samples-per-bucket', type=int, default=30)
    args = parser.parse_args()

    if not os.path.exists(args.metadata):
        print(f"Missing metadata: {args.metadata}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.metadata)
    n_cls = len(label_map_dg(df))

    selection, test_meta = classify_test_set(
        df, args.results_dir, args.target_domain,
        args.scenario_prefix, args.samples_per_bucket)
    if selection is None:
        return

    for bucket_name, items in selection.items():
        print(f"\nProcessing bucket: {bucket_name}")
        save_dir = os.path.join(args.output_dir, bucket_name)
        os.makedirs(save_dir, exist_ok=True)

        for item in items:
            idx = item['idx']
            row = test_meta.iloc[idx]
            img_path = row['filepath']
            label = row['label']
            if not os.path.exists(img_path):
                continue

            orig = Image.open(img_path).convert('RGB').resize((224, 224))
            orig_np = np.array(orig)
            img_f = np.float32(orig) / 255.0
            input_t = preprocess_image(orig,
                                       mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225]).to(DEVICE)

            row_images = [add_header(orig_np, "Original")]

            for arch in MODELS:
                folder = glob.glob(
                    f"{args.results_dir}/*{args.scenario_prefix}_{arch}_{args.target_domain}*")
                if not folder:
                    row_images.append(np.zeros((266, 224, 3), dtype=np.uint8))
                    continue
                ckpt = glob.glob(f"{folder[0]}/*.pth")
                if not ckpt:
                    row_images.append(np.zeros((266, 224, 3), dtype=np.uint8))
                    continue

                try:
                    model, _ = build_model(arch, n_cls)
                    state = torch.load(ckpt[0], map_location=DEVICE)
                    if 'model_state_dict' in state:
                        state = state['model_state_dict']
                    state = {k.replace('module.', '', 1): v for k, v in state.items()}
                    model.load_state_dict(state, strict=False)
                    model.to(DEVICE).eval()

                    cam = GradCAM(model, target_layers_for(arch, model))
                    gray = cam(input_t, targets=None)[0, :]
                    heat = show_cam_on_image(img_f, gray, use_rgb=True)
                    is_ok = item['status'].get(arch, False)
                    row_images.append(add_header(heat, DISPLAY[arch], is_ok))
                except (RuntimeError, ValueError, KeyError) as e:
                    print(f"  {arch} failed on idx={idx}: {type(e).__name__}: {e}")
                    row_images.append(np.zeros((266, 224, 3), dtype=np.uint8))

            combined = np.hstack(row_images)
            Image.fromarray(combined).save(
                os.path.join(save_dir, f"{label}_{idx}.png"))

    print(f"\nDone. See {args.output_dir}/")


if __name__ == "__main__":
    main()
