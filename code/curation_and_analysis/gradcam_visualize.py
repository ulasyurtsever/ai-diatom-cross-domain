
import argparse
import glob
import os
import sys
from collections import OrderedDict

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import preprocess_image, show_cam_on_image


def _ddir():
    """Return the repository data/ directory (with trailing slash), located by
    walking up from this script or the CWD; falls back to ../data/."""
    from pathlib import Path as _P
    for b in [_P(__file__).resolve().parent, *_P(__file__).resolve().parents, _P.cwd(), *_P.cwd().parents]:
        if (b / "data" / "diatom_metadata.csv").exists():
            return str(b / "data") + "/"
    return "../data/"


# train.py is one level up when this script is run from the repo root
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
try:
    from train import build_model
except ImportError:
    print("Could not import train.build_model. Run this script from the repo root.")
    raise


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 10


def swin_reshape(tensor):
    """Swin transformer feature maps come back as (B, L, C). Grad-CAM
    needs (B, C, H, W), so we infer H = W = sqrt(L) and permute."""
    if tensor is None or tensor.ndim == 4:
        return tensor
    B, L, C = tensor.shape
    H = int(np.sqrt(L))
    return tensor.reshape(B, H, H, C).permute(0, 3, 1, 2)


def target_layers(arch, model):
    if "resnet" in arch:
        return [model.layer4[-1]]
    if "efficientnet" in arch:
        return [model.features[-1]]
    if "convnext" in arch:
        return [model.features[-1]]
    if "swin" in arch:
        return [model.features[-1][-1].norm1]
    if "maxvit" in arch:
        return [model.blocks[-1]]
    return [list(model.children())[-1]]


def num_classes_for_experiment(exp_name, meta):
    """Replicate the n_classes selection used in train.py."""
    if "S2_LOSO" in exp_name:
        return meta[meta['is_dg_class']]['label'].nunique()
    for d in meta['domain'].unique():
        if d in exp_name:
            return meta[meta['domain'] == d]['label'].nunique()
    return int(meta['label_idx'].max()) + 1


def load_state(model, ckpt_path):
    state = torch.load(ckpt_path, map_location=DEVICE)
    if 'model_state_dict' in state:
        state = state['model_state_dict']
    if next(iter(state)).startswith('module.'):
        state = OrderedDict((k.replace('module.', '', 1), v) for k, v in state.items())
    model.load_state_dict(state, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='../results/per_experiment')
    parser.add_argument('--metadata', default=_ddir()+'diatom_metadata.csv')
    parser.add_argument('--output-dir', default='GradCAM_Output')
    parser.add_argument('--n-samples', type=int, default=NUM_SAMPLES)
    args = parser.parse_args()

    if not os.path.exists(args.metadata):
        print(f"Missing metadata: {args.metadata}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    meta = pd.read_csv(args.metadata)

    checkpoints = glob.glob(os.path.join(args.results_dir, "**/best_model_*.pth"),
                            recursive=True)
    if not checkpoints:
        print(f"No checkpoints under {args.results_dir}/")
        return

    print(f"Running Grad-CAM on {len(checkpoints)} checkpoints (device: {DEVICE})")

    for ckpt in checkpoints:
        folder = os.path.basename(os.path.dirname(ckpt))
        arch = os.path.basename(ckpt).replace("best_model_", "").replace(".pth", "")
        print(f"\n  {arch}  [{folder}]")

        try:
            n_cls = num_classes_for_experiment(folder, meta)
            model, _ = build_model(arch, n_cls)
            model = load_state(model, ckpt).to(DEVICE).eval()

            layers = target_layers(arch, model)
            kwargs = {'reshape_transform': swin_reshape} if "swin" in arch else {}
            cam = GradCAM(model=model, target_layers=layers, **kwargs)

            # Choose test images from the matching domain when possible
            test_domain = next((d for d in meta['domain'].unique() if d in folder), None)
            if test_domain:
                subset = meta[(meta['domain'] == test_domain) &
                              (meta['split'] == 'test')]
            else:
                subset = meta[meta['split'] == 'test']
            if subset.empty:
                subset = meta[meta['split'] == 'test']

            n = min(args.n_samples, len(subset))
            samples = subset.sample(n, random_state=42)

            for idx, row in samples.iterrows():
                if not os.path.exists(row['filepath']):
                    continue
                img = Image.open(row['filepath']).convert('RGB').resize((224, 224))
                img_f = np.float32(img) / 255.0
                input_t = preprocess_image(img,
                                           mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225]).to(DEVICE)
                gray = cam(input_tensor=input_t, targets=None)[0, :]
                heat = show_cam_on_image(img_f, gray, use_rgb=True)
                combined = np.hstack((np.uint8(img_f * 255), heat))
                out = os.path.join(args.output_dir,
                                   f"{folder}_{row['label']}_{idx}.jpg")
                Image.fromarray(combined).save(out)

        except (RuntimeError, ValueError, KeyError) as e:
            print(f"  failed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
