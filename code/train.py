"""
Single-seed training pipeline for the cross-domain diatom classification benchmark.

For one random seed (default 42), this script trains every (backbone, scenario)
configuration: the five backbones under Single-Source, Standard-Mixed, and
Leave-One-Source-Out (LOSO, unfolded over the six held-out source databases),
for 40 runs in total. Each run writes its own subdirectory under results/ with
the trained checkpoint, per-epoch log, per-sample test predictions, confusion
matrix, and final metrics.

The full study reported in the manuscript averages three independent seeds
({42, 202, 1337}); the multi-seed campaign is orchestrated by
code/experiment_package/exp1_multiseed.py, which reuses the seed-42 outputs
produced here and trains the two additional seeds.

Hardware and optimisation follow the manuscript: AdamW with a base learning
rate of 1e-4 that is linearly scaled by the batch size (batch/32), giving an
effective 4e-4 at the batch size of 128 used on the reference GPU
(NVIDIA RTX 6000 Ada Generation, 48 GB). Batch size and worker counts are
detected automatically from the available hardware.

Expected input:  data/diatom_metadata.csv (run from the repository root).
Expected output: results/per_experiment/<experiment_name>/*.csv + best_model_*.pth
                 (the same directory that code/compute_statistics.py reads).
"""
import gc
import os
import platform
import random
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# -------------------------------------------------------------------------
# Hardware detection
# -------------------------------------------------------------------------
def detect_hardware():
    cfg = {'DEVICE': 'cpu', 'BATCH_SIZE': 16, 'NUM_WORKERS': 2,
           'DEVICE_TDP_WATT': 100, 'INFO': 'CPU'}

    if torch.backends.mps.is_available():
        cfg.update({'DEVICE': 'mps', 'BATCH_SIZE': 32, 'NUM_WORKERS': 4,
                    'DEVICE_TDP_WATT': 30, 'INFO': 'Apple Silicon (MPS)'})
    elif torch.cuda.is_available():
        cfg['DEVICE'] = 'cuda'
        n_gpu = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        if n_gpu > 1:
            cfg.update({'BATCH_SIZE': 128, 'NUM_WORKERS': 12,
                        'DEVICE_TDP_WATT': 700,
                        'INFO': f"Dual-GPU ({n_gpu}x {gpu_name})"})
        else:
            cfg.update({'BATCH_SIZE': 64, 'NUM_WORKERS': 8,
                        'DEVICE_TDP_WATT': 480,
                        'INFO': f"Single-GPU ({gpu_name})"})
    return cfg


HW = detect_hardware()

CONFIG = {
    'MODELS_TO_TEST': ['resnet50', 'efficientnet_v2_s', 'convnext_tiny',
                       'swin_v2_t', 'maxvit_t'],
    'CSV_PATH': 'data/diatom_metadata.csv',
    'RESULTS_DIR': 'results/per_experiment',
    'DEVICE_TDP_WATT': HW['DEVICE_TDP_WATT'],
    'BATCH_SIZE': HW['BATCH_SIZE'],
    'NUM_WORKERS': HW['NUM_WORKERS'],
    'DEVICE_INFO': HW['INFO'],
    'EPOCHS': 500,
    'PATIENCE': 20,
    'BASE_LR': 1e-4,
    'WEIGHT_DECAY': 0.05,
    'LABEL_SMOOTHING': 0.1,
    'SEED': 42,
    # Augmentation magnitudes
    'COLOR_JITTER': 0.2,
    'ROTATION_DEG': 15,
    'RANDOM_ERASE_P': 0.5,
    'CROP_SCALE_LO': 0.6,
    'CROP_SCALE_HI': 1.0,
}


# -------------------------------------------------------------------------
# Reproducibility
# -------------------------------------------------------------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


# -------------------------------------------------------------------------
# Early stopping (val-loss based)
# -------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=7, path='checkpoint.pth'):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self._save(model)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self._save(model)
            self.counter = 0

    def _save(self, model):
        # Strip DataParallel wrapper before saving
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, self.path)


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class DiatomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Let the exception propagate. If the image is unreadable we want to
        # know about it; silently returning a black image would corrupt the
        # training distribution.
        image = Image.open(row['filepath']).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(row['label_idx'], dtype=torch.long)


# -------------------------------------------------------------------------
# Backbones
# -------------------------------------------------------------------------
def build_model(arch, num_classes):
    """Returns (model, transforms). The classification head is replaced with
    a fresh layer sized to num_classes; dropout is added where the original
    head did not already include it (ResNet, Swin, EfficientNet).

    Pretrained weights use torchvision's ``DEFAULT`` enum, matching the
    configuration with which the released results were produced. Note that
    ``ResNet50_Weights.DEFAULT`` resolves to IMAGENET1K_V2 (acc@1 = 80.86 %),
    while the other four backbones have a single IMAGENET1K_V1 checkpoint to
    which DEFAULT also points. To reproduce the published numbers exactly,
    pin the torchvision version in requirements.txt rather than changing
    these enums."""
    if arch == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(0.5),
                                 nn.Linear(model.fc.in_features, num_classes))
    elif arch == 'efficientnet_v2_s':
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)
        in_f = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, num_classes))
    elif arch == 'convnext_tiny':
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        in_f = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_f, num_classes)
    elif arch == 'swin_v2_t':
        weights = models.Swin_V2_T_Weights.DEFAULT
        model = models.swin_v2_t(weights=weights)
        in_f = model.head.in_features
        model.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_f, num_classes))
    elif arch == 'maxvit_t':
        weights = models.MaxVit_T_Weights.DEFAULT
        model = models.maxvit_t(weights=weights)
        in_f = model.classifier[5].in_features
        model.classifier[5] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    return model, weights.transforms()


# -------------------------------------------------------------------------
# Data splitting per scenario
# -------------------------------------------------------------------------
def make_dataloaders(config, base_transform):
    df = pd.read_csv(config['CSV_PATH'])

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224,
                                     scale=(config['CROP_SCALE_LO'],
                                            config['CROP_SCALE_HI'])),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(config['ROTATION_DEG']),
        transforms.ColorJitter(config['COLOR_JITTER'],
                               config['COLOR_JITTER'],
                               config['COLOR_JITTER']),
        base_transform,
        transforms.RandomErasing(p=config['RANDOM_ERASE_P'],
                                 scale=(0.02, 0.15)),
    ])
    eval_tf = base_transform

    mode = config['EXPERIMENT_MODE']
    target = config.get('TEST_DOMAIN')

    if mode == 'STANDARD':
        train_df = df[df['split'] == 'train'].copy()
        val_df = df[df['split'] == 'val'].copy()
        test_df = df[df['split'] == 'test'].copy()
        classes = (df[['label_idx', 'label']].drop_duplicates()
                   .sort_values('label_idx')['label'].tolist())

    elif mode == 'SINGLE':
        sub = df[df['domain'] == target].copy()
        train_df = sub[sub['split'] == 'train'].copy()
        val_df = sub[sub['split'] == 'val'].copy()
        test_df = sub[sub['split'] == 'test'].copy()
        # Re-index labels to a dense 0..N-1 for this single domain
        classes = sorted(sub['label'].unique())
        lmap = {n: i for i, n in enumerate(classes)}
        for d in (train_df, val_df, test_df):
            d['label_idx'] = d['label'].map(lmap)

    elif mode == 'LOSO':
        sub = df[df['is_dg_class']].copy()
        classes = sorted(sub['label'].unique())
        lmap = {n: i for i, n in enumerate(classes)}
        sub['label_idx'] = sub['label'].map(lmap)

        test_df = sub[sub['domain'] == target].copy()
        seen = sub[sub['domain'] != target].copy()
        train_df = seen[seen['split'] == 'train'].copy()
        val_df = seen[seen['split'] == 'val'].copy()

        # If the validation pool from the seen domains is too thin, carve a
        # 10% slice out of the training set instead.
        if len(val_df) < 2:
            from sklearn.model_selection import train_test_split
            train_df, val_df = train_test_split(
                train_df, test_size=0.1, random_state=config['SEED'])
    else:
        raise ValueError(f"Unknown experiment mode: {mode}")

    # Class weights with defensive handling for classes that are absent from
    # the training fold (gives them weight 1.0 instead of crashing).
    n_classes = len(classes)
    train_labels = train_df['label_idx'].values
    present = np.unique(train_labels)
    w_present = compute_class_weight('balanced', classes=present, y=train_labels)

    weights = torch.ones(n_classes, dtype=torch.float)
    for c, w in zip(present, w_present):
        weights[int(c)] = w

    print(f"  Class weights ready ({len(present)}/{n_classes} classes present in train).")

    common = dict(batch_size=config['BATCH_SIZE'],
                  num_workers=config['NUM_WORKERS'],
                  pin_memory=True)
    train_loader = DataLoader(DiatomDataset(train_df, train_tf), shuffle=True, **common)
    val_loader = DataLoader(DiatomDataset(val_df, eval_tf), shuffle=False, **common)
    test_loader = DataLoader(DiatomDataset(test_df, eval_tf), shuffle=False, **common)

    return train_loader, val_loader, test_loader, classes, weights


# -------------------------------------------------------------------------
# Training one experiment
# -------------------------------------------------------------------------
def run_experiment(config):
    out_dir = os.path.join(config['RESULTS_DIR'], config['EXPERIMENT_NAME'])
    if os.path.exists(os.path.join(out_dir, "final_metrics.csv")):
        print(f"  skip (already done): {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    seed_everything(config['SEED'])

    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available()
        else 'cpu')

    # Determine the number of output classes for this scenario
    df_meta = pd.read_csv(config['CSV_PATH'])
    mode = config['EXPERIMENT_MODE']
    if mode == 'STANDARD':
        n_cls = int(df_meta['label_idx'].max()) + 1
    elif mode == 'LOSO':
        n_cls = df_meta[df_meta['is_dg_class']]['label'].nunique()
    elif mode == 'SINGLE':
        n_cls = df_meta[df_meta['domain'] == config['TEST_DOMAIN']]['label'].nunique()

    model, base_tf = build_model(config['MODEL_ARCH'], n_cls)
    train_dl, val_dl, test_dl, classes, class_weights = make_dataloaders(config, base_tf)

    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    lr = config['BASE_LR'] * (config['BATCH_SIZE'] / 32)
    optimizer = optim.AdamW(model.parameters(),
                            lr=lr,
                            weight_decay=config['WEIGHT_DECAY'])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device),
                                    label_smoothing=config['LABEL_SMOOTHING'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=0.1, patience=5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    ckpt_path = os.path.join(out_dir, f"best_model_{config['MODEL_ARCH']}.pth")
    early_stop = EarlyStopping(patience=config['PATIENCE'], path=ckpt_path)

    print(f"\n>> Starting {config['EXPERIMENT_NAME']}  (max epochs: {config['EPOCHS']})")
    t0 = time.time()
    history = []

    for epoch in range(config['EPOCHS']):
        ep_start = time.time()

        # Training
        model.train()
        tr_loss = 0.0
        tr_preds, tr_labels = [], []
        loop = tqdm(train_dl, desc=f"Ep {epoch+1}/{config['EPOCHS']}", leave=False)
        for img, lbl in loop:
            img, lbl = img.to(device), lbl.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(img)
                loss = criterion(out, lbl)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tr_loss += loss.item()
            tr_preds.extend(out.argmax(1).detach().cpu().numpy())
            tr_labels.extend(lbl.detach().cpu().numpy())

        # Validation
        model.eval()
        va_loss = 0.0
        va_preds, va_labels = [], []
        with torch.no_grad():
            for img, lbl in val_dl:
                img, lbl = img.to(device), lbl.to(device)
                out = model(img)
                va_loss += criterion(out, lbl).item()
                va_preds.extend(out.argmax(1).cpu().numpy())
                va_labels.extend(lbl.cpu().numpy())

        tr_loss /= len(train_dl)
        va_loss /= len(val_dl)
        tr_acc = accuracy_score(tr_labels, tr_preds)
        tr_f1 = f1_score(tr_labels, tr_preds, average='macro')
        va_acc = accuracy_score(va_labels, va_preds)
        va_f1 = f1_score(va_labels, va_preds, average='macro')
        cur_lr = optimizer.param_groups[0]['lr']
        ep_time = time.time() - ep_start

        history.append({
            'epoch': epoch + 1,
            'train_loss': tr_loss, 'train_acc': tr_acc, 'train_f1': tr_f1,
            'val_loss': va_loss, 'val_acc': va_acc, 'val_f1': va_f1,
            'lr': cur_lr, 'time': ep_time,
        })
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "log.csv"), index=False)

        print(f"Ep {epoch+1:3d} | T_loss {tr_loss:.4f} V_loss {va_loss:.4f} "
              f"| T_acc {tr_acc:.3f} V_acc {va_acc:.3f} "
              f"| T_F1 {tr_f1:.3f} V_F1 {va_f1:.3f} | lr {cur_lr:.1e}")

        scheduler.step(va_loss)
        early_stop(va_loss, model)
        if early_stop.early_stop:
            print("  -> early stopping triggered")
            break

    duration = time.time() - t0

    # Reload best checkpoint
    state_dict = torch.load(ckpt_path)
    has_module = next(iter(state_dict)).startswith('module.')
    is_dp = isinstance(model, nn.DataParallel)
    if is_dp and not has_module:
        state_dict = {'module.' + k: v for k, v in state_dict.items()}
    elif (not is_dp) and has_module:
        state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    # Test
    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for img, lbl in tqdm(test_dl, desc="Test"):
            img, lbl = img.to(device), lbl.to(device)
            out = model(img)
            prob = torch.softmax(out, 1)
            preds.extend(out.argmax(1).cpu().numpy())
            trues.extend(lbl.cpu().numpy())
            probs.extend(prob.cpu().numpy())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
    try:
        if n_cls > 2:
            y_bin = label_binarize(trues, classes=range(n_cls))
            auc = roc_auc_score(y_bin, probs, multi_class='ovr', average='macro')
        else:
            auc = roc_auc_score(trues, np.array(probs)[:, 1])
    except ValueError:
        # Test set may not contain every class — AUC undefined in that case.
        auc = float('nan')

    kwh = (config['DEVICE_TDP_WATT'] * (duration / 3600)) / 1000

    prob_df = pd.DataFrame(probs, columns=classes)
    pred_df = pd.DataFrame({'True_Label': trues, 'Predicted_Label': preds})
    pd.concat([pred_df, prob_df], axis=1).to_csv(
        os.path.join(out_dir, "test_predictions.csv"), index=False)

    metrics = {
        'Experiment': config['EXPERIMENT_NAME'],
        'Model': config['MODEL_ARCH'],
        'Mode': config['EXPERIMENT_MODE'],
        'Domain': config.get('TEST_DOMAIN') or 'ALL',
        'Device': config['DEVICE_INFO'],
        'Acc': round(acc, 4),
        'F1': round(f1, 4),
        'AUC': round(auc, 4) if not np.isnan(auc) else '',
        'Time_Sec': round(duration, 2),
        'Energy_kWh': round(kwh, 5),
    }
    pd.DataFrame([metrics]).to_csv(
        os.path.join(out_dir, "final_metrics.csv"), index=False)

    try:
        cm = confusion_matrix(trues, preds, labels=range(n_cls))
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            os.path.join(out_dir, "confusion_matrix.csv"))
    except ValueError as e:
        print(f"  (confusion matrix not written: {e})")

    print(f">> Done  Acc: {acc:.4f}  F1: {f1:.4f}\n")

    del model, optimizer, scheduler, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# -------------------------------------------------------------------------
# Entry point: runs the 5 backbones across the 3 scenarios
# -------------------------------------------------------------------------
def main():
    print(f"Diatom cross-domain benchmark — running on: {CONFIG['DEVICE_INFO']}")
    print(f"Platform: {platform.platform()}, PyTorch: {torch.__version__}")

    if not os.path.exists(CONFIG['CSV_PATH']):
        print(f"ERROR: metadata not found at {CONFIG['CSV_PATH']!r}")
        return

    df = pd.read_csv(CONFIG['CSV_PATH'])
    dg_domains = sorted(df[df['is_dg_class']]['domain'].unique().tolist())

    print("\n--- Scenario A: Single-Source (ADIAC) ---")
    for arch in CONFIG['MODELS_TO_TEST']:
        cfg = CONFIG.copy()
        cfg.update({'MODEL_ARCH': arch,
                    'EXPERIMENT_MODE': 'SINGLE',
                    'TEST_DOMAIN': 'ADIAC_Database',
                    'EXPERIMENT_NAME': f"S0_Single_{arch}_ADIAC_Database"})
        try:
            run_experiment(cfg)
        except Exception:
            traceback.print_exc()

    print("\n--- Scenario B: Standard Mixed (all 6 sources pooled) ---")
    for arch in CONFIG['MODELS_TO_TEST']:
        cfg = CONFIG.copy()
        cfg.update({'MODEL_ARCH': arch,
                    'EXPERIMENT_MODE': 'STANDARD',
                    'TEST_DOMAIN': None,
                    'EXPERIMENT_NAME': f"S1_Standard_{arch}"})
        try:
            run_experiment(cfg)
        except Exception:
            traceback.print_exc()

    print("\n--- Scenario C: Leave-One-Source-Out ---")
    for domain in dg_domains:
        print(f"\n[held-out target: {domain}]")
        for arch in CONFIG['MODELS_TO_TEST']:
            cfg = CONFIG.copy()
            cfg.update({'MODEL_ARCH': arch,
                        'EXPERIMENT_MODE': 'LOSO',
                        'TEST_DOMAIN': domain,
                        'EXPERIMENT_NAME': f"S2_LOSO_{arch}_{domain}"})
            try:
                run_experiment(cfg)
            except Exception:
                traceback.print_exc()

    print("\nAll experiments finished.")


if __name__ == "__main__":
    main()
