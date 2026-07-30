"""
Shared training/eval engine for the R1 revision experiments.

It reuses the EXACT model definitions, seeding and early-stopping from the
released train.py (imported as `base`) so results stay comparable, and adds:
  * per-run seed / batch / lr / crop-resolution overrides
  * configurable augmentation flags & intensity  (for the ablation)
  * two dedicated DG methods: Deep CORAL and MixStyle
  * identical per-experiment output files + extra bookkeeping columns

Output per run  (<out_dir>/):
  final_metrics.csv, test_predictions.csv, confusion_matrix.csv, log.csv
"""
import os, sys, time, gc, json, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm


# --------------------------------------------------------------------------
# Repo discovery + import of the original train.py
# --------------------------------------------------------------------------
def resolve_repo_root():
    """Locate the released repo root (contains train.py + data/diatom_metadata.csv)."""
    cands = []
    if os.environ.get("REPO_ROOT"):
        cands.append(os.environ["REPO_ROOT"])
    cands.append(os.getcwd())
    here = os.path.dirname(os.path.abspath(__file__))
    cands += [here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))]
    for c in cands:
        if (os.path.exists(os.path.join(c, "train.py")) and
                os.path.exists(os.path.join(c, "data", "diatom_metadata.csv"))):
            return os.path.abspath(c)
    raise SystemExit(
        "Could not find the repo root (train.py + data/diatom_metadata.csv).\n"
        "Set it explicitly:  REPO_ROOT=/path/to/GitHub_Repo_Ready python <script>.py")


REPO_ROOT = resolve_repo_root()
sys.path.insert(0, REPO_ROOT)
import train as base   # noqa: E402  (uses base.build_model, base.seed_everything, ...)

CSV_PATH = os.path.join(REPO_ROOT, "data", "diatom_metadata.csv")
HW = base.detect_hardware()


def device_and_batch(batch_override=None):
    dev = torch.device('cuda' if torch.cuda.is_available()
                       else 'mps' if torch.backends.mps.is_available() else 'cpu')
    bs = batch_override or int(os.environ.get('DIATOM_BATCH', 0)) or HW['BATCH_SIZE']
    return dev, bs


# --------------------------------------------------------------------------
# MixStyle  (Zhou et al., ICLR 2021) — domain-agnostic feature-statistic mixing
# --------------------------------------------------------------------------
class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        self.p, self.beta, self.eps = p, torch.distributions.Beta(alpha, alpha), eps

    def forward(self, x):
        if (not self.training) or (random.random() > self.p) or x.size(0) < 2:
            return x
        B = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        x_norm = (x - mu) / sig
        lam = self.beta.sample((B, 1, 1, 1)).to(x.device)
        perm = torch.randperm(B, device=x.device)
        mu_mix = lam * mu + (1 - lam) * mu[perm]
        sig_mix = lam * sig + (1 - lam) * sig[perm]
        return x_norm * sig_mix + mu_mix


def insert_mixstyle(model, arch, p, alpha):
    """Insert MixStyle after early stages of the CNN backbones."""
    ms = MixStyle(p=p, alpha=alpha)
    ms2 = MixStyle(p=p, alpha=alpha)
    if arch == "resnet50":
        model.layer1 = nn.Sequential(model.layer1, ms)
        model.layer2 = nn.Sequential(model.layer2, ms2)
    elif arch == "convnext_tiny":
        # model.features: [stem, stage0, down, stage1, down, stage2, down, stage3]
        feats = list(model.features)
        feats.insert(2, ms)      # after first stage block
        feats.insert(5, ms2)     # after second stage block (index shifted by 1)
        model.features = nn.Sequential(*feats)
    else:
        raise ValueError(f"MixStyle insertion not defined for {arch}")
    return model


# --------------------------------------------------------------------------
# CORAL  (Sun & Saenko, ECCVW 2016) — align 2nd-order feature stats across
# the source domains present in each mini-batch.
# --------------------------------------------------------------------------
def _cov(f):
    n = f.size(0)
    fc = f - f.mean(0, keepdim=True)
    return (fc.t() @ fc) / max(n - 1, 1)

def coral_loss(feat, domain_idx):
    doms = torch.unique(domain_idx)
    covs = []
    d = feat.size(1)
    for dm in doms:
        fi = feat[domain_idx == dm]
        if fi.size(0) >= 2:
            covs.append(_cov(fi))
    if len(covs) < 2:
        return feat.new_zeros(())
    loss, k = feat.new_zeros(()), 0
    for i in range(len(covs)):
        for j in range(i + 1, len(covs)):
            loss = loss + ((covs[i] - covs[j]) ** 2).sum() / (4 * d * d)
            k += 1
    return loss / max(k, 1)


def final_linear(model, arch):
    m = model.module if isinstance(model, nn.DataParallel) else model
    return {"resnet50": lambda: m.fc[1],
            "efficientnet_v2_s": lambda: m.classifier[1],
            "convnext_tiny": lambda: m.classifier[2],
            "swin_v2_t": lambda: m.head[1],
            "maxvit_t": lambda: m.classifier[5]}[arch]()


# --------------------------------------------------------------------------
# Dataset (optionally returns a domain index, for CORAL)
# --------------------------------------------------------------------------
class DiatomDS(Dataset):
    def __init__(self, df, transform=None, with_domain=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.with_domain = with_domain

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r['filepath']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(r['label_idx'], dtype=torch.long)
        if self.with_domain:
            return img, y, torch.tensor(int(r['domain_idx']), dtype=torch.long)
        return img, y


# --------------------------------------------------------------------------
# Transforms with configurable augmentation
# --------------------------------------------------------------------------
def native_size(base_tf):
    for attr in ("crop_size", "resize_size"):
        v = getattr(base_tf, attr, None)
        if v:
            return v[0] if isinstance(v, (list, tuple)) else int(v)
    return 224

def build_train_tf(base_tf, crop_size, aug):
    ops = [transforms.RandomResizedCrop(crop_size, scale=(0.6, 1.0))]
    if aug.get("flip", True):
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    if aug.get("rotation", True):
        ops += [transforms.RandomRotation(aug.get("rotation_deg", 15))]
    if aug.get("jitter", True):
        j = aug.get("jitter_mag", 0.2)
        ops += [transforms.ColorJitter(j, j, j)]
    ops += [base_tf]
    if aug.get("erase", True):
        ops += [transforms.RandomErasing(p=aug.get("erase_p", 0.5), scale=(0.02, 0.15))]
    return transforms.Compose(ops)


# --------------------------------------------------------------------------
# Split logic — mirrors base.make_dataloaders exactly (STANDARD/SINGLE/LOSO)
# --------------------------------------------------------------------------
def prepare_dfs(df, mode, target, seed):
    if mode == "STANDARD":
        classes = (df[['label_idx', 'label']].drop_duplicates()
                   .sort_values('label_idx')['label'].tolist())
        tr = df[df['split'] == 'train'].copy()
        va = df[df['split'] == 'val'].copy()
        te = df[df['split'] == 'test'].copy()
    elif mode == "SINGLE":
        sub = df[df['domain'] == target].copy()
        classes = sorted(sub['label'].unique())
        lmap = {n: i for i, n in enumerate(classes)}
        for d in (sub,):
            pass
        tr = sub[sub['split'] == 'train'].copy()
        va = sub[sub['split'] == 'val'].copy()
        te = sub[sub['split'] == 'test'].copy()
        for d in (tr, va, te):
            d['label_idx'] = d['label'].map(lmap)
    elif mode == "LOSO":
        sub = df[df['is_dg_class']].copy()
        classes = sorted(sub['label'].unique())
        lmap = {n: i for i, n in enumerate(classes)}
        sub['label_idx'] = sub['label'].map(lmap)
        te = sub[sub['domain'] == target].copy()
        seen = sub[sub['domain'] != target].copy()
        tr = seen[seen['split'] == 'train'].copy()
        va = seen[seen['split'] == 'val'].copy()
        if len(va) < 2:
            from sklearn.model_selection import train_test_split
            tr, va = train_test_split(tr, test_size=0.1, random_state=seed)
    else:
        raise ValueError(mode)
    # domain index for CORAL (based on training domains)
    doms = sorted(tr['domain'].unique())
    dmap = {d: i for i, d in enumerate(doms)}
    for d in (tr, va, te):
        d['domain_idx'] = d['domain'].map(lambda x: dmap.get(x, -1))
    return tr, va, te, classes


def class_weights_for(tr, n_classes):
    y = tr['label_idx'].values
    present = np.unique(y)
    wp = compute_class_weight('balanced', classes=present, y=y)
    w = torch.ones(n_classes, dtype=torch.float)
    for c, wv in zip(present, wp):
        w[int(c)] = wv
    return w


class FocalLoss(nn.Module):
    """Multiclass alpha-balanced focal loss (Lin et al., 2017).
    weight = per-class alpha (same 'balanced' weights as WCE); gamma focuses on
    hard examples. Reduces to weighted CE when gamma=0."""
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(
            logits, target, weight=self.weight,
            label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce).clamp(min=1e-6, max=1.0)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# --------------------------------------------------------------------------
# One run
# --------------------------------------------------------------------------
def train_one(*, out_dir, arch, mode, target, seed=42, method="erm",
              batch=None, lr_base=1e-4, crop=None, aug=None,
              epochs=None, patience=None, coral_lambda=0.5,
              mixstyle_p=0.5, mixstyle_alpha=0.1, extra_cols=None,
              loss="wce"):
    if os.path.exists(os.path.join(out_dir, "final_metrics.csv")):
        print(f"  skip (done): {out_dir}")
        return
    os.makedirs(out_dir, exist_ok=True)
    base.seed_everything(seed)
    aug = dict(aug or {})
    epochs = epochs or base.CONFIG['EPOCHS']
    patience = patience or base.CONFIG['PATIENCE']

    df = pd.read_csv(CSV_PATH)
    tr, va, te, classes = prepare_dfs(df, mode, target, seed)
    n_cls = len(classes)

    model, base_tf = base.build_model(arch, n_cls)
    if method == "mixstyle":
        model = insert_mixstyle(model, arch, mixstyle_p, mixstyle_alpha)

    dev, bs = device_and_batch(batch)
    crop_size = crop or native_size(base_tf)
    train_tf = build_train_tf(base_tf, crop_size, aug)

    with_dom = (method == "coral")
    nw = int(os.environ.get('DIATOM_WORKERS', HW['NUM_WORKERS']))
    common = dict(batch_size=bs, num_workers=nw, pin_memory=True)
    tr_dl = DataLoader(DiatomDS(tr, train_tf, with_dom), shuffle=True, **common)
    va_dl = DataLoader(DiatomDS(va, base_tf), shuffle=False, **common)
    te_dl = DataLoader(DiatomDS(te, base_tf), shuffle=False, **common)

    model = model.to(dev)
    use_dp = (method != "coral") and torch.cuda.device_count() > 1
    if use_dp:
        model = nn.DataParallel(model)

    lr = lr_base * (bs / 32)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=base.CONFIG['WEIGHT_DECAY'])
    w = class_weights_for(tr, n_cls).to(dev)
    _ls = base.CONFIG['LABEL_SMOOTHING']
    if loss == "focal":
        crit = FocalLoss(gamma=2.0, weight=w, label_smoothing=_ls)
    elif loss == "ce":
        crit = nn.CrossEntropyLoss(label_smoothing=_ls)
    else:  # "wce" (default, unchanged baseline)
        crit = nn.CrossEntropyLoss(weight=w, label_smoothing=_ls)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=(dev.type == 'cuda'))
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=(dev.type == 'cuda'))

    ckpt = os.path.join(out_dir, f"best_model_{arch}.pth")
    stopper = base.EarlyStopping(patience=patience, path=ckpt)

    # CORAL feature hook
    feat_box = {}
    hook_handle = None
    if method == "coral":
        def pre_hook(mod, inp):
            feat_box['f'] = inp[0]
        hook_handle = final_linear(model, arch).register_forward_pre_hook(pre_hook)

    t0, history = time.time(), []
    for ep in range(epochs):
        model.train(); tl = 0.0
        for batch_data in tqdm(tr_dl, desc=f"{os.path.basename(out_dir)} ep{ep+1}", leave=False):
            if with_dom:
                img, lbl, dom = batch_data
                dom = dom.to(dev)
            else:
                img, lbl = batch_data
            img, lbl = img.to(dev), lbl.to(dev)
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=(dev.type == 'cuda')):
                out = model(img)
                loss = crit(out, lbl)
                if method == "coral" and 'f' in feat_box:
                    loss = loss + coral_lambda * coral_loss(feat_box['f'].float(), dom)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item()

        model.eval(); vl = 0.0; vp, vy = [], []
        with torch.no_grad():
            for img, lbl in va_dl:
                img, lbl = img.to(dev), lbl.to(dev)
                out = model(img)
                vl += crit(out, lbl).item()
                vp.extend(out.argmax(1).cpu().numpy()); vy.extend(lbl.cpu().numpy())
        tl /= len(tr_dl); vl /= len(va_dl)
        history.append({'epoch': ep + 1, 'train_loss': tl, 'val_loss': vl,
                        'val_acc': accuracy_score(vy, vp),
                        'val_f1': f1_score(vy, vp, average='macro'),
                        'lr': opt.param_groups[0]['lr']})
        pd.DataFrame(history).to_csv(os.path.join(out_dir, "log.csv"), index=False)
        sched.step(vl); stopper(vl, model)
        if stopper.early_stop:
            print("  early stop"); break

    dur = time.time() - t0
    if hook_handle:
        hook_handle.remove()

    # reload best
    sd = torch.load(ckpt, map_location=dev)
    has_mod = next(iter(sd)).startswith('module.')
    is_dp = isinstance(model, nn.DataParallel)
    if is_dp and not has_mod:
        sd = {'module.' + k: v for k, v in sd.items()}
    elif (not is_dp) and has_mod:
        sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
    model.load_state_dict(sd)

    model.eval(); preds, trues, probs = [], [], []
    with torch.no_grad():
        for img, lbl in tqdm(te_dl, desc="test", leave=False):
            img, lbl = img.to(dev), lbl.to(dev)
            out = model(img); p = torch.softmax(out, 1)
            preds.extend(out.argmax(1).cpu().numpy()); trues.extend(lbl.cpu().numpy())
            probs.extend(p.cpu().numpy())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='macro')
    try:
        yb = label_binarize(trues, classes=range(n_cls))
        auc = roc_auc_score(yb, probs, multi_class='ovr', average='macro')
    except ValueError:
        auc = float('nan')

    pd.concat([pd.DataFrame({'True_Label': trues, 'Predicted_Label': preds}),
               pd.DataFrame(probs, columns=classes)], axis=1).to_csv(
        os.path.join(out_dir, "test_predictions.csv"), index=False)
    # image-traceable index -> any downstream table/figure without retraining
    try:
        ti = te.reset_index(drop=True)[['filepath', 'domain', 'label']].copy()
        ti['True_Label'] = trues; ti['Predicted_Label'] = preds
        ti['MaxProb'] = [float(np.max(pr)) for pr in probs]
        ti['Correct'] = ti['True_Label'] == ti['Predicted_Label']
        ti.to_csv(os.path.join(out_dir, "test_index.csv"), index=False)
    except Exception as _e:
        print("  (test_index not written:", _e, ")")
    try:
        cm = confusion_matrix(trues, preds, labels=range(n_cls))
        pd.DataFrame(cm, index=classes, columns=classes).to_csv(
            os.path.join(out_dir, "confusion_matrix.csv"))
    except ValueError:
        pass

    row = {'Experiment': os.path.basename(out_dir), 'Model': arch, 'Mode': mode,
           'Domain': target or 'ALL', 'Method': method, 'Seed': seed,
           'Batch': bs, 'LR': round(lr, 6), 'CropRes': crop_size,
           'Device': HW['INFO'], 'Acc': round(acc, 4), 'F1': round(f1, 4),
           'AUC': round(auc, 4) if not np.isnan(auc) else '',
           'Time_Sec': round(dur, 2),
           'Energy_kWh': round(int(os.environ.get('DIATOM_TDP_WATT', base.CONFIG['DEVICE_TDP_WATT'])) * (dur / 3600) / 1000, 5)}
    if extra_cols:
        row.update(extra_cols)
    pd.DataFrame([row]).to_csv(os.path.join(out_dir, "final_metrics.csv"), index=False)
    print(f">> {os.path.basename(out_dir)}  Acc {acc:.4f}  F1 {f1:.4f}")

    del model, opt, sched, scaler
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def loso_domains():
    df = pd.read_csv(CSV_PATH)
    return sorted(df[df['is_dg_class']]['domain'].unique().tolist())
