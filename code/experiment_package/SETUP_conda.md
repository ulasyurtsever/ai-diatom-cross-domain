# Server setup (conda) — step by step

Exact pinned versions from the original run (Python 3.11, torch 2.1.2,
torchvision 0.16.2). Match them so the reference RTX 6000 Ada Generation run stays consistent with the
released torchvision `Weights.DEFAULT` behaviour.

## 0) Miniconda (skip if conda already installed)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate
conda init bash        # then open a new shell
```

## 1) Create + activate env
```bash
conda create -n diatom python=3.11 -y
conda activate diatom
```

## 2) PyTorch with CUDA for the RTX 6000 Ada Generation (CUDA 12.1 build)
```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8 driver instead:
# pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

## 3) Remaining pinned dependencies
```bash
cd ~/Desktop/Diatoms                     # or ~/Masaüstü/Diatoms
grep -v -E '^torch==|^torchvision==' requirements.txt > /tmp/req_notorch.txt
pip install -r /tmp/req_notorch.txt
pip install fvcore                        # optional, enables FLOPs column
```
(Stripping the torch lines prevents pip from overwriting the CUDA build with a
CPU wheel from PyPI.)

## 4) Verify GPU + versions
```bash
python -c "import torch, torchvision; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| tv', torchvision.__version__)"
# expect: CUDA: True | GPU: NVIDIA RTX 6000 Ada Generation | torch 2.1.2 | tv 0.16.2
```

## 5) Data integrity (expect 12353/12353)
```bash
python - <<'PY'
import pandas as pd, os
d = pd.read_csv("data/diatom_metadata.csv")
missing = [p for p in d["filepath"] if not os.path.exists(p)]
print(f"{len(d)-len(missing)}/{len(d)} images found; missing={len(missing)}")
PY
```

## 6) Single-GPU config (match original batch 128)
```bash
export DIATOM_BATCH=128
export DIATOM_WORKERS=8
```

## 7) Smoke test (~1–2 min → SMOKE TEST PASSED)
```bash
python experiment_package/smoke_test.py
```

## 8) Full run (tmux; multi-day, survives disconnects)
```bash
tmux new -s diatom
conda activate diatom
cd ~/Desktop/Diatoms
export DIATOM_BATCH=128 DIATOM_WORKERS=8
bash experiment_package/run_all.sh 2>&1 | tee run_all.console.log
# detach: Ctrl+b then d   |   reattach: tmux attach -t diatom
# (alternative to tmux) nohup bash experiment_package/run_all.sh > run_all.console.log 2>&1 &
```

Outputs: `results_revision/aggregated/REVISION_RESULTS_SUMMARY.md` and the single
`results_revision_export.tar.gz` (bring that back for writing).

## Note on `nvidia-smi` "CUDA Version: 13"
That number is the **driver's** max supported CUDA runtime, not a requirement.
NVIDIA drivers are backward-compatible, so a CUDA-13 driver runs the cu121 (CUDA
12.1) PyTorch wheel fine. Keep `torch==2.1.2 ... cu121` (matches the original
results). Verify with `python -c "import torch; print(torch.cuda.is_available())"`
-> should print `True`.

Fallback ONLY if that prints False (unlikely): install a newer, uniform version
`pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124`
and note in the paper that all runs used torch 2.4.1 (internal consistency is
preserved because every run uses the same version).
