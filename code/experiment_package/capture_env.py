"""Capture the exact software/hardware environment for reproducibility."""
import os, sys, json, platform, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine

def run():
    out = os.path.join(engine.REPO_ROOT, "results_revision", "environment")
    os.makedirs(out, exist_ok=True)
    info = {"python": sys.version, "platform": platform.platform()}
    try:
        import torch, torchvision
        info.update({"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                     "gpu_count": torch.cuda.device_count(),
                     "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    except Exception as e:
        info["torch_error"] = str(e)
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=engine.REPO_ROOT).decode().strip()
    except Exception:
        info["git_commit"] = "n/a"
    json.dump(info, open(os.path.join(out, "environment.json"), "w"), indent=2)
    try:
        with open(os.path.join(out, "pip_freeze.txt"), "w") as f:
            subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=f)
    except Exception:
        pass
    print("environment captured ->", out)

if __name__ == "__main__":
    run()
