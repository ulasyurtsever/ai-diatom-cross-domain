"""EXP4 — inference / efficiency metrics.  No training.
Measures, per backbone at its native input size:
  parameters (M), FLOPs/MACs (G), model size (MB), peak GPU memory (MB),
  single-image latency mean/std (ms), and throughput (images/s).
Outputs -> results_revision/efficiency/efficiency_metrics.csv
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, torch
import engine, config_revision as C

N_CLS = 27   # LOSO core class count (head size does not affect these metrics materially)


def flops_of(model, x):
    try:
        from fvcore.nn import FlopCountAnalysis
        return float(FlopCountAnalysis(model, x).total()) / 1e9   # GFLOPs (MACs)
    except Exception:
        pass
    try:
        from thop import profile
        macs, _ = profile(model, inputs=(x,), verbose=False)
        return float(macs) / 1e9
    except Exception:
        return float("nan")


def run():
    out_dir = os.path.join(engine.REPO_ROOT, C.RESULTS_SUBDIR, "efficiency")
    os.makedirs(out_dir, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for arch in C.INFER_MODELS:
        model, base_tf = engine.base.build_model(arch, N_CLS)
        model = model.to(dev).eval()
        res = engine.native_size(base_tf)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

        x1 = torch.randn(1, 3, res, res, device=dev)
        gflops = flops_of(model, x1)

        # latency (batch=1)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(C.INFER_WARMUP):
                model(x1)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            ts = []
            for _ in range(C.INFER_ITERS):
                t0 = time.time(); model(x1)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                ts.append((time.time() - t0) * 1000)
        lat_mean, lat_std = float(np.mean(ts)), float(np.std(ts))
        peak_mb = (torch.cuda.max_memory_allocated() / 1024**2) if dev.type == "cuda" else float("nan")

        # throughput (batched)
        b = C.INFER_BATCH_FOR_THROUGHPUT
        xb = torch.randn(b, 3, res, res, device=dev)
        with torch.no_grad():
            for _ in range(5):
                model(xb)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time(); n = 0
            for _ in range(C.INFER_ITERS):
                model(xb); n += b
            if dev.type == "cuda":
                torch.cuda.synchronize()
            fps = n / (time.time() - t0)

        rows.append({"Model": arch, "InputRes": res, "Params_M": round(params, 2),
                     "ModelSize_MB": round(size_mb, 1), "GFLOPs": round(gflops, 2) if gflops == gflops else "",
                     "Latency_ms_mean": round(lat_mean, 3), "Latency_ms_std": round(lat_std, 3),
                     "Throughput_img_s": round(fps, 1),
                     "PeakGPUMem_MB": round(peak_mb, 1) if peak_mb == peak_mb else "",
                     "Device": engine.HW["INFO"]})
        print(f"  {arch:18s} params={params:.1f}M  GFLOPs={gflops:.2f}  lat={lat_mean:.2f}ms  fps={fps:.0f}")
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "efficiency_metrics.csv"), index=False)
    print(f"written: {out_dir}/efficiency_metrics.csv")


if __name__ == "__main__":
    run()
