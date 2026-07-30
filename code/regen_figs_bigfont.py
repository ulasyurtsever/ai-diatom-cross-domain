"""Regenerate submission figures 2,3,4,5,7 from canonical multi-seed result data with
enlarged, consistent fonts. Numbers are unchanged; only the
typographic theme differs. Writes figureN.png into regen_figures/ and
submission_v22/. Prints anchor values for verification.
"""
import os, glob, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGG  = os.path.join(ROOT, "results_revision", "aggregated")
MS   = os.path.join(ROOT, "results_revision", "multiseed")
OUT_DIRS = [os.path.join(ROOT, "regen_figures"), os.path.join(ROOT, "submission_v22")]
SEEDS = ["seed_42", "seed_202", "seed_1337"]
DOMAINS = ["ADIAC_Database","AFD_Database","DIA_Database","DONA_Database","FCE_LTER_Database","LOIR_Database"]
DOM_SHORT = {d: d.replace("_Database","").replace("FCE_LTER","FCE_LTER") for d in DOMAINS}
MODELS = ["resnet50","efficientnet_v2_s","convnext_tiny","swin_v2_t","maxvit_t"]
SHORT = {"resnet50":"ResNet-50","efficientnet_v2_s":"EfficientNetV2-S","convnext_tiny":"ConvNeXt-Tiny",
         "swin_v2_t":"Swin V2-Tiny","maxvit_t":"MaxViT-Tiny"}

plt.rcParams.update({
    "font.size": 17, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
    "figure.dpi": 300, "savefig.bbox": "tight", "axes.titleweight": "bold",
})

def save(fig, name):
    for d in OUT_DIRS:
        os.makedirs(d, exist_ok=True)
        fig.savefig(os.path.join(d, name), dpi=300, bbox_inches="tight")
    plt.close(fig)

def preds(model, dom, seed):
    p = os.path.join(MS, seed, f"S2_LOSO_{model}_{dom}", "test_predictions.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

# ---------- Figure 2: generalization gap ----------
def figure2():
    S = pd.read_csv(os.path.join(AGG,"multiseed_SINGLE_mean_std.csv")).set_index("Model")
    T = pd.read_csv(os.path.join(AGG,"multiseed_STANDARD_mean_std.csv")).set_index("Model")
    L = pd.read_csv(os.path.join(AGG,"multiseed_LOSO_mean_std.csv")).set_index("Model")
    order = L["Acc_mean"].sort_values(ascending=False).index.tolist()
    x = np.arange(len(order)); w = 0.26
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ss  = [S.loc[m,"Acc_mean"]*100 for m in order]
    sm  = [T.loc[m,"Acc_mean"]*100 for m in order]
    lo  = [L.loc[m,"Acc_mean"]*100 for m in order]
    loe = [L.loc[m,"Acc_std"]*100 for m in order]
    ax.bar(x-w, ss, w, label="Single-Source (ADIAC)", color="#2f5d9e")
    ax.bar(x,   sm, w, label="Standard Mixed (in-distribution upper bound)", color="#5aa9d6")
    ax.bar(x+w, lo, w, yerr=loe, capsize=6, label="LOSO (mean ± SD over 6 held-out sources)", color="#d1495b")
    for i,m in enumerate(order):
        ytop = max(ss[i], sm[i])
        ax.annotate(f"Δ {ss[i]-lo[i]:.1f}", (x[i], ytop), textcoords="offset points",
                    xytext=(0,6), ha="center", fontsize=14, color="#222", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[m] for m in order], rotation=12)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0,105)
    ax.legend(loc="upper right", framealpha=.95); ax.grid(axis="y", alpha=.3)
    save(fig, "figure2.png")
    print(f"[fig2] ConvNeXt LOSO={L.loc['convnext_tiny','Acc_mean']*100:.1f}  ResNet LOSO={L.loc['resnet50','Acc_mean']*100:.1f}")

# ---------- Figure 3: per-domain accuracy heatmap ----------
def figure3():
    mean = pd.DataFrame(index=MODELS, columns=DOMAINS, dtype=float)
    sd   = pd.DataFrame(index=MODELS, columns=DOMAINS, dtype=float)
    for m in MODELS:
        for d in DOMAINS:
            accs=[]
            for s in SEEDS:
                df=preds(m,d,s)
                if df is not None:
                    accs.append((df["True_Label"]==df["Predicted_Label"]).mean()*100)
            mean.loc[m,d]=np.mean(accs); sd.loc[m,d]=np.std(accs,ddof=1)
    annot = mean.round(1).astype(str) + "\n±" + sd.round(1).astype(str)
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.heatmap(mean.astype(float), annot=annot.values, fmt="", cmap="Blues", ax=ax,
                vmin=0, vmax=80, linewidths=0.5, linecolor="white",
                cbar_kws={"label":"Held-out accuracy (%)"}, annot_kws={"size":14})
    ax.set_xticklabels([DOM_SHORT[d] for d in DOMAINS], rotation=20, ha="right")
    ax.set_yticklabels([SHORT[m] for m in MODELS], rotation=0)
    ax.set_xlabel("Held-out target domain (LOSO)"); ax.set_ylabel("Architecture")
    save(fig, "figure3.png")
    print(f"[fig3] ResNet LOIR={mean.loc['resnet50','LOIR_Database']:.1f}  ConvNeXt LOIR={mean.loc['convnext_tiny','LOIR_Database']:.1f}")

# ---------- Figure 4: class imbalance vs per-class F1 (ConvNeXt) ----------
def figure4():
    meta = pd.read_csv(os.path.join(ROOT,"data","diatom_metadata.csv"))
    core = meta[meta["is_dg_class"]==True] if "is_dg_class" in meta.columns else meta
    tot = core.groupby("label").size()
    per_dom = core.groupby(["label","domain"]).size().unstack(fill_value=0)
    # mean training images per class across 6 folds = total - held_out (avg over folds)
    mean_train = {}
    for c in tot.index:
        vals=[tot[c]-per_dom.loc[c,d] if d in per_dom.columns else tot[c] for d in DOMAINS]
        mean_train[c]=np.mean(vals)
    # per-class F1 for ConvNeXt: mean over seeds x domains
    cls_f1={}
    for d in DOMAINS:
        for s in SEEDS:
            df=preds("convnext_tiny",d,s)
            if df is None: continue
            names=df.columns.tolist()[2:]
            yt=df["True_Label"].values; yp=df["Predicted_Label"].values
            f1=f1_score(yt,yp,labels=list(range(len(names))),average=None,zero_division=0)
            for i,nm in enumerate(names):
                if (yt==i).sum()>0:
                    cls_f1.setdefault(nm,[]).append(f1[i]*100)
    rows=[(c, mean_train.get(c,np.nan), np.mean(v)) for c,v in cls_f1.items() if c in mean_train]
    dfp=pd.DataFrame(rows,columns=["cls","train","f1"]).dropna()
    fig, ax = plt.subplots(figsize=(12, 7))
    sc=ax.scatter(dfp["train"], dfp["f1"], c=dfp["f1"], cmap="viridis", s=130, edgecolor="k", linewidth=.4)
    for _,r in dfp.iterrows():
        ax.annotate(r["cls"], (r["train"], r["f1"]), fontsize=11, xytext=(4,3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("Mean training images per class (log scale)")
    ax.set_ylabel("LOSO Macro-F1 per class (%)")
    cb=fig.colorbar(sc, ax=ax); cb.set_label("Per-class F1 (%)")
    ax.grid(alpha=.3)
    save(fig, "figure4.png")
    print(f"[fig4] n classes={len(dfp)}  F1 range={dfp['f1'].min():.0f}-{dfp['f1'].max():.0f}")

# ---------- Figure 5: accuracy vs inference latency ----------
def figure5():
    e=pd.read_csv(os.path.join(AGG,"efficiency_metrics.csv")).set_index("Model")
    L=pd.read_csv(os.path.join(AGG,"multiseed_LOSO_mean_std.csv")).set_index("Model")
    fig, ax = plt.subplots(figsize=(10, 7))
    colors=plt.cm.tab10(np.linspace(0,1,len(MODELS)))
    for m,c in zip(MODELS,colors):
        if m in e.index and m in L.index:
            xx=e.loc[m,"Latency_ms_mean"]; yy=L.loc[m,"F1_mean"]*100
            ax.scatter(xx,yy,s=240,color=c,edgecolor="k",zorder=3)
            ax.annotate(SHORT[m],(xx,yy),xytext=(8,6),textcoords="offset points",fontsize=15)
    ax.set_xlabel("Inference latency (ms / image, batch = 1)")
    ax.set_ylabel("LOSO Macro-F1 (%)"); ax.set_title("Accuracy vs inference cost (LOSO)")
    ax.grid(alpha=.3)
    save(fig, "figure5.png")
    print(f"[fig5] ConvNeXt lat={e.loc['convnext_tiny','Latency_ms_mean']:.2f} F1={L.loc['convnext_tiny','F1_mean']*100:.1f}")

# ---------- Figure 7: confusion (ConvNeXt seed_42) ----------
def figure7():
    rows=[]
    for d in DOMAINS:
        df=preds("convnext_tiny",d,"seed_42")
        if df is None: continue
        cols=df.columns.tolist()[2:]
        for t,p in zip(df["True_Label"].astype(int),df["Predicted_Label"].astype(int)):
            if t<len(cols) and p<len(cols): rows.append((cols[t],cols[p]))
    big=pd.DataFrame(rows,columns=["True","Pred"]); mis=big[big["True"]!=big["Pred"]]
    classes=sorted(set(mis["True"].value_counts().head(12).index)|set(mis["Pred"].value_counts().head(12).index))
    sub=big[(big["True"].isin(classes))&(big["Pred"].isin(classes))]
    cm=pd.crosstab(sub["True"],sub["Pred"],normalize="index")*100
    cm=cm.reindex(index=classes,columns=classes,fill_value=0)
    fig, ax = plt.subplots(figsize=(12.5, 10))
    sns.heatmap(cm,annot=True,fmt=".0f",cmap="OrRd",ax=ax,vmin=0,vmax=80,
                linewidths=0.3,linecolor="white",annot_kws={"size":11},
                cbar_kws={"label":"Row-normalized prediction frequency (%)"})
    ax.set_xlabel("Predicted genus / class"); ax.set_ylabel("True genus / class")
    plt.setp(ax.get_xticklabels(),rotation=45,ha="right"); plt.setp(ax.get_yticklabels(),rotation=0)
    save(fig, "figure7.png")
    print(f"[fig7] Sell->Nav={cm.loc['Sellaphora','Navicula']:.0f} Cymb->Gomph={cm.loc['Cymbella','Gomphonema']:.0f}")

if __name__=="__main__":
    figure2(); figure3(); figure4(); figure5(); figure7()
    print("done -> figureN.png written to regen_figures/ and submission_v22/")
