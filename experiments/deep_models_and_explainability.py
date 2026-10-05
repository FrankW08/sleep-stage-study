#!/usr/bin/env python
# coding: utf-8

# In[31]:

#Phase 3
import numpy as np, pandas as pd, time, math, torch, torch.nn as nn, torch.nn.functional as Fnn
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report, confusion_matrix
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "sleep-edf-database-expanded-1.0.0"

# Load data
X_path = ROOT/"X_sc_std.npy"      # preferred (standardized); fall back to X_sc_raw.npy if needed
if not X_path.exists(): X_path = ROOT/"X_sc_raw.npy"
X = np.load(X_path)               # (n, 4, 3000)
y = np.load(ROOT/"y_sc.npy")
groups = pd.read_csv(ROOT/"subjects_sc.csv")["recording_id"].str.slice(0,6).values  # subject key

print(X.shape, y.shape, np.unique(groups).size, "subjects")
NUM_CLASSES = 5


# In[4]:


def zscore_fit(x):   # x: (n, c, t)
    mu = x.mean(axis=(0,2), keepdims=True)
    sd = x.std(axis=(0,2), keepdims=True) + 1e-8
    return mu, sd
def zscore_apply(x, mu, sd): return (x - mu) / sd


# In[5]:


#Dataset
class EpochDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def make_loaders(Xtr, ytr, Xte, yte, bs=128, num_workers=0):
    return (
        DataLoader(EpochDataset(Xtr, ytr), batch_size=bs, shuffle=True,
                   num_workers=num_workers, pin_memory=False),
        DataLoader(EpochDataset(Xte, yte), batch_size=bs, shuffle=False,
                   num_workers=num_workers, pin_memory=False),
    )


# In[6]:


# Model A TIny 1D CNN
class TinyCNN(nn.Module):
    def __init__(self, in_ch=4, ncls=5, base=32, p=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, base,  kernel_size=7, padding=3), nn.BatchNorm1d(base), nn.ReLU(),
            nn.Conv1d(base, base,   kernel_size=7, padding=3), nn.BatchNorm1d(base), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(p),

            nn.Conv1d(base, base*2, kernel_size=5, padding=2), nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.Conv1d(base*2, base*2,kernel_size=5, padding=2), nn.BatchNorm1d(base*2), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(p),

            nn.Conv1d(base*2, base*4, kernel_size=3, padding=1, dilation=2), nn.BatchNorm1d(base*4), nn.ReLU(),
            nn.Conv1d(base*4, base*4, kernel_size=3, padding=1, dilation=2), nn.BatchNorm1d(base*4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.head = nn.Linear(base*4, ncls)
    def forward(self, x):
        x = self.net(x)             # (B, C, 1)
        x = x.squeeze(-1)           # (B, C)
        return self.head(x)


# In[7]:


# Model B Lightweight TCN
class TCNBlock(nn.Module):
    def __init__(self, ch, k=5, d=1, p=0.2):
        super().__init__()
        pad = (k - 1) * d // 2      # SAME-length padding (k should be odd)
        self.conv1 = nn.Conv1d(ch, ch, k, padding=pad, dilation=d)
        self.bn1   = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, k, padding=pad, dilation=d)
        self.bn2   = nn.BatchNorm1d(ch)
        self.drop  = nn.Dropout(p)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.drop(h)
        return x + h                # same length => safe residual

class TinyTCN(nn.Module):
    def __init__(self, in_ch=4, ncls=5, base=32, layers=4, p=0.2):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, base, kernel_size=3, padding=1)
        blocks = []
        for i in range(layers):
            blocks += [TCNBlock(base, k=5, d=2**i, p=p)]
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(base, ncls)
    def forward(self, x):
        x = F.relu(self.stem(x))
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


# In[8]:


#Training and evaluation
def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

def train_one(model, train_loader, val_loader, class_weights=None, lr=1e-3, epochs=12):
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    if class_weights is not None:
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    best = {"f1":-1, "state":None}
    for ep in range(1, epochs+1):
        model.train(); loss_sum=0; n=0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, weight=class_weights)
            optim.zero_grad(); loss.backward(); optim.step()
            loss_sum += loss.item()*len(yb); n += len(yb)
        sched.step()
        # quick val
        f1,k,_ = eval_model(model, val_loader)
        if f1>best["f1"]:
            best={"f1":f1,"state":{k:v.cpu() for k,v in model.state_dict().items()}}
        print(f"ep {ep:02d}  loss {loss_sum/n:.4f}  val macroF1 {f1:.3f}  kappa {k:.3f}")
    model.load_state_dict(best["state"])
    return model

@torch.no_grad()
def eval_model(model, loader):
    model.eval()
    y_true=[]; y_pred=[]
    for xb,yb in loader:
        xb=xb.to(device)
        logits = model(xb)
        y_true.append(yb.numpy()); y_pred.append(logits.argmax(1).cpu().numpy())
    y_true=np.concatenate(y_true); y_pred=np.concatenate(y_pred)
    macro = f1_score(y_true, y_pred, average="macro")
    kappa = cohen_kappa_score(y_true, y_pred)
    return macro, kappa, (y_true, y_pred)


# In[9]:


# 5-fold subjectwise CVrunner
def run_cv(ModelClass, model_name, epochs=12, bs=128, lr=1e-3):
    gkf = GroupKFold(n_splits=5)
    scores=[]
    fold=0
    for tr, te in gkf.split(X, y, groups):
        fold+=1
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        # if using raw, standardize with train stats only
        if "raw" in X_path.name:
            mu,sd = zscore_fit(Xtr); Xtr=zscore_apply(Xtr,mu,sd); Xte=zscore_apply(Xte,mu,sd)
        # class weights from train
        cw = np.bincount(ytr, minlength=NUM_CLASSES).astype(float)
        cw = cw.max()/cw;  # inverse freq-ish
        train_loader, val_loader = make_loaders(Xtr,ytr,Xte,yte, bs=bs)
        model = ModelClass()
        print(f"\n[{model_name}] Fold {fold}  params={count_params(model):,}")
        t0=time.time()
        model = train_one(model, train_loader, val_loader, class_weights=cw, lr=lr, epochs=epochs)
        t_train = time.time()-t0
        macro, kappa, (yt, yp) = eval_model(model, val_loader)
        # inference time per epoch (ms) on CPU/GPU
        xb = torch.from_numpy(Xte[:bs]).float().to(device)
        t1=time.time(); _=model(xb); torch.cuda.synchronize() if device.type=="cuda" else None
        infer_time = (time.time()-t1)/len(xb)*1000
        print(f"[{model_name}] Fold {fold}  macroF1={macro:.3f}  kappa={kappa:.3f}  infer={infer_time:.2f} ms/epoch")
        scores.append((macro,kappa,infer_time,count_params(model)))
    scores=np.array(scores)
    print(f"\n[{model_name}]  CV Macro-F1: {scores[:,0].mean():.3f}±{scores[:,0].std():.3f}  "
          f"Kappa: {scores[:,1].mean():.3f}±{scores[:,1].std():.3f}  "
          f"Infer: {scores[:,2].mean():.1f} ms   Params: ~{int(scores[:,3].mean()):,}")
    return scores


# In[10]:


cnn_scores = run_cv(TinyCNN, "TinyCNN", epochs=12, bs=128, lr=1e-3)
tcn_scores = run_cv(TinyTCN, "TinyTCN", epochs=12, bs=128, lr=1e-3)


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === fill in your final numbers here ===
results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost", "TinyCNN", "TinyTCN"],
    "MacroF1": [0.657, 0.667, 0.683, 0.659, 0.605],
    "CohenKappa": [0.595, 0.612, 0.626, 0.597, 0.534],
    "Params": [None, None, None, 114_501, 42_309],
    "Inference(ms/epoch)": [None, None, None, 0.5, 1.0]
})

# --- Macro-F1 barplot ---
plt.figure(figsize=(7,4))
sns.barplot(data=results, x="Model", y="MacroF1", palette="viridis")
plt.title("Model Comparison – Macro-F1 (Sleep-EDF)")
plt.ylabel("Macro-F1")
plt.ylim(0.5,0.75)
plt.xticks(rotation=30, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()

# --- Cohen’s κ barplot ---
plt.figure(figsize=(7,4))
sns.barplot(data=results, x="Model", y="CohenKappa", palette="magma")
plt.title("Model Comparison – Cohen’s κ (Sleep-EDF)")
plt.ylabel("Cohen’s κ")
plt.ylim(0.45,0.7)
plt.xticks(rotation=30, ha="right")
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.show()


# In[21]:


results.sort_values("MacroF1", ascending=False).round(3)
results.to_csv(ROOT / "phase3_model_comparison.csv", index=False)


# In[22]:


#PHASE4


# In[23]:


pick = ["EEG Fpz-Cz","EEG Pz-Oz","EOG horizontal","EMG submental"]
# -> channel indices
CH = {"EEG_FpzCz":0, "EEG_PzOz":1, "EOG":2, "EMG":3}


# In[36]:


# Helpers: subselect channels, add optional noise, CV on custom X
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, cohen_kappa_score
#import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# --- reuse these from before if already defined ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EpochDataset(Dataset):
    def __init__(self, X, y): self.X=torch.from_numpy(X).float(); self.y=torch.from_numpy(y).long()
    def __len__(self): return len(self.y)
    def __getitem__(self,i): return self.X[i], self.y[i]

def make_loaders(Xtr,ytr,Xte,yte,bs=128,workers=0):
    return (
        DataLoader(EpochDataset(Xtr,ytr), batch_size=bs, shuffle=True,  num_workers=workers, pin_memory=False),
        DataLoader(EpochDataset(Xte,yte), batch_size=bs, shuffle=False, num_workers=workers, pin_memory=False),
    )

@torch.no_grad()
def eval_model(model, loader):
    model.eval(); y_true=[]; y_pred=[]
    for xb, yb in loader:
        xb=xb.to(device); logits = model(xb)
        y_true.append(yb.numpy()); y_pred.append(logits.argmax(1).cpu().numpy())
    y_true=np.concatenate(y_true); y_pred=np.concatenate(y_pred)
    return (
        f1_score(y_true,y_pred,average="macro"),
        cohen_kappa_score(y_true,y_pred)
    )

def zscore_fit(X):  # if you’re using X_sc_raw.npy
    mu = X.mean(axis=(0,2), keepdims=True); sd = X.std(axis=(0,2), keepdims=True) + 1e-8
    return mu, sd
def zscore_apply(X, mu, sd): return (X - mu)/sd

def count_params(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)

# ---------- ablation utilities ----------
def select_channels(X, idx_list):
    """X: (n, C, T) -> subselect along channel dim; idx_list like [0] or [0,2]."""
    return X[:, idx_list, :].copy()

def add_gaussian_noise(X, sigma_rel=0.0, seed=0):
    """sigma_rel * per-channel std noise."""
    if sigma_rel<=0: return X
    rng = np.random.default_rng(seed)
    std = X.std(axis=(0,2), keepdims=True) + 1e-8
    noise = rng.normal(0, 1, size=X.shape).astype(np.float32) * (sigma_rel * std)
    return X + noise

def run_cv_on(ModelClass, X_in, y, groups, model_name, epochs=12, bs=128, lr=1e-3):
    gkf = GroupKFold(n_splits=5)
    scores=[]
    for fold,(tr,te) in enumerate(gkf.split(X_in,y,groups),1):
        Xtr, ytr, Xte, yte = X_in[tr], y[tr], X_in[te], y[te]

        # standardize if this is raw (skip if you loaded X_sc_std.npy)
        if 'raw' in str(X_path.name).lower():
            mu,sd = zscore_fit(Xtr); Xtr=zscore_apply(Xtr,mu,sd); Xte=zscore_apply(Xte,mu,sd)

        # class weights
        cw = np.bincount(ytr, minlength=5).astype(float); cw = cw.max()/cw
        cw_t = torch.tensor(cw, dtype=torch.float32, device=device)

        train_loader, val_loader = make_loaders(Xtr,ytr,Xte,yte,bs=bs)

        model = ModelClass().to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

        best_f1=-1; best_state=None
        for ep in range(1, epochs+1):
            model.train(); loss_sum=n=0
            for xb,yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb, weight=cw_t)
                optim.zero_grad(); loss.backward(); optim.step()
                loss_sum += loss.item()*len(yb); n+=len(yb)
            sched.step()
            f1,k = eval_model(model, val_loader)
            if f1>best_f1: best_f1=f1; best_state={k:v.cpu() for k,v in model.state_dict().items()}
        model.load_state_dict(best_state)
        f1,k = eval_model(model, val_loader)

        # quick CPU/GPU inference timing
        xb = torch.from_numpy(Xte[:min(len(Xte),128)]).float().to(device)
        t0 = torch.cuda.Event(enable_timing=True) if device.type=="cuda" else None
        t1 = torch.cuda.Event(enable_timing=True) if device.type=="cuda" else None
        if device.type=="cuda":
            t0.record(); _=model(xb); t1.record(); torch.cuda.synchronize()
            infer = t0.elapsed_time(t1)/len(xb)  # ms/sample
        else:
            import time; s=time.time(); _=model(xb); infer=(time.time()-s)/len(xb)*1000
        scores.append((f1,k,infer,count_params(model)))
        print(f"[{model_name}] fold {fold}: MacroF1={f1:.3f}, κ={k:.3f}, infer={infer:.2f} ms, params={scores[-1][3]:,}")
    scores=np.array(scores)
    print(f"[{model_name}]  CV MacroF1 {scores[:,0].mean():.3f}±{scores[:,0].std():.3f}, "
          f"κ {scores[:,1].mean():.3f}±{scores[:,1].std():.3f}, "
          f"infer ~{scores[:,2].mean():.2f} ms, params ~{int(scores[:,3].mean())}")
    return scores


# In[25]:


# Define the ablation config
# Channel indices based on your pick order:
CH = {"EEG_FpzCz":0, "EEG_PzOz":1, "EOG":2, "EMG":3}

ABLATIONS = [
    ("EEG-only (Fpz-Cz)",         [CH["EEG_FpzCz"]]),
    ("EEG-only (both EEGs)",      [CH["EEG_FpzCz"], CH["EEG_PzOz"]]),
    ("EEG + EOG",                 [CH["EEG_FpzCz"], CH["EOG"]]),
    ("Full PSG (EEG+EEG+EOG+EMG)",[CH["EEG_FpzCz"], CH["EEG_PzOz"], CH["EOG"], CH["EMG"]]),
]


# In[26]:


# ===================== Phase 4 Ablation Runner (prints + displays) =====================
import time, traceback, numpy as np, pandas as pd
from IPython.display import display, HTML, clear_output

# --------------------- knobs ---------------------
MODELS = ["TinyCNN", "TinyTCN"]   # or ["TinyCNN"] if you want faster runs
QUICK   = True                    # True = fast smoke test; False = full run
EPOCHS  = 2 if QUICK else 12
BATCH   = 64 if QUICK else 128
NOISE   = 0.0                     # e.g., 0.1 to simulate wearables
# -------------------------------------------------

# small helper: model factory resolver
def model_factory(tag, in_ch):
    if tag == "TinyCNN":
        return lambda: TinyCNN(in_ch=in_ch)
    if tag == "TinyTCN":
        return lambda: TinyTCN(in_ch=in_ch)
    raise ValueError(f"Unknown model tag: {tag}")

phase4_rows = []

def run_ablation(tag):
    global phase4_rows
    for name, idxs in ABLATIONS:
        # subset channels + optional noise
        X_s = select_channels(X, idxs).copy()
        X_s = add_gaussian_noise(X_s, sigma_rel=NOISE)

        in_ch = int(X_s.shape[1])          # recompute per config
        banner = f"{tag} | {name} | ch={idxs} | X={tuple(X_s.shape)} | epochs={EPOCHS}"
        print("\n=== " + banner, flush=True)

        try:
            # CV run
            scores = run_cv_on(
                model_factory(tag, in_ch),
                X_s, y, groups,
                model_name=f"{tag}-{name}",
                epochs=EPOCHS, bs=BATCH, lr=1e-3
            )

            # append one row
            phase4_rows.append([
                name, tag, in_ch,
                float(scores[:,0].mean()),  # MacroF1
                float(scores[:,1].mean()),  # Kappa
                float(scores[:,2].mean()),  # Infer_ms
                int(scores[:,3].mean())     # Params
            ])

            # live table while running
            tmp = (pd.DataFrame(
                phase4_rows,
                columns=["Config","Model","nCh","MacroF1","Kappa","Infer_ms","Params"]
            ).sort_values(["Model","nCh"]).reset_index(drop=True).round(3))

            clear_output(wait=True)
            display(HTML("<h3>Phase 4 – Channel Ablation (running)</h3>"))
            display(tmp.style.set_caption("Partial results (auto-updates)"))

        except Exception as e:
            print("❌ Error while running:", banner)
            traceback.print_exc()
            # continue to next config instead of stopping everything
            continue

# --------------------- run selection ---------------------
t0 = time.time()
if "TinyCNN" in MODELS: run_ablation("TinyCNN")
if "TinyTCN" in MODELS: run_ablation("TinyTCN")
t1 = time.time()

# --------------------- final table ---------------------
phase4 = (pd.DataFrame(
    phase4_rows,
    columns=["Config","Model","nCh","MacroF1","Kappa","Infer_ms","Params"]
).sort_values(["Model","nCh"]).reset_index(drop=True).round(3))

clear_output(wait=True)
display(HTML(f"<h3>Phase 4 – Channel Ablation Results "
             f"(done in {t1-t0:.1f}s, QUICK={QUICK})</h3>"))
display(phase4.style.set_table_styles(
    [{'selector':'th','props':[('background','#f0f0f0'),('font-weight','bold')]}]
).set_caption("Performance vs Channel Count"))

# save
out_path = ROOT/"phase4_ablation_results.csv"
phase4.to_csv(out_path, index=False)
print("✅ Saved ->", out_path)
# ===============================================================================


# In[1]:


# -----------------------
# Phase 5 - SHAP Setup
# -----------------------

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import torch.nn.functional as Fnn   # safe alias

# -----------------------
# 1. Check that F is your feature matrix
# -----------------------

# F MUST be a NumPy array like (N, D)
print("Type of F:", type(F))

# If this errors or prints something wrong, F is overwritten.
# In that case, run:  %whos  to find the real feature matrix.


# -----------------------
# 2. Build DataFrame for SHAP
# -----------------------

n_features = F.shape[1]                       # Number of feature columns
feature_names = [f"f{i}" for i in range(n_features)]  # Simple column names

# Convert NumPy -> Pandas DataFrame
X_df = pd.DataFrame(F, columns=feature_names)

print("X_df shape:", X_df.shape)
print("Example feature names:", feature_names[:10])


# -----------------------
# 3. Initialize SHAP (for better Jupyter visualization)
# -----------------------

shap.initjs()


# In[ ]:


import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# ---------------------------------------------------
# 1. Prepare X and feature names (F is your numpy array)
# ---------------------------------------------------
feature_names = [f"f{i}" for i in range(F.shape[1])]
X = pd.DataFrame(F, columns=feature_names)
class_names = ["W", "N1", "N2", "N3", "REM"]

# ---------------------------------------------------
# 2. Fit Random Forest on ALL data
# ---------------------------------------------------
rf_global = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)
rf_global.fit(X, y)

shap.initjs()

# ---------------------------------------------------
# 3. SHAP values for RF
# ---------------------------------------------------
explainer_rf = shap.TreeExplainer(rf_global)

n_sample = min(2000, len(X))
X_sample = shap.sample(X, n_sample, random_state=0)

shap_values_rf = explainer_rf.shap_values(X_sample)
print("Raw shap_values_rf type:", type(shap_values_rf))

# ----- Standardize to array with shape (n_samples, n_features, n_classes)
if isinstance(shap_values_rf, list):
    # Old SHAP style: list of length n_classes, each (n_samples, n_features)
    # Stack to (n_classes, n_samples, n_features) → then transpose
    sv_arr = np.stack(shap_values_rf, axis=0)        # (C, N, F)
    sv_arr = np.transpose(sv_arr, (1, 2, 0))         # (N, F, C)
else:
    # Newer style: already a numpy array
    sv_arr = np.array(shap_values_rf)                # e.g. (N, F, C) or (C, N, F)
    if sv_arr.shape[2] == len(class_names):
        # (N, F, C)
        pass
    elif sv_arr.shape[0] == len(class_names):
        # (C, N, F) → (N, F, C)
        sv_arr = np.transpose(sv_arr, (1, 2, 0))
    else:
        raise RuntimeError(f"Unexpected SHAP shape: {sv_arr.shape}")

print("Standardized RF SHAP shape:", sv_arr.shape)   # should be (n_sample, n_features, n_classes)

N, Fdim, C = sv_arr.shape

# ---------------------------------------------------
# 4. Global Feature Importance (all classes combined)
# ---------------------------------------------------
# mean(|SHAP|) across classes → (N, F)
vals_all = np.mean(np.abs(sv_arr), axis=2)

shap.summary_plot(
    vals_all,
    X_sample,
    feature_names=feature_names,
    show=False
)
plt.title("Random Forest – Global Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("shap_rf_global_summary.png", dpi=220)
plt.close()

# ---------------------------------------------------
# 5. Per-class Feature Importance
# ---------------------------------------------------
for c, name in enumerate(class_names):
    sv_c = sv_arr[:, :, c]      # (N, F) for class c

    shap.summary_plot(
        sv_c,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title(f"Random Forest – Class {name} Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(f"shap_rf_class_{name}.png", dpi=220)
    plt.close()

# ---------------------------------------------------
# 6. Single-sample Waterfall Plot
# ---------------------------------------------------
i = 0   # pick any sample index from 0 .. N-1
x_row = X_sample.iloc[[i]]

row_sv = explainer_rf.shap_values(x_row)

# Standardize single-sample SHAP to (features, classes)
if isinstance(row_sv, list):
    # list length C, each (1, F)
    row_arr = np.stack(row_sv, axis=0)         # (C, 1, F)
    row_arr = np.transpose(row_arr, (1, 2, 0)) # (1, F, C)
    row_arr = row_arr[0]                       # (F, C)
else:
    row_arr = np.array(row_sv)
    if row_arr.ndim == 2:                      # (F, ) binary case
        row_arr = row_arr[:, None]             # (F, 1)
    elif row_arr.shape[2] == C:
        row_arr = row_arr[0]                   # (F, C)
    elif row_arr.shape[0] == C:
        row_arr = row_arr[:, 0, :]             # (C, F) → (C, F)
        row_arr = row_arr.T                    # (F, C)
    else:
        raise RuntimeError(f"Unexpected row SHAP shape: {row_arr.shape}")

# Choose predicted class for this sample
pred_class = int(rf_global.predict(x_row)[0])

shap.plots._waterfall.waterfall_legacy(
    explainer_rf.expected_value[pred_class],
    row_arr[:, pred_class],
    feature_names=feature_names,
    show=False
)
plt.title(f"Random Forest – Waterfall for Sample {i} (class {class_names[pred_class]})")
plt.tight_layout()
plt.savefig("shap_rf_sample_waterfall.png", dpi=220)
plt.close()


# In[ ]:




