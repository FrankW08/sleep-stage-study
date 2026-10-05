#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import mne, numpy as np, pandas as pd
import re


# In[2]:


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "sleep-edf-database-expanded-1.0.0"
SC   = ROOT / "sleep-cassette"

psg_files = sorted(SC.glob("*-PSG.edf"))

def expected_hyp(psg_path: Path) -> Path:
    base = psg_path.stem.split("-")[0]     # e.g., SC4001E0
    prefix, last = base[:-1], base[-1]     # SC4001E , 0
    hyp_tag = "EC" if last == "0" else "EH"  # E0→EC, E1→EH
    hyp_name = f"{prefix}{hyp_tag}-Hypnogram.edf"
    return psg_path.with_name(hyp_name)

pairs = []
missing = []

for psg in psg_files:
    hyp = expected_hyp(psg)
    if hyp.exists():
        pairs.append((psg, hyp))
    else:
        # fallback: if the expected one is missing, try any Hypnogram sharing the night id (first 7 chars)
        night_key = psg.stem.split("-")[0][:7]   # SC4001E
        cands = list(SC.glob(f"{night_key}?-Hypnogram.edf"))
        if cands:
            pairs.append((psg, sorted(cands)[0]))
        else:
            missing.append(psg.name)

print(f"PSG files: {len(psg_files)}")
print(f"Paired:    {len(pairs)}")
print(f"Missing:   {len(missing)}")
pairs[:3], missing[:5]


# In[3]:


LABEL_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}
KEEP = set(LABEL_MAP)
CLASS_ORDER = ["W","N1","N2","N3","REM"]
CLASS_TO_ID = {c:i for i,c in enumerate(CLASS_ORDER)}

def epoch_one(psg_path, hyp_path, epoch_len=30.0,
              pick=("EEG Fpz-Cz","EEG Pz-Oz","EOG horizontal","EMG submental")):
    # Load PSG
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    raw.pick([ch for ch in pick if ch in raw.ch_names])

    # Load hypnogram and convert to a single Annotations object with mapped labels
    hyp = mne.read_annotations(hyp_path)  # no verbose kw on older MNE
    onsets, durations, desc = [], [], []
    for i in range(len(hyp)):
        d = hyp.description[i]
        if d in KEEP:
            onsets.append(hyp.onset[i])
            durations.append(hyp.duration[i])
            desc.append(LABEL_MAP[d])

    if not onsets:   # no usable labels
        return None, None

    ann = mne.Annotations(onset=onsets,
                          duration=durations,
                          description=desc,
                          orig_time=getattr(hyp, "orig_time", None))
    raw.set_annotations(ann)

    # Events from annotations using our fixed event id
    events, _ = mne.events_from_annotations(raw, event_id=CLASS_TO_ID, verbose=False)

    # Epoch into 30 s segments (slightly <30s to avoid right-edge issues)
    epochs = mne.Epochs(raw, events, CLASS_TO_ID,
                        tmin=0, tmax=30 - 1/raw.info["sfreq"],
                        baseline=None, preload=True,
                        reject_by_annotation=True, verbose=False)

    X = epochs.get_data()       # (n_epochs, n_channels, n_samples)
    y = epochs.events[:, -1]    # integers 0..4
    return X, y


# In[34]:


def epoch_one(psg_path, hyp_path, epoch_len=30.0,
              pick=("EEG Fpz-Cz","EEG Pz-Oz","EOG horizontal","EMG submental")):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    # unify/strip channel names then pick
    raw.rename_channels({ch: ch.strip() for ch in raw.ch_names})
    chans = [ch for ch in pick if ch in raw.ch_names]
    if not chans:
        raise RuntimeError(f"No expected channels found in {psg_path.name}. Found: {raw.ch_names}")
    raw.pick(chans)

    hyp = mne.read_annotations(hyp_path)  # no 'verbose' arg on older MNE
    onsets, durations, desc = [], [], []
    for i in range(len(hyp)):
        d = hyp.description[i]
        if d in KEEP:
            onsets.append(hyp.onset[i])
            durations.append(hyp.duration[i])
            desc.append(LABEL_MAP[d])
    if not onsets:
        return None, None

    ann = mne.Annotations(onset=onsets, duration=durations, description=desc,
                          orig_time=getattr(hyp, "orig_time", None))
    raw.set_annotations(ann)

    events, _ = mne.events_from_annotations(raw, event_id=CLASS_TO_ID, verbose=False)
    epochs = mne.Epochs(raw, events, CLASS_TO_ID,
                        tmin=0, tmax=30 - 1/raw.info["sfreq"],
                        baseline=None, preload=True, reject_by_annotation=True, verbose=False)
    X = epochs.get_data()
    y = epochs.events[:, -1]
    return X, y

# --- smoke test on first pair (guaranteed to exist due to assert)
psg0, hyp0 = pairs[0]
print("Testing:", psg0.name, "<->", hyp0.name)
X0, y0 = epoch_one(psg0, hyp0)
print("X0:", X0.shape)
print("y0 counts:", np.bincount(y0, minlength=5))


# In[4]:


from pathlib import Path
import numpy as np

OUT = ROOT / "derived_sc"
OUT.mkdir(exist_ok=True)

def save_npz(psg, hyp):
    rec_id = psg.stem.split("-")[0]     # e.g., SC4001E0
    X, y = epoch_one(psg, hyp)
    if X is None:
        print("Skipped (no usable labels):", rec_id)
        return 0
    np.savez_compressed(OUT / f"{rec_id}.npz", X=X, y=y)
    print(f"{rec_id}: {X.shape[0]} epochs")
    return X.shape[0]

total_epochs = sum(save_npz(psg, hyp) for psg, hyp in pairs)
print("Total epochs saved:", total_epochs)


# In[5]:


import numpy as np, pandas as pd

Xs, ys, subj = [], [], []
for f in sorted(OUT.glob("*.npz")):
    d = np.load(f)
    Xs.append(d["X"]); ys.append(d["y"])
    subj += [f.stem]*len(d["y"])

X = np.concatenate(Xs, axis=0)
y = np.concatenate(ys, axis=0)
np.save(ROOT/"X_sc_raw.npy", X)
np.save(ROOT/"y_sc.npy", y)
pd.Series(subj, name="recording_id").to_csv(ROOT/"subjects_sc.csv", index=False)

print("Final arrays:", X.shape, y.shape)


# In[6]:


import pandas as pd, matplotlib.pyplot as plt

name_map = {i:c for c,i in CLASS_TO_ID.items()}
pd.Series(y).map(name_map).value_counts()[["W","N1","N2","N3","REM"]].plot(kind="bar")
plt.title("Sleep stage distribution (SC)"); plt.ylabel("epochs"); plt.show()


# In[7]:


X_std = X.copy()
for c in range(X_std.shape[1]):  # per-channel across all epochs
    flat = X_std[:, c, :].reshape(-1)
    X_std[:, c, :] = (X_std[:, c, :] - flat.mean()) / (flat.std() + 1e-8)
np.save(ROOT/"X_sc_std.npy", X_std)


# In[8]:


from scipy.signal import welch
fs = mne.io.read_raw_edf(pairs[0][0], preload=False, verbose=False).info["sfreq"]

def bandpowers(sig, fs, bands=((0.5,4),(4,8),(8,12),(12,30))):
    f, Pxx = welch(sig, fs=fs, nperseg=int(fs*2))
    return np.array([Pxx[(f>=lo)&(f<hi)].sum() for lo,hi in bands])

F = []
for i in range(X.shape[0]):
    F.append(np.concatenate([bandpowers(X[i,c], fs) for c in range(X.shape[1])]))
F = np.vstack(F)
np.save(ROOT/"F_sc_bandpower.npy", F)
print("Feature matrix:", F.shape)  # (n_epochs, n_channels*4bands)


# In[9]:


#phase 2
import numpy as np, pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "sleep-edf-database-expanded-1.0.0"

# Required files from Phase 1
X_path = ROOT/"X_sc_raw.npy"          # (n_epochs, 4, 3000)
y_path = ROOT/"y_sc.npy"              # (n_epochs,)
subj_csv = ROOT/"subjects_sc.csv"     # column 'recording_id' (e.g., 'SC4001E0')

X = np.load(X_path)         # if memory is tight, you can skip time features below and just use F_sc_bandpower.npy
y = np.load(y_path)
subj_rec = pd.read_csv(subj_csv)["recording_id"].astype(str)

# Group by subject (first 6 chars, e.g., SC4001)
groups = subj_rec.str.slice(0, 6)
print(X.shape, y.shape, groups.nunique(), "unique subjects")


# In[10]:


from scipy.signal import welch

fs = 100.0  # Sleep-EDF EEG/EOG/EMG are typically 100 Hz
bands = [(0.5,4), (4,8), (8,12), (12,30)]
band_names = ["delta","theta","alpha","beta"]

def time_features(sig):
    # sig shape: (samples,)
    mean  = sig.mean()
    std   = sig.std()
    rms   = np.sqrt((sig**2).mean())
    zcr   = ((sig[:-1] * sig[1:]) < 0).mean()   # zero-crossing rate
    p2p   = sig.max() - sig.min()
    return np.array([mean, std, rms, zcr, p2p])

def freq_features(sig, fs=fs):
    f, Pxx = welch(sig, fs=fs, nperseg=int(fs*2))
    # absolute power per band
    abs_pow = np.array([Pxx[(f>=lo)&(f<hi)].sum() for lo,hi in bands])
    total = Pxx[(f>=0.5)&(f<30)].sum() + 1e-12
    rel_pow = abs_pow / total
    # spectral entropy on 0.5-30 Hz normalized PSD
    psd = Pxx[(f>=0.5)&(f<30)]
    psd = psd / (psd.sum() + 1e-12)
    sent = -(psd * np.log(psd + 1e-12)).sum()
    # ratios Δ/Θ, Θ/Α, Α/Β (safe divide)
    ratios = np.array([
        abs_pow[0]/(abs_pow[1]+1e-12),
        abs_pow[1]/(abs_pow[2]+1e-12),
        abs_pow[2]/(abs_pow[3]+1e-12),
    ])
    # concat: abs(4) + rel(4) + entropy(1) + ratios(3) = 12
    return np.concatenate([abs_pow, rel_pow, [sent], ratios])

def features_one_epoch(epoch):
    # epoch shape: (n_channels, n_samples)
    feats = []
    for c in range(epoch.shape[0]):
        sig = epoch[c]
        feats.append(np.concatenate([time_features(sig), freq_features(sig, fs=fs)]))
    return np.concatenate(feats)  # per channel stacked

# Build features for all epochs (this can take a couple of minutes)
F = np.vstack([features_one_epoch(X[i]) for i in range(X.shape[0])])
print("Feature matrix shape:", F.shape)


# In[11]:


from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report, confusion_matrix
import numpy as np

gkf = GroupKFold(n_splits=5)

def eval_model(clf, X, y, groups):
    y_true, y_pred = [], []
    for tr, te in gkf.split(X, y, groups):
        pipe = Pipeline([
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", clf)
        ])
        pipe.fit(X[tr], y[tr])
        yhat = pipe.predict(X[te])
        y_true.append(y[te]); y_pred.append(yhat)
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    kappa    = cohen_kappa_score(y_true, y_pred)
    report   = classification_report(y_true, y_pred, digits=3, output_dict=False)
    cm       = confusion_matrix(y_true, y_pred, labels=[0,1,2,3,4])
    return macro_f1, kappa, report, cm


# In[12]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=2000, multi_class="multinomial",
                            solver="lbfgs", class_weight="balanced", n_jobs=None)
lr_f1, lr_kappa, lr_report, lr_cm = eval_model(logreg, F, y, groups)
print("LogReg  Macro-F1:", round(lr_f1,3), "  Cohen's κ:", round(lr_kappa,3))
print(lr_report)
lr_cm


# In[13]:


#Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=1,
                            class_weight="balanced_subsample", random_state=42, n_jobs=-1)
rf_f1, rf_kappa, rf_report, rf_cm = eval_model(rf, F, y, groups)
print("RF      Macro-F1:", round(rf_f1,3), "  Cohen's κ:", round(rf_kappa,3))
print(rf_report)
rf_cm


# In[18]:


from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=600, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", objective="multi:softmax",
    num_class=5, n_jobs=-1, reg_lambda=1.0
)
xgb_f1, xgb_kappa, xgb_report, xgb_cm = eval_model(xgb, F, y, groups)
print("XGBoost Macro-F1:", round(xgb_f1,3), "Cohen's κ:", round(xgb_kappa,3))
print(xgb_report)
xgb_cm


# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns  # if you don't have seaborn, replace with plain matplotlib

labels = ["W","N1","N2","N3","REM"]

def plot_cm(cm, title):
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout(); plt.show()

plot_cm(lr_cm, "LogReg Confusion Matrix")
plot_cm(rf_cm, "Random Forest Confusion Matrix")
if "xgb_cm" in locals() and xgb_cm is not None:
    plot_cm(xgb_cm, "XGBoost Confusion Matrix")


# In[20]:


rows = [
    ("Logistic Regression", lr_f1, lr_kappa),
    ("Random Forest", rf_f1, rf_kappa),
]
if "xgb_f1" in locals() and xgb_f1 is not None:
    rows.append(("XGBoost", xgb_f1, xgb_kappa))

res = pd.DataFrame(rows, columns=["Model","Macro-F1","CohenKappa"])
res.sort_values("Macro-F1", ascending=False, inplace=True)
res.to_csv(ROOT/"baseline_results.csv", index=False)
res


# In[21]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.barplot(data=res, x="Model", y="Macro-F1", palette="viridis")
plt.title("Baseline Model Comparison (Macro-F1)")
plt.ylim(0,1)
plt.show()

plt.figure(figsize=(6,4))
sns.barplot(data=res, x="Model", y="CohenKappa", palette="magma")
plt.title("Baseline Model Comparison (Cohen's κ)")
plt.ylim(0,1)
plt.show()


# In[27]:


import shap
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

# ---------------------------------------------------
# 1. Refit XGBoost on ALL data (same hyperparameters)
# ---------------------------------------------------
xgb_global = XGBClassifier(
    n_estimators=600, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", objective="multi:softmax",
    num_class=5, n_jobs=-1, reg_lambda=1.0
)
xgb_global.fit(X, y)   # X: DataFrame, y: labels

feature_names = list(X.columns)
class_names = ["W", "N1", "N2", "N3", "REM"]

shap.initjs()

# ---------------------------------------------------
# 2. Build SHAP Explainer using a CALLABLE
#    → use predict_proba and permutation algorithm
# ---------------------------------------------------
n_sample = min(2000, len(X))
X_sample = shap.sample(X, n_sample, random_state=0)

# NOTE: we pass xgb_global.predict_proba, not the model itself
explainer_xgb = shap.Explainer(
    xgb_global.predict_proba,  # callable
    X_sample,                  # background / masker
    algorithm="permutation"    # model-agnostic but slower
)

# 3. Compute SHAP values
shap_values_exp = explainer_xgb(X_sample)
vals = shap_values_exp.values
print("Raw SHAP values shape:", vals.shape)
# Possible shapes:
#  - (n, f)           -> 2D, binary or aggregated
#  - (n, f, c)        -> 3D, (samples, features, classes)
#  - (n, c, f)        -> 3D, (samples, classes, features)

# ---------------------------------------------------
# 3a. Normalize shapes into a common form
#     We want vals_std: (n_samples, n_features, n_classes)
# ---------------------------------------------------
if vals.ndim == 2:
    # Only one class (unlikely here); treat as (n, f, 1)
    vals_std = vals[:, :, None]
elif vals.ndim == 3:
    if vals.shape[2] == len(class_names):
        # (n, f, c) already
        vals_std = vals
    elif vals.shape[1] == len(class_names):
        # (n, c, f) → transpose to (n, f, c)
        vals_std = np.transpose(vals, (0, 2, 1))
    else:
        # Fallback: assume last axis is features, add 1 class
        vals_std = vals[:, :, None]
else:
    raise RuntimeError("Unexpected SHAP values shape")

n_samples, n_features, n_classes = vals_std.shape
print("Standardized SHAP shape:", vals_std.shape)

# ---------------------------------------------------
# 4. Global feature importance (all classes combined)
#    mean(|SHAP|) over class axis
# ---------------------------------------------------
sv_global = np.mean(np.abs(vals_std), axis=2)  # (n_samples, n_features)

shap.summary_plot(
    sv_global,
    X_sample,
    feature_names=feature_names,
    show=False
)
plt.title("XGBoost – Global Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("shap_xgb_global_summary.png", dpi=220)
plt.close()

# ---------------------------------------------------
# 5. Per-class feature importance
# ---------------------------------------------------
for c, name in enumerate(class_names):
    sv_c = vals_std[:, :, c]  # (n_samples, n_features)

    shap.summary_plot(
        sv_c,
        X_sample,
        feature_names=feature_names,
        plot_type="bar",
        show=False
    )
    plt.title(f"XGBoost – Class {name} Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(f"shap_xgb_class_{name}.png", dpi=220)
    plt.close()

# ---------------------------------------------------
# 6. Single–sample explanation (waterfall for one epoch)
# ---------------------------------------------------
i = 0                 # you can change this to any sample index
x_row = X_sample.iloc[[i]]

row_exp = explainer_xgb(x_row)
row_vals = row_exp.values  # shape: (1, f, c) or similar

# Standardize single-sample shape to (features, classes)
if row_vals.ndim == 2:
    row_vals_std = row_vals[:, None]   # (f, 1)
elif row_vals.ndim == 3:
    # Try to get (f, c)
    if row_vals.shape[2] == n_classes:
        row_vals_std = row_vals[0]          # (f, c)
    elif row_vals.shape[1] == n_classes:
        row_vals_std = row_vals[0].T        # (f, c)
    else:
        row_vals_std = row_vals[0][:, None] # (f, 1)
else:
    raise RuntimeError("Unexpected row SHAP shape")

# Base values per class
base_vals = np.atleast_1d(row_exp.base_values)
if base_vals.ndim == 1 and base_vals.size == n_classes:
    base_std = base_vals
else:
    # If only one base value is returned, broadcast
    base_std = np.repeat(base_vals.reshape(-1), n_classes)[:n_classes]

# Choose the model's predicted class for this sample
pred_class = int(xgb_global.predict(x_row)[0])

shap.plots._waterfall.waterfall_legacy(
    base_std[pred_class],
    row_vals_std[:, pred_class],
    feature_names=feature_names,
    show=False
)
plt.title(
    f"XGBoost – SHAP Waterfall for Sample {i} "
    f"(class {class_names[pred_class]})"
)
plt.tight_layout()
plt.savefig("shap_xgb_sample_waterfall.png", dpi=220)
plt.close()


# In[29]:


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




