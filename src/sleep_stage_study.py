"""
Sleep Stage Study - Consolidated Analysis Script
Sleep staging from wearable and PSG signals using lightweight temporal models.

Usage:
    - Run this file as a script or copy sections into a Jupyter notebook.
    - All paths are relative to the project root (one level above ``src``).
"""

# =========================================================
# 0. Imports & Global Config
# =========================================================

import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from torch.utils.data import Dataset, DataLoader

import shap
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")          # keep notebook clean
plt.style.use("seaborn-v0_8")
sns.set_context("talk")

# ---------- Paths: project root / raw / derived ----------

try:
    # When running as a script (src/sleep_stage_study.py).
    SOURCE_DIR = Path(__file__).resolve().parent
except NameError:
    # When running inside Jupyter from the src/ folder.
    SOURCE_DIR = Path(".").resolve()

PROJECT_ROOT = SOURCE_DIR.parent

# Raw EDF root (the zip you extracted)
RAW_EDF_ROOT = PROJECT_ROOT / "sleep-edf-database-expanded-1.0.0"

# >>> THIS is your actual PSG/Hyp folder <<<
SC_ROOT = RAW_EDF_ROOT / "sleep-cassette"

# Output / derived folders
DERIVED_DIR = PROJECT_ROOT / "derived_sc"
FEATURE_DIR = PROJECT_ROOT / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
SHAP_DIR = PROJECT_ROOT / "shap_figs"

for d in [DERIVED_DIR, FEATURE_DIR, RESULTS_DIR, SHAP_DIR]:
    d.mkdir(exist_ok=True)

NUM_CLASSES = 5
CLASS_NAMES = ["W", "N1", "N2", "N3", "REM"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("SC_ROOT:", SC_ROOT)

# =========================================================
# 1. Phase 1 – Load EDF pairs & Save Per-Recording Epochs
# =========================================================
# You only need to run this ONCE to create .npz files in DERIVED_DIR.
# After that, you can skip it and just load the .npz files in later
# phases (2–6).
# =========================================================

def _expected_hyp(psg_path: Path) -> Path:
    """
    Sleep-EDF naming rule:
      SC4001E0-PSG.edf  -> SC4001EC-Hypnogram.edf
      SC4001E1-PSG.edf  -> SC4001EH-Hypnogram.edf

    We convert the last character (0/1) of the PSG night code to EC/EH.
    """
    # e.g. 'SC4001E0-PSG' -> 'SC4001E0'
    base = psg_path.stem.split("-")[0]
    prefix, last = base[:-1], base[-1]  # 'SC4001E', '0'
    hyp_tag = "EC" if last == "0" else "EH"
    hyp_name = f"{prefix}{hyp_tag}-Hypnogram.edf"
    return psg_path.with_name(hyp_name)   # still inside SC_ROOT


def build_pairs_list():
    """
    Build list of (psg_path, hypnogram_path) pairs from SC_ROOT.
    Handles the EC/EH naming and falls back to fuzzy match if needed.
    """
    psg_files = sorted(SC_ROOT.glob("*-PSG.edf"))
    pairs = []
    missing = []

    for psg in psg_files:
        # First try the exact EC/EH naming rule
        hyp = _expected_hyp(psg)
        if hyp.exists():
            pairs.append((psg, hyp))
            continue

        # Fallback: try any Hypnogram that shares the first 7 chars
        # e.g. SC4001E?-Hypnogram.edf
        night_key = psg.stem.split("-")[0][:7]  # 'SC4001E'
        cands = list(SC_ROOT.glob(f"{night_key}?-Hypnogram.edf"))
        if cands:
            pairs.append((psg, sorted(cands)[0]))
        else:
            missing.append(psg.name)

    print(f"PSG files found : {len(psg_files)}")
    print(f"PSG/Hyp pairs   : {len(pairs)}")
    if missing:
        print(f"Missing hypnograms for {len(missing)} PSG files (showing up to 5):")
        print("  ", missing[:5])
    return pairs


def epoch_one(psg_path: Path, hyp_path: Path, epoch_len_sec: float = 30.0):
    """
    Epoch a single PSG/Hyp pair into 30-second epochs.

    Returns
    -------
    X : np.ndarray, shape (n_epochs, n_channels, n_samples)
    y : np.ndarray, shape (n_epochs,), labels in {0..4}
    """
    import mne

    # Load PSG and annotations
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    ann = mne.read_annotations(hyp_path)
    raw.set_annotations(ann, emit_warning=False)

    # Map annotation strings to 0..4 labels
    STAGE_MAP = {
        "Sleep stage W": 0,
        "Sleep stage N1": 1,
        "Sleep stage N2": 2,
        "Sleep stage N3": 3,
        "Sleep stage R": 4,
    }

    # Only keep the events we care about (keys in STAGE_MAP)
    events, _ = mne.events_from_annotations(
        raw,
        event_id=STAGE_MAP,
        verbose=False,
    )

    if len(events) == 0:
        print(f"  [WARN] No sleep-stage events in {psg_path.name}, skipping.")
        return None, None

    sfreq = raw.info["sfreq"]
    tmax = epoch_len_sec - 1.0 / sfreq  # inclusive end

    epochs = mne.Epochs(
        raw,
        events,
        event_id=None,            # use all events already filtered above
        tmin=0.0,
        tmax=tmax,
        baseline=None,
        preload=True,
        verbose=False,
        on_missing="ignore",      # in case some stages never appear
    )

    X = epochs.get_data()          # (n_epochs, n_channels, n_samples)
    y = epochs.events[:, -1]       # event_id we passed in STAGE_MAP

    return X.astype(np.float32), y.astype(np.int64)


def phase1_build_npz(force: bool = False):
    """
    Phase 1 driver: iterate over PSG/Hyp pairs, save each subject
    into DERIVED_DIR / '{rec_id}.npz' with
        X : epochs array
        y : labels
        subject : subject_id per epoch (string)
    Set force=True to overwrite existing .npz files.
    """
    pairs = build_pairs_list()
    total_epochs = 0

    for psg, hyp in pairs:
        # e.g. 'SC4001E0-PSG' -> 'SC4001E0'
        rec_id = psg.stem.split("-")[0]
        out_path = DERIVED_DIR / f"{rec_id}.npz"

        if out_path.exists() and not force:
            # Already processed earlier
            continue

        print(f"\nProcessing {rec_id} ...")
        X, y = epoch_one(psg, hyp)

        if X is None or len(X) == 0:
            print(f"  Skipped {rec_id}: no usable epochs")
            continue

        # Subject ID array: one subject id per epoch
        subj_arr = np.full(len(y), rec_id, dtype="<U16")

        np.savez_compressed(
            out_path,
            X=X,
            y=y,
            subject=subj_arr,
        )
        print(f"  Saved {X.shape[0]} epochs -> {out_path.name}")
        total_epochs += X.shape[0]

    print("\nTotal epochs across all recordings:", total_epochs)


# =========================================================
# 2. Phase 2 – Stack All Subjects into X / y / groups & Features F
# =========================================================

def phase2_stack_data():
    """
    Loads all .npz files from DERIVED_DIR and concatenates them into:
        X_raw:   (N, C, T)
        y:       (N,)
        groups:  (N,) subject IDs for GroupKFold
    Also computes band-power features F and saves all arrays.
    """
    from scipy.signal import welch

    npz_files = sorted(DERIVED_DIR.glob("*.npz"))
    X_list, y_list, g_list = [], [], []

    for path in npz_files:
        data = np.load(path, allow_pickle=True)
        X_list.append(data["X"])
        y_list.append(data["y"])
        # 'subject' is per-epoch ID; if not saved, derive from filename
        if "subject" in data:
            g_list.append(data["subject"])
        else:
            rec_id = path.stem
            g_list.append(np.full_like(data["y"], rec_id))

    X_raw = np.concatenate(X_list, axis=0).astype(np.float32)
    y = np.concatenate(y_list, axis=0).astype(np.int64)
    groups = np.concatenate(g_list, axis=0)

    print("X_raw shape:", X_raw.shape)
    print("y counts:", np.bincount(y, minlength=NUM_CLASSES))

    # ---- Save CNN/TCN input (z-score by channel) ----
    mu = X_raw.mean(axis=(0, 2), keepdims=True)
    sd = X_raw.std(axis=(0, 2), keepdims=True) + 1e-8
    X_std = (X_raw - mu) / sd

    np.save(PROJECT_ROOT / "X_sc_std.npy", X_std)
    np.save(PROJECT_ROOT / "y_labels.npy", y)
    np.save(PROJECT_ROOT / "groups_subject.npy", groups)

    # ---- Bandpower features F (per epoch x (channels * bands)) ----
    fs = 100.0  # set to your PSG sampling rate
    bands = [(0.5, 4), (4, 8), (8, 12), (12, 30)]

    def bandpowers(sig):
        # sig: (T,), 1D
        f, Pxx = welch(sig, fs=fs, nperseg=int(fs * 2))
        feats = []
        for lo, hi in bands:
            mask = (f >= lo) & (f < hi)
            feats.append(Pxx[mask].sum())
        return np.array(feats, dtype=np.float32)

    N, C, T = X_raw.shape
    F_list = []

    print("Computing band-power features...")
    for i in tqdm(range(N), desc="Bandpowers"):
        feats_epoch = [bandpowers(X_raw[i, ch]) for ch in range(C)]
        F_list.append(np.concatenate(feats_epoch))

    F = np.vstack(F_list).astype(np.float32)
    print("Feature matrix F shape:", F.shape)

    FEATURE_DIR.mkdir(exist_ok=True)
    np.save(FEATURE_DIR / "F_sc_bandpower.npy", F)
    np.save(FEATURE_DIR / "y_labels.npy", y)
    np.save(FEATURE_DIR / "groups_subject.npy", groups)


# =========================================================
# 3. Phase 3 – Classical Baselines (LogReg / RF / XGB)
# =========================================================

def load_features():
    # Band-power feature matrix and labels from Phase 2
    F = np.load(FEATURE_DIR / "F_sc_bandpower.npy").astype(np.float32)
    y = np.load(FEATURE_DIR / "y_labels.npy")
    groups = np.load(FEATURE_DIR / "groups_subject.npy")

    print("F shape:", F.shape)
    print("y counts (raw):", np.bincount(y, minlength=NUM_CLASSES))
    return F, y, groups


def eval_sklearn_model(clf, X, y, groups, name="Model"):
    gkf = GroupKFold(n_splits=5)
    macros, kappas = [], []

    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        Xtr, Xte = X[tr], X[te]
        ytr, yte = y[tr], y[te]

        clf.fit(Xtr, ytr)
        yp = clf.predict(Xte)

        macro = f1_score(yte, yp, average="macro")
        kappa = cohen_kappa_score(yte, yp)
        macros.append(macro)
        kappas.append(kappa)

        print(f"[{name}] Fold {fold}: MacroF1={macro:.3f}, κ={kappa:.3f}")

    macros = np.array(macros)
    kappas = np.array(kappas)
    print(
        f"[{name}] CV Macro-F1 {macros.mean():.3f}±{macros.std():.3f}, "
        f"κ {kappas.mean():.3f}±{kappas.std():.3f}"
    )
    return macros, kappas


def phase3_run_baselines():
    print("\n=== Phase 3: Classical baselines (LogReg, RF, XGBoost) ===")
    F, y_raw, groups = load_features()

    # ---------------------------------------------------------
    # 1) Re-encode labels so they are 0..(C-1)
    #    e.g. {0, 4}  → {0, 1}
    # ---------------------------------------------------------
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    print("Unique raw labels:", le.classes_)
    print("Encoded labels used by models:", np.unique(y))
    print("Encoded label counts:", np.bincount(y, minlength=n_classes))
    print("Raw → encoded mapping:")
    for enc, raw in enumerate(le.classes_):
        print(f"  raw {raw}  →  enc {enc}")

    # small helper so we don’t repeat code
    def run_model(clf, name: str):
        print(f"\n--- {name} ---")
        eval_sklearn_model(clf, F, y, groups, name=name)

    # ---------------------------------------------------------
    # 2) Logistic Regression (one-vs-rest / multinomial)
    # ---------------------------------------------------------
    logreg = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        class_weight="balanced",
        multi_class="ovr" if n_classes == 2 else "multinomial",
        solver="lbfgs",
    )
    run_model(logreg, "LogReg")

    # ---------------------------------------------------------
    # 3) Random Forest
    # ---------------------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    run_model(rf, "RandomForest")

    # ---------------------------------------------------------
    # 4) XGBoost
    #    - binary:logistic for 2 classes
    #    - multi:softprob for >2 classes
    # ---------------------------------------------------------
    if n_classes == 2:
        xgb_objective = "binary:logistic"
        xgb_num_class = None
    else:
        xgb_objective = "multi:softprob"
        xgb_num_class = n_classes

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective=xgb_objective,
        num_class=xgb_num_class,
        eval_metric="mlogloss",
        n_jobs=-1,
        reg_lambda=1.0,
        use_label_encoder=False,
        random_state=42,
    )
    run_model(xgb, "XGBoost")

    print("\nPhase 3 completed.\n")

# =========================================================
# 4. Phase 4 – TinyCNN & TinyTCN (Deep models)
# =========================================================

class TinyCNN(nn.Module):
    def __init__(self, in_ch=4, ncls=NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, 16, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(32, ncls)

    def forward(self, x):
        # x: (B, C, T)
        x = Fnn.relu(self.bn1(self.conv1(x)))
        x = Fnn.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class TCNBlock(nn.Module):
    def __init__(self, ch, k=5, d=1, p=0.2):
        super().__init__()
        pad = (k - 1) * d // 2
        self.conv1 = nn.Conv1d(ch, ch, kernel_size=k, padding=pad, dilation=d)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, kernel_size=k, padding=pad, dilation=d)
        self.bn2 = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        h = Fnn.relu(self.bn1(self.conv1(x)))
        h = self.drop(h)
        h = Fnn.relu(self.bn2(self.conv2(h)))
        h = self.drop(h)
        return x + h


class TinyTCN(nn.Module):
    def __init__(self, in_ch=4, ncls=NUM_CLASSES, base=32, layers=4, p=0.2):
        super().__init__()
        self.stem = nn.Conv1d(in_ch, base, kernel_size=3, padding=1)
        blocks = []
        for i in range(layers):
            blocks.append(TCNBlock(base, k=5, d=2**i, p=p))
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(base, ncls)

    def forward(self, x):
        x = Fnn.relu(self.stem(x))
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class EpochDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def make_loaders(Xtr, ytr, Xte, yte, bs=128, workers=0):
    return (
        DataLoader(EpochDataset(Xtr, ytr), batch_size=bs, shuffle=True,
                   num_workers=workers, pin_memory=False),
        DataLoader(EpochDataset(Xte, yte), batch_size=bs, shuffle=False,
                   num_workers=workers, pin_memory=False),
    )


@torch.no_grad()
def eval_model_torch(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        y_true.append(yb.numpy())
        y_pred.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return (
        f1_score(y_true, y_pred, average="macro"),
        cohen_kappa_score(y_true, y_pred),
        (y_true, y_pred),
    )


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ---------- helper: load raw X / y / groups for deep models ----------
def load_raw_for_deep(keep_only_w_rem=True):
    """
    Re-stack per-recording .npz files from DERIVED_DIR.

    Returns X_raw: (N, C, T), y_raw, groups (subject IDs or similar).
    """
    Xs, ys, subj_ids = [], [], []

    for f in sorted(DERIVED_DIR.glob("*.npz")):
        d = np.load(f)
        Xs.append(d["X"].astype(np.float32))
        ys.append(d["y"].astype(np.int64))
        # subjects were saved as strings in Phase 1
        subj_ids.append(d["subject"].astype(str))

    X_raw = np.concatenate(Xs, axis=0)
    y_raw = np.concatenate(ys, axis=0)
    subj_ids = np.concatenate(subj_ids, axis=0)
    # group by first 6 chars of recording id (SC4001 etc.)
    groups = np.array([s[:6] for s in subj_ids])

    print("X_raw shape:", X_raw.shape)
    print("y counts (raw):", np.bincount(y_raw))

    if keep_only_w_rem:
        mask = np.isin(y_raw, [0, 4])    # W & REM
        X_raw = X_raw[mask]
        y_raw = y_raw[mask]
        groups = groups[mask]
        print("After W/REM filter, shape:", X_raw.shape)
        print("y counts (W/REM):", np.bincount(y_raw))

    # global z-score per channel
    mu = X_raw.mean(axis=(0, 2), keepdims=True)
    sd = X_raw.std(axis=(0, 2), keepdims=True) + 1e-8
    X_std = (X_raw - mu) / sd
    X_std = X_std.astype(np.float32)

    return X_std, y_raw, groups


def run_cv_deep(ModelClass, model_name, X, y, groups,
                epochs=12, bs=128, lr=1e-3):
    """
    Memory-safe 5-fold CV for TinyCNN/TinyTCN on X.
    """
    gkf = GroupKFold(n_splits=5)
    scores = []

    for fold, (tr, te) in enumerate(
            tqdm(gkf.split(X, y, groups),
                 total=gkf.get_n_splits(),
                 desc=f"[{model_name}] CV Folds"), 1):

        Xtr, ytr = X[tr], y[tr]
        Xte, yte = X[te], y[te]

        train_loader, val_loader = make_loaders(Xtr, ytr, Xte, yte, bs=bs)

        model = ModelClass().to(device)
        print(f"\n[{model_name}] Fold {fold}  params={count_params(model):,}")

        optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)

        best_f1 = -1.0
        best_state = None

        for ep in range(1, epochs + 1):
            model.train()
            loss_sum = 0.0
            n = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = Fnn.cross_entropy(logits, yb)
                optim.zero_grad()
                loss.backward()
                optim.step()
                loss_sum += loss.item() * len(yb)
                n += len(yb)
            sched.step()

            f1, kappa, _ = eval_model_torch(model, val_loader)
            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

            print(f"  Epoch {ep:02d}: loss={loss_sum/n:.4f}, "
                  f"MacroF1={f1:.3f}, κ={kappa:.3f}", end="\r")

        model.load_state_dict(best_state)
        macro, kappa, _ = eval_model_torch(model, val_loader)

        xb = torch.from_numpy(Xte[:min(len(Xte), bs)]).float().to(device)
        t0 = time.time()
        _ = model(xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_time = (time.time() - t0) / len(xb) * 1000

        print(f"\n[{model_name}] Fold {fold}  "
              f"macroF1={macro:.3f}  κ={kappa:.3f}  "
              f"infer={infer_time:.2f} ms/epoch")

        scores.append((macro, kappa, infer_time, count_params(model)))

        del Xtr, Xte, train_loader, val_loader, model

    scores = np.array(scores)
    print(
        f"\n[{model_name}]  CV Macro-F1: {scores[:,0].mean():.3f}±{scores[:,0].std():.3f}  "
        f"κ: {scores[:,1].mean():.3f}±{scores[:,1].std():.3f}  "
        f"Infer: {scores[:,2].mean():.1f} ms   "
        f"Params: ~{int(scores[:,3].mean()):,}"
    )
    return scores


def phase4_run_deep_models():
    # Load standardized raw epochs + labels + groups
    X_std, y, groups = load_raw_for_deep(keep_only_w_rem=True)

    print("Deep data X shape:", X_std.shape)   # (N, C, T)
    n_ch = X_std.shape[1]                     # <- 7 channels

    # use lambdas so run_cv_deep can build a fresh model each fold
    cnn_scores = run_cv_deep(
        lambda: TinyCNN(in_ch=n_ch, ncls=NUM_CLASSES),
        "TinyCNN",
        X_std, y, groups,
        epochs=12, bs=128, lr=1e-3
    )

    tcn_scores = run_cv_deep(
        lambda: TinyTCN(in_ch=n_ch, ncls=NUM_CLASSES),
        "TinyTCN",
        X_std, y, groups,
        epochs=12, bs=128, lr=1e-3
    )

    np.save(RESULTS_DIR / "cnn_scores.npy", cnn_scores)
    np.save(RESULTS_DIR / "tcn_scores.npy", tcn_scores)


# =========================================================
# 5. Phase 5 – SHAP for Random Forest & XGBoost (binary W vs REM)
# =========================================================

def phase5_shap_classical():
    # -------- 1) Load features & raw labels --------
    F, y_raw, groups = load_features()  # y_raw is {0, 4} after W/REM filter
    F = F.astype(np.float32, copy=False)
    feature_names = [f"f{i}" for i in range(F.shape[1])]
    X_df = pd.DataFrame(F, columns=feature_names)

    print("F shape:", F.shape)
    print("y counts (raw):", np.bincount(y_raw))
    uniq_raw = np.unique(y_raw)
    print("Unique raw labels:", uniq_raw)

    # -------- 2) Encode labels to 0..C-1 --------
    # For your data, this will be 2 classes: 0 -> 0 (W), 4 -> 1 (REM).
    enc_map = {raw: i for i, raw in enumerate(uniq_raw)}   # e.g. {0:0, 4:1}
    y = np.array([enc_map[v] for v in y_raw], dtype=int)

    n_classes = len(uniq_raw)
    class_names_local = [CLASS_NAMES[raw] for raw in uniq_raw]  # e.g. ["W","REM"]

    print("Encoded labels used by models:", np.unique(y))
    print("Encoded label counts:", np.bincount(y, minlength=n_classes))
    print("Raw → encoded mapping:")
    for raw, enc in enc_map.items():
        print(f"  raw {raw} → enc {enc}")

    shap.initjs()

    # -------- 3) Fit global models (RF + XGB) --------
    # Random Forest is happy with arbitrary labels, but we use encoded y for consistency.
    rf_global = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf_global.fit(X_df, y)

    # XGBoost: use binary objective if 2 classes, otherwise multi-class
    if n_classes == 2:
        xgb_global = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="binary:logistic",
            n_jobs=-1,
            reg_lambda=1.0,
            random_state=42,
        )
    else:
        xgb_global = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="multi:softmax",
            num_class=n_classes,
            n_jobs=-1,
            reg_lambda=1.0,
            random_state=42,
        )
    xgb_global.fit(X_df, y)

    # -------- 4) RF SHAP (TreeExplainer) --------
    n_sample = min(2000, len(X_df))
    X_sample = shap.sample(X_df, n_sample, random_state=0)

    explainer_rf = shap.TreeExplainer(rf_global)
    shap_values_rf = explainer_rf.shap_values(X_sample)

    # Standardize RF SHAP to (N, F, C)
    if isinstance(shap_values_rf, list):
        sv_rf = np.stack(shap_values_rf, axis=0)      # (C, N, F)
        sv_rf = np.transpose(sv_rf, (1, 2, 0))        # (N, F, C)
    else:
        sv_rf = np.array(shap_values_rf)
        if sv_rf.shape[0] == n_classes:               # (C, N, F)
            sv_rf = np.transpose(sv_rf, (1, 2, 0))    # (N, F, C)

    print("RF SHAP shape:", sv_rf.shape)

    # Global importance: mean |SHAP| over classes
    vals_all_rf = np.mean(np.abs(sv_rf), axis=2)
    shap.summary_plot(vals_all_rf, X_sample,
                      feature_names=feature_names, show=False)
    plt.title("Random Forest – Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_rf_global_summary.png", dpi=220)
    plt.close()

    # Per-class bar plots
    for c, name in enumerate(class_names_local):
        sv_c = sv_rf[:, :, c]
        shap.summary_plot(sv_c, X_sample,
                          feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.title(f"Random Forest – Class {name} Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"shap_rf_class_{name}.png", dpi=220)
        plt.close()

    # -------- 5) XGB SHAP (permutation Explainer) --------
    explainer_xgb = shap.Explainer(
        xgb_global.predict_proba,
        X_sample,
        algorithm="permutation",
    )
    shap_values_xgb = explainer_xgb(X_sample)
    vals_xgb = shap_values_xgb.values
    print("XGB raw SHAP values shape:", vals_xgb.shape)

    # Standardize XGB SHAP to (N, F, C)
    if vals_xgb.ndim == 2:
        sv_xgb = vals_xgb[:, :, None]
        n_classes_eff = 1
    elif vals_xgb.ndim == 3:
        if vals_xgb.shape[2] == n_classes:           # (N, F, C)
            sv_xgb = vals_xgb
        elif vals_xgb.shape[1] == n_classes:         # (N, C, F)
            sv_xgb = np.transpose(vals_xgb, (0, 2, 1))
        else:
            sv_xgb = vals_xgb[:, :, None]
        n_classes_eff = sv_xgb.shape[2]
    else:
        raise RuntimeError("Unexpected XGB SHAP shape")

    print("Standardized XGB SHAP shape:", sv_xgb.shape)

    # Global XGB importance
    vals_all_xgb = np.mean(np.abs(sv_xgb), axis=2)
    shap.summary_plot(vals_all_xgb, X_sample,
                      feature_names=feature_names, show=False)
    plt.title("XGBoost – Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_xgb_global_summary.png", dpi=220)
    plt.close()

    # Per-class bar plots (only if we actually have multiple classes)
    for c in range(n_classes_eff):
        name = class_names_local[c] if c < len(class_names_local) else f"class_{c}"
        sv_c = sv_xgb[:, :, c]
        shap.summary_plot(sv_c, X_sample,
                          feature_names=feature_names,
                          plot_type="bar", show=False)
        plt.title(f"XGBoost – Class {name} Feature Importance (SHAP)")
        plt.tight_layout()
        plt.savefig(SHAP_DIR / f"shap_xgb_class_{name}.png", dpi=220)
        plt.close()

    # -------- 6) Single-sample waterfall for XGBoost --------
    i = 0
    x_row = X_sample.iloc[[i]]
    row_exp = explainer_xgb(x_row)
    row_vals = row_exp.values  # shapes: (1, F) or (1, F, C)

    if row_vals.ndim == 2:
        # ----- Single-output case (e.g., binary model) -----
        shap_vec = row_vals[0]                     # (F,)
        base_flat = np.asarray(row_exp.base_values).reshape(-1)
        base_val = float(base_flat[0])            # scalar
        title_suffix = "binary output"
    elif row_vals.ndim == 3:
        # ----- Multi-output case (per-class SHAP) -----
        n_classes_eff = row_vals.shape[2]

        # Flatten any shape (C,), (1,C), (C,1), etc. to 1D
        base_flat = np.asarray(row_exp.base_values).reshape(-1)

        if base_flat.size >= n_classes_eff:
            base_all = base_flat[:n_classes_eff]
        else:
            # broadcast if SHAP only gives one base value
            base_all = np.repeat(base_flat, n_classes_eff)[:n_classes_eff]

        # choose predicted class for this sample
        pred_class = int(xgb_global.predict(x_row)[0])
        if pred_class < len(CLASS_NAMES):
            pred_name = CLASS_NAMES[pred_class]
        else:
            pred_name = f"class_{pred_class}"

        shap_vec = row_vals[0, :, pred_class]     # (F,)
        base_val = float(base_all[pred_class])    # scalar
        title_suffix = f"class {pred_name}"
    else:
        raise RuntimeError(f"Unexpected row SHAP shape: {row_vals.shape}")

    shap.plots._waterfall.waterfall_legacy(
        base_val,
        shap_vec,
        feature_names=feature_names,
        show=False,
    )
    plt.title(f"XGBoost – SHAP Waterfall Sample {i} ({title_suffix})")
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_xgb_sample_waterfall.png", dpi=220)
    plt.close()




# =========================================================
# 6. Phase 6 – Confusion Matrix for Best Classical Model
# =========================================================
# Uses XGBoost on the same W vs REM setting as Phase 3/5.
# Produces a 2x2 confusion matrix image in RESULTS_DIR.
# =========================================================

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import numpy as np


def phase6_confusion_matrix():
    print("=== Phase 6: Confusion Matrix ===")

    # 1. Load features from Phase 2
    F = np.load(FEATURE_DIR / "F_sc_bandpower.npy").astype(np.float32)
    y_raw = np.load(FEATURE_DIR / "y_labels.npy")
    groups = np.load(FEATURE_DIR / "groups_subject.npy")

    print("F shape:", F.shape)
    print("y counts (raw):", np.bincount(y_raw))

    # 2. Encode labels the same way as Phase 3:
    #    raw 0 -> enc 0 (W)
    #    raw 4 -> enc 1 (REM)
    unique_raw = np.unique(y_raw)
    print("Unique raw labels:", unique_raw)

    if np.array_equal(unique_raw, np.array([0, 4])):
        raw_to_enc = {0: 0, 4: 1}
        y = np.vectorize(raw_to_enc.get)(y_raw)
        class_names_bin = ["W", "REM"]
        n_classes = 2
        print("Encoded labels used for XGBoost:", np.bincount(y, minlength=n_classes))
        print("Raw -> encoded mapping: 0→0 (W), 4→1 (REM)")
    else:
        # Fallback: multi-class (not expected in your current setup)
        y = y_raw
        class_names_bin = CLASS_NAMES
        n_classes = NUM_CLASSES

    # 3. Take fold 1 of subject-wise CV for confusion matrix
    gkf = GroupKFold(n_splits=5)
    (train_idx, test_idx) = list(gkf.split(F, y, groups))[0]

    Xtr, Xte = F[train_idx], F[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    # 4. Train XGBoost (binary W vs REM)
    if n_classes == 2:
        model = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="binary:logistic",
            n_jobs=-1,
            reg_lambda=1.0,
        )
    else:
        # generic multi-class version (probably not used here)
        model = XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            objective="multi:softmax",
            num_class=n_classes,
            n_jobs=-1,
            reg_lambda=1.0,
        )

    print("Training XGBoost for confusion matrix...")
    model.fit(Xtr, ytr)

    # 5. Predictions
    yp = model.predict(Xte)

    # 6. Confusion matrix
    cm = confusion_matrix(yte, yp, labels=list(range(n_classes)))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names_bin
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    plt.title("XGBoost Confusion Matrix (Fold 1, W vs REM)", fontsize=14, pad=20)
    plt.tight_layout(pad=2.0)

    # 7. Save figure
    out_path = RESULTS_DIR / "confmat_xgb_fold1.png"
    fig.savefig(out_path, dpi=220)
    plt.close()

    print(f"Confusion matrix saved to: {out_path}")

    # Useful for Phase 7 error analysis
    return cm, (yte, yp)


# =========================================================
# 7. Phase 7 – Export results table + comparison plot
# =========================================================

def phase7_export_results():
    """
    Re-run classical W/REM baselines, combine with deep results,
    and save:
      - results/model_summary.csv
      - results/model_summary_latex.txt
      - results/model_macroF1_bar.png
    """

    # ----- 1) Load bandpower features and W/REM labels -----
    F, y_raw, groups = load_features()  # reuses Phase 3 helper

    # y_raw currently has raw sleep labels (0..4), but only 0 and 4 appear
    # Encode them as 0 -> 0 (W), 4 -> 1 (REM) for binary classifiers
    unique_raw = np.unique(y_raw)
    print("Unique raw labels:", unique_raw)

    # Safety check: we expect only W & REM
    if not np.all(np.isin(unique_raw, [0, 4])):
        raise ValueError(f"Phase7 expects only labels 0 and 4, but got {unique_raw}")

    y_bin = np.where(y_raw == 0, 0, 1)
    print("Encoded labels used by Phase 7 models:", np.unique(y_bin))

    # ----- 2) Helper to summarize classical sklearn models -----
    results = []

    def summarize_sklearn_model(name, clf):
        print(f"\n--- {name} (Phase 7) ---")
        macros, kappas = eval_sklearn_model(clf, F, y_bin, groups, name=name)
        results.append({
            "Model":          name,
            "Family":         "Classical",
            "MacroF1_mean":   float(macros.mean()),
            "MacroF1_std":    float(macros.std()),
            "Kappa_mean":     float(kappas.mean()),
            "Kappa_std":      float(kappas.std()),
            "Infer_ms":       np.nan,   # not measured here
            "Params":         np.nan,   # tree / linear models
        })

    # ----- 3) Classical models (same hyperparams as Phase 3) -----

    # Logistic Regression
    logreg = LogisticRegression(
        max_iter=2000,
        multi_class="ovr",
        n_jobs=-1,
        class_weight="balanced",
    )
    summarize_sklearn_model("LogReg", logreg)

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    summarize_sklearn_model("RandomForest", rf)

    # XGBoost (binary – num_class=2)
    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        objective="binary:logistic",
        n_jobs=-1,
        reg_lambda=1.0,
    )
    summarize_sklearn_model("XGBoost", xgb)

    # ----- 4) Deep model results from Phase 4 (TinyCNN / TinyTCN) -----

    def summarize_deep(name, scores):
        # scores: (5, 4) = (MacroF1, Kappa, infer_ms, params)
        scores = np.asarray(scores)
        results.append({
            "Model":          name,
            "Family":         "Deep",
            "MacroF1_mean":   float(scores[:, 0].mean()),
            "MacroF1_std":    float(scores[:, 0].std()),
            "Kappa_mean":     float(scores[:, 1].mean()),
            "Kappa_std":      float(scores[:, 1].std()),
            "Infer_ms":       float(scores[:, 2].mean()),
            "Params":         int(scores[:, 3].mean()),
        })

    # Load saved arrays from Phase 4
    cnn_path = RESULTS_DIR / "cnn_scores.npy"
    tcn_path = RESULTS_DIR / "tcn_scores.npy"

    if cnn_path.exists():
        cnn_scores = np.load(cnn_path)
        summarize_deep("TinyCNN", cnn_scores)
    else:
        print("WARNING: cnn_scores.npy not found – TinyCNN not included in table.")

    if tcn_path.exists():
        tcn_scores = np.load(tcn_path)
        summarize_deep("TinyTCN", tcn_scores)
    else:
        print("WARNING: tcn_scores.npy not found – TinyTCN not included in table.")

    # ----- 5) Build DataFrame and save as CSV + LaTeX -----

    df = pd.DataFrame(results)
    # Sort by Macro-F1 (best at top)
    df = df.sort_values("MacroF1_mean", ascending=False).reset_index(drop=True)

    csv_path = RESULTS_DIR / "model_summary.csv"
    latex_path = RESULTS_DIR / "model_summary_latex.txt"

    df.to_csv(csv_path, index=False)

    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format="%.3f"))

    print("\nSaved model summary table to:")
    print("  -", csv_path)
    print("  -", latex_path)

    # ----- 6) Bar plot: Macro-F1 comparison -----

    plt.figure(figsize=(8, 4))

    # Bar plot with error bars
    plt.bar(df["Model"], df["MacroF1_mean"],
            yerr=df["MacroF1_std"], capsize=4)

    plt.ylabel("Macro-F1 (5-fold mean ± std)", fontsize=11)
    plt.title("Model Comparison – W vs REM (Bandpower)", fontsize=14)

    # Fix overlapping labels: smaller font + rotation
    plt.xticks(rotation=20, ha='right', fontsize=9)

    # Add more margin below the plot
    plt.subplots_adjust(bottom=0.25)

    plt.ylim(0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    bar_path = RESULTS_DIR / "model_macroF1_bar.png"
    plt.savefig(bar_path, dpi=220)
    plt.close()

    print("Saved comparison plot to:")
    print("  -", bar_path)
    print("\nPhase 7 export complete.")



if __name__ == "__main__":
    # Uncomment the phases you want to (re)run.
    # WARNING: Phase 1 & 2 are expensive and produce large files.

    # phase1_build_npz(force=False)
    # phase2_stack_data()
    # phase3_run_baselines()
    # phase4_run_deep_models()
    # phase5_shap_classical()
    # phase6_confusion_matrix()
    phase7_export_results()
    pass
