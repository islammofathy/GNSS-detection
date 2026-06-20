"""
GNSS Spoofing Detection — Six Model Comparison
===============================================
ML  : SVM (SVM.pkl) | Random Forest (RandomForest.pkl)
DL  : Attention-BiLSTM | CNN-LSTM | Transformer | Transformer-Attention
"""

import re, io, os, gzip, shutil, warnings, subprocess, tempfile, zipfile
import requests
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, f1_score, precision_score, recall_score,
)

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════
# 0. PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="GNSS Spoofing Detector — 6 Models",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. CSS
# ═══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
body { font-family: 'Segoe UI', sans-serif; }
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg,#0f2027,#203a43,#2c5364);
}
section[data-testid="stSidebar"] * { color:#e0eafc !important; }
div[data-testid="metric-container"] {
    background:#f8fafc; border:1px solid #e2e8f0;
    border-radius:10px; padding:14px 18px;
    box-shadow:0 1px 4px rgba(0,0,0,.06);
}
.section-header {
    background:linear-gradient(90deg,#1a73e8,#0d47a1);
    color:white !important; padding:10px 18px;
    border-radius:8px; margin-bottom:12px;
    font-size:1.05rem; font-weight:600; letter-spacing:.3px;
}
.winner-banner {
    background:linear-gradient(135deg,#11998e,#38ef7d);
    color:white; padding:18px 24px; border-radius:12px;
    font-size:1.25rem; font-weight:700; text-align:center; margin-top:10px;
}
.alert-spoof { color:#e53e3e; font-weight:700; }
.alert-clean { color:#38a169; font-weight:700; }
button[data-baseweb="tab"] { font-size:.95rem; font-weight:600; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 2. CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════
WINDOW        = 30
ML_N_FEATURES = 256
DL_INPUT_DIM  = 64    # confirmed from checkpoint: lstm.weight_ih_l0 [256, 64]
DL_NUM_CLASSES = 7    # confirmed from checkpoint: fc.weight [7, ...]

MODEL_COLORS = {
    "SVM":                   "#1a73e8",
    "Random Forest":         "#0d47a1",
    "Attention-BiLSTM":      "#e84e1a",
    "CNN-LSTM":              "#e8a21a",
    "Transformer":           "#9c27b0",
    "Transformer-Attention": "#009688",
}
ML_MODELS  = ["SVM", "Random Forest"]
DL_MODELS  = ["Attention-BiLSTM", "CNN-LSTM", "Transformer", "Transformer-Attention"]
ALL_MODELS = ML_MODELS + DL_MODELS

# ═══════════════════════════════════════════════════════════════════════════
# 3. DEEP LEARNING ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════
class AttentionBiLSTM(nn.Module):
    """
    Exact dims from checkpoint:
      lstm.weight_ih_l0 : [256, 64]  → input_dim=64, hidden_dim=64 (BiLSTM: 4*64=256)
      lstm.weight_ih_l0_reverse: [256, 64]  → bidirectional confirmed
      attention.weight  : inferred [1, 128]
      fc.weight         : [7, 128]   → num_classes=7, hidden_dim*2=128
    """
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=1,
                 dropout=0.3, num_classes=7):
        super().__init__()
        self.lstm      = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                 batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.dropout   = nn.Dropout(dropout)
        self.fc        = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        w   = torch.softmax(self.attention(out), dim=1)
        ctx = (out * w).sum(dim=1)
        return self.fc(self.dropout(ctx))


class CNNLSTMModel(nn.Module):
    """
    Exact dims from checkpoint:
      conv1.weight : [64, 64, 3]  → in_channels=64, out_channels=64
      lstm.weight_ih_l0: [256, 64] → 4*hidden=256, hidden_dim=64, input=64 (conv output)
      lstm is UNIDIRECTIONAL (no _reverse keys in saved model → bidirectional=False)
      fc.weight    : [7, 64]      → num_classes=7, hidden_dim=64
    """
    def __init__(self, input_dim=64, num_filters=64,
                 kernel_size=3, hidden_dim=64, num_layers=1,
                 dropout=0.3, num_classes=7):
        super().__init__()
        self.conv1   = nn.Conv1d(input_dim, num_filters, kernel_size, padding=1)
        self.relu    = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm    = nn.LSTM(num_filters, hidden_dim, num_layers=num_layers,
                               batch_first=True, bidirectional=False)  # ← unidirectional
        self.fc      = nn.Linear(hidden_dim, num_classes)              # ← hidden_dim not *2

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))


class TransformerModel(nn.Module):
    """
    Exact dims from checkpoint:
      input_projection.weight      : [64, 64]    → input_dim=64, d_model=64
      transformer.layers.0.linear1.weight: [2048, 64] → dim_feedforward=2048
      transformer.layers.0.linear2.weight: [64, 2048]
      fc.weight                    : [7, 64]      → num_classes=7
    """
    def __init__(self, input_dim=64, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=2048, dropout=0.1, num_classes=7):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.transformer(self.dropout(x))
        return self.fc(x.mean(dim=1))


class TransformerAttentionModel(nn.Module):
    """
    Exact dims from checkpoint:
      input_projection.weight      : [64, 64]    → input_dim=64, d_model=64
      transformer.layers.*.linear1 : [2048, 64]  → dim_feedforward=2048
      attention.weight             : [1, 64]
      fc.weight                    : [7, 64]      → num_classes=7
    """
    def __init__(self, input_dim=64, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=2048, dropout=0.1, num_classes=7):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.attention   = nn.Linear(d_model, 1)
        self.dropout     = nn.Dropout(dropout)
        self.fc          = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x   = self.input_projection(x)
        x   = self.transformer(self.dropout(x))
        w   = torch.softmax(self.attention(x), dim=1)
        ctx = (x * w).sum(dim=1)
        return self.fc(ctx)


# ═══════════════════════════════════════════════════════════════════════════
# 4. MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════

# Resolve paths relative to this script — works on Streamlit Cloud & local
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def model_path(filename):
    return os.path.join(BASE_DIR, filename)

@st.cache_resource(show_spinner=False)
def load_all_models():
    device = torch.device("cpu")
    models = {}
    errors = {}

    # ── ML models ────────────────────────────────────────────────────
    for name, fname in [("SVM", "SVM.pkl"), ("Random Forest", "RandomForest.pkl")]:
        path = model_path(fname)
        try:
            models[name] = joblib.load(path)
        except Exception as e:
            errors[name] = str(e)

    # ── DL models ────────────────────────────────────────────────────
    dl_specs = [
        ("Attention-BiLSTM",      "best_attention_bilstm.pth",      AttentionBiLSTM),
        ("CNN-LSTM",              "best_cnn_lstm.pth",               CNNLSTMModel),
        ("Transformer",           "best_transformer.pth",            TransformerModel),
        ("Transformer-Attention", "best_transformer_attention.pth",  TransformerAttentionModel),
    ]
    for name, fname, cls in dl_specs:
        path = model_path(fname)
        try:
            # Load without weights_only restriction so OrderedDict works
            raw = torch.load(path, map_location=device, weights_only=False)

            # Extract state dict from any wrapper format
            if isinstance(raw, dict):
                if "model_state_dict" in raw:
                    state_dict = raw["model_state_dict"]
                elif "state_dict" in raw:
                    state_dict = raw["state_dict"]
                else:
                    state_dict = raw
            elif hasattr(raw, "state_dict"):
                raw.eval()
                models[name] = raw
                continue
            else:
                errors[name] = f"Unknown format: {type(raw).__name__}"
                continue

            # Instantiate architecture and load weights
            m = cls()
            try:
                m.load_state_dict(state_dict, strict=True)
                m.eval()
                models[name] = m
            except RuntimeError:
                # CNN-LSTM: auto-detect exact dims from checkpoint tensors
                if name == "CNN-LSTM":
                    ck_keys     = list(state_dict.keys())
                    has_reverse = any("reverse" in k for k in ck_keys)
                    ih   = state_dict.get("lstm.weight_ih_l0")
                    fc_w = state_dict.get("fc.weight")
                    if ih is not None and fc_w is not None:
                        hidden_dim = ih.shape[0] // 4
                        fc_in      = fc_w.shape[1]
                        fc_out     = fc_w.shape[0]
                        m2 = CNNLSTMModel.__new__(CNNLSTMModel)
                        nn.Module.__init__(m2)
                        m2.conv1   = nn.Conv1d(64, 64, 3, padding=1)
                        m2.relu    = nn.ReLU()
                        m2.dropout = nn.Dropout(0.3)
                        m2.lstm    = nn.LSTM(64, hidden_dim, num_layers=1,
                                            batch_first=True, bidirectional=has_reverse)
                        m2.fc      = nn.Linear(fc_in, fc_out)
                        m2.load_state_dict(state_dict, strict=True)
                        m2.eval()
                        models[name] = m2
                    else:
                        errors[name] = f"CNN-LSTM: keys={ck_keys[:6]} has_reverse={has_reverse}"
                else:
                    raise

        except Exception as e:
            errors[name] = f"{type(e).__name__}: {str(e)[:300]}"

    return models, errors, device


# ═══════════════════════════════════════════════════════════════════════════
# 5. PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════
def clean_columns(df):
    df = df.copy()
    df.columns = [re.sub(r'\.\d+$', '', c) for c in df.columns]
    return df

def remove_duplicate_cols(df):
    return df.loc[:, ~df.columns.duplicated()]

def npy_to_dataframe(arr):
    if arr.ndim == 1: arr = arr.reshape(-1, 1)
    return pd.DataFrame(arr, columns=[f"feature_{i}" for i in range(arr.shape[1])])

def rinex_to_dataframe(text):
    rows = []
    for line in text.splitlines():
        nums = re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', line)
        if len(nums) >= 4:
            rows.append([float(v) for v in nums])
    if not rows:
        raise ValueError("No numeric data extracted from RINEX file.")
    max_len = max(len(r) for r in rows)
    padded  = [r + [np.nan] * (max_len - len(r)) for r in rows]
    df = pd.DataFrame(padded, columns=[f"obs_{i}" for i in range(max_len)])
    return df.dropna(axis=1, thresh=len(padded) // 2)

def align_features(df, target_n):
    n = df.shape[1]
    if n < target_n:
        for i in range(target_n - n):
            df[f"__pad_{i}"] = 0.0
    elif n > target_n:
        df = df.iloc[:, :target_n]
    return df

def build_ml_features(df_feat, window=WINDOW, target_features=ML_N_FEATURES):
    """
    Build per-window stat features for ML models.
    Concatenates [mean, std, min, max] per column then pads/trims to target_features.
    Uses 4 stats × n_cols. With target_features=256: expects 64 input columns (4×64=256).
    """
    X = df_feat.values.astype(np.float32)
    rows = []
    n_windows = (len(X) - window) // window + 1
    for i in range(n_windows):
        start = i * window
        end   = start + window
        if end > len(X):
            break
        w = X[start:end]
        feat = np.concatenate([
            np.mean(w, 0), np.std(w, 0),
            np.min(w, 0),  np.max(w, 0),
        ])
        rows.append(feat)
    if not rows:
        return np.zeros((0, target_features), dtype=np.float32)
    arr = np.array(rows, dtype=np.float32)
    # Pad or trim to exactly target_features columns
    if arr.shape[1] < target_features:
        arr = np.hstack([arr, np.zeros(
            (arr.shape[0], target_features - arr.shape[1]), dtype=np.float32)])
    else:
        arr = arr[:, :target_features]
    return arr


def build_sequences(df_feat, window=WINDOW, target_dim=DL_INPUT_DIM):
    """
    Build (n_windows, window, target_dim) sequences for DL models.
    Uses non-overlapping windows. Pads/trims feature dimension to target_dim.
    """
    X = df_feat.values.astype(np.float32)
    seqs = []
    n_windows = (len(X) - window) // window + 1
    for i in range(n_windows):
        start = i * window
        end   = start + window
        if end > len(X):
            break
        seqs.append(X[start:end])
    if not seqs:
        return np.zeros((0, window, target_dim), dtype=np.float32)
    arr = np.array(seqs, dtype=np.float32)   # (n_win, window, n_cols)
    # Pad or trim feature dimension
    n_cols = arr.shape[2]
    if n_cols < target_dim:
        pad = np.zeros((arr.shape[0], window, target_dim - n_cols), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=2)
    else:
        arr = arr[:, :, :target_dim]
    return arr


# ═══════════════════════════════════════════════════════════════════════════
# 6. INFERENCE
# ═══════════════════════════════════════════════════════════════════════════
def predict_ml(model, X):
    """
    Run ML model inference. Handles both binary and multi-class (7-class) models.
    Returns binary preds (0=clean, 1=spoofed) and P(spoofed) probability.
    """
    raw_preds = model.predict(X)
    n_classes = len(model.classes_) if hasattr(model, "classes_") else 2

    if n_classes == 2:
        # Binary: class 0=clean, 1=spoofed
        binary_preds = raw_preds.astype(int)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[:, 1]
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(X)
            proba = 1 / (1 + np.exp(-raw if raw.ndim == 1 else raw[:, 1]))
        else:
            proba = binary_preds.astype(float)
    else:
        # Multi-class (7 classes): class 0=clean, 1-6=spoofed variants
        # Binary: any non-zero prediction = spoofed
        binary_preds = (raw_preds != 0).astype(int)
        if hasattr(model, "predict_proba"):
            proba_all = model.predict_proba(X)   # (n, 7)
            proba = 1.0 - proba_all[:, 0]        # P(spoofed) = 1 - P(clean)
        elif hasattr(model, "decision_function"):
            df_ = model.decision_function(X)
            proba = binary_preds.astype(float)
        else:
            proba = binary_preds.astype(float)

    return binary_preds.astype(int), proba.astype(float)


def predict_dl(model, X, device):
    """
    Run DL model inference. Handles both binary and multi-class (7-class) models.
    Returns binary preds (0=clean, 1=spoofed) and P(spoofed) probability.
    """
    model.eval()
    t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(t)
        probs  = torch.softmax(logits, 1).cpu().numpy()
    n_classes = probs.shape[1]
    if n_classes == 2:
        preds = (probs[:, 1] > 0.5).astype(int)
        proba = probs[:, 1]
    else:
        # Multi-class: class 0=clean, 1-6=spoofed variants
        preds = (np.argmax(probs, axis=1) != 0).astype(int)
        proba = 1.0 - probs[:, 0]   # P(spoofed) = 1 - P(clean)
    return preds.astype(int), proba.astype(float)


# ═══════════════════════════════════════════════════════════════════════════
# 7. METRICS & CHARTS
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred, y_proba, name):
    try:   auc = roc_auc_score(y_true, y_proba)
    except: auc = float("nan")
    return dict(
        model=name,
        accuracy =accuracy_score(y_true, y_pred),
        f1       =f1_score(y_true, y_pred, average="binary", zero_division=0),
        precision=precision_score(y_true, y_pred, average="binary", zero_division=0),
        recall   =recall_score(y_true, y_pred, average="binary", zero_division=0),
        auc_roc  =auc,
    )

def confusion_heatmap(y_true, y_pred, title):
    cm  = confusion_matrix(y_true, y_pred)
    lbl = ["Clean", "Spoofed"]
    fig = px.imshow(cm, text_auto=True, x=lbl, y=lbl,
                    color_continuous_scale="Blues", title=title,
                    labels=dict(x="Predicted", y="Actual"))
    fig.update_layout(margin=dict(t=50,b=10), height=300)
    return fig

def metric_radar(metrics_list):
    dims   = ["accuracy","f1","precision","recall","auc_roc"]
    labels = [d.replace("_"," ").title() for d in dims]
    colors = list(MODEL_COLORS.values())
    fig    = go.Figure()
    for i, m in enumerate(metrics_list):
        vals = [m[d] for d in dims]
        fig.add_trace(go.Scatterpolar(
            r=vals+[vals[0]], theta=labels+[labels[0]],
            fill="toself", name=m["model"],
            line_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0,1])),
        legend=dict(orientation="h"), title="Performance Radar",
        height=400, margin=dict(t=50,b=10),
    )
    return fig

def proba_timeline_fig(results):
    fig = go.Figure()
    for name, res in results.items():
        fig.add_trace(go.Scatter(
            y=res["proba"], name=name, mode="lines",
            line=dict(color=MODEL_COLORS.get(name,"#888")),
        ))
    fig.update_layout(
        title="Spoofing Probability per Window",
        xaxis_title="Window", yaxis_title="P(Spoofed)",
        height=320, margin=dict(t=50,b=20), legend=dict(orientation="h"),
    )
    return fig

def pred_timeline_fig(results):
    fig = go.Figure()
    for name, res in results.items():
        fig.add_trace(go.Scatter(
            y=res["preds"], name=name, mode="lines+markers",
            line=dict(color=MODEL_COLORS.get(name,"#888")),
        ))
    fig.update_layout(
        title="Predictions per Window (0=Clean, 1=Spoofed)",
        xaxis_title="Window",
        yaxis=dict(tickvals=[0,1], ticktext=["Clean","Spoofed"]),
        height=300, margin=dict(t=50,b=20), legend=dict(orientation="h"),
    )
    return fig

def proba_hist_fig(results):
    fig = go.Figure()
    for name, res in results.items():
        fig.add_trace(go.Histogram(
            x=res["proba"], name=name, opacity=0.6, nbinsx=30,
            marker_color=MODEL_COLORS.get(name,"#888"),
        ))
    fig.update_layout(
        barmode="overlay",
        title="Spoofing Probability Distribution",
        xaxis_title="P(Spoofed)", yaxis_title="Windows",
        height=300, margin=dict(t=50,b=20), legend=dict(orientation="h"),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 8. RTKLIB AUTO-INSTALLER
# ═══════════════════════════════════════════════════════════════════════════
RTKLIB_DIR = os.path.expanduser("~/.rtklib")
RTKLIB_EXE = os.path.join(RTKLIB_DIR, "rnx2rtkp")

def is_rtklib_installed():
    return os.path.isfile(RTKLIB_EXE) and os.access(RTKLIB_EXE, os.X_OK)

def install_rtklib_linux(ph):
    os.makedirs(RTKLIB_DIR, exist_ok=True)
    # Strategy 1: apt-get
    ph.info("📦 Trying apt-get install rtklib…")
    try:
        subprocess.run(["apt-get","install","-y","-q","rtklib"],
                       capture_output=True, timeout=120)
        apt = shutil.which("rnx2rtkp")
        if apt:
            shutil.copy(apt, RTKLIB_EXE); os.chmod(RTKLIB_EXE, 0o755)
            ph.success("✅ RTKLIB installed via apt-get!"); return True
    except Exception: pass
    # Strategy 2: requests + compile
    ph.info("📥 Downloading RTKLIB source…")
    try:
        r = requests.get("https://github.com/tomojitakasu/RTKLIB/archive/refs/tags/v2.4.3b34.zip",
                         timeout=120, stream=True)
        if r.status_code == 200:
            zp = "/tmp/rtklib.zip"
            with open(zp,"wb") as f:
                for chunk in r.iter_content(65536): f.write(chunk)
            with zipfile.ZipFile(zp) as z: z.extractall("/tmp/")
            ph.info("🔨 Compiling rnx2rtkp…")
            subprocess.run(["make","-C","/tmp/RTKLIB-2.4.3b34/app/rnx2rtkp/gcc","-j2"],
                           capture_output=True, timeout=300)
            src = "/tmp/RTKLIB-2.4.3b34/app/rnx2rtkp/gcc/rnx2rtkp"
            if os.path.isfile(src):
                shutil.copy(src, RTKLIB_EXE); os.chmod(RTKLIB_EXE, 0o755)
                ph.success("✅ RTKLIB compiled!"); return True
    except Exception as e: ph.warning(f"Compile failed: {e}")
    ph.error("❌ RTKLIB installation failed. Add `rtklib` to packages.txt.")
    return False

def get_rtklib_exe(ph=None):
    if os.name == "nt": return _find_rtklib_windows()
    if is_rtklib_installed(): return RTKLIB_EXE
    if ph and install_rtklib_linux(ph): return RTKLIB_EXE
    raise RuntimeError("RTKLIB not available.")

def _find_rtklib_windows():
    found = shutil.which("rnx2rtkp") or shutil.which("rnx2rtkp.exe")
    if found: return found
    for p in [r"C:\RTKLIB\bin\rnx2rtkp.exe",
               r"C:\RTKLIB-2.4.3b34\bin\rnx2rtkp.exe"]:
        if os.path.isfile(p): return p
    raise FileNotFoundError("rnx2rtkp.exe not found. Enter full path below.")


# ═══════════════════════════════════════════════════════════════════════════
# 9. NAVIGATION FILE DOWNLOADER
# ═══════════════════════════════════════════════════════════════════════════
def download_nav_file(year, month, day, save_dir):
    doy  = datetime(year, month, day).timetuple().tm_yday
    yr2  = str(year)[2:]
    name = f"brdc{doy:03d}0.{yr2}n"
    urls = [
        f"https://cddis.nasa.gov/archive/gnss/data/daily/{year}/{doy:03d}/{yr2}n/{name}.gz",
        f"https://cddis.nasa.gov/archive/gnss/data/daily/{year}/brdc/{name}.gz",
        f"https://igs.ign.fr/pub/igs/data/daily/{year}/{doy:03d}/{name}.gz",
        f"https://ftp.gfz-potsdam.de/pub/GNSS/data/daily/{year}/{doy:03d}/{name}.gz",
    ]
    hdrs = {"User-Agent": "Mozilla/5.0 RTKLIB/2.4.3", "Accept": "*/*"}
    for url in urls:
        try:
            r = requests.get(url, headers=hdrs, timeout=30, stream=True)
            if r.status_code == 200:
                gz = os.path.join(save_dir, name+".gz")
                nav = os.path.join(save_dir, name)
                with open(gz,"wb") as f:
                    for chunk in r.iter_content(8192): f.write(chunk)
                with gzip.open(gz,"rb") as fi, open(nav,"wb") as fo:
                    shutil.copyfileobj(fi, fo)
                os.remove(gz)
                return nav
        except Exception: continue
    raise RuntimeError("Could not download nav file from any IGS mirror.")


# ═══════════════════════════════════════════════════════════════════════════
# 10. RTKLIB PROCESSING HELPERS
# ═══════════════════════════════════════════════════════════════════════════
RTKLIB_CHANNELS = 8
SAMPLING_FREQ   = 4_000_000

def write_rtklib_conf(path):
    with open(path,"w") as f:
        f.write("pos1-posmode=single\npos1-elmask=15\npos1-ionoopt=brdc\n"
                "pos1-tropopt=saas\npos1-sateph=brdc\nout-solformat=llh\n"
                "out-outstat=residual\nout-outhead=on\nout-outopt=on\n")

def run_rnx2rtkp(obs, nav, exe, work):
    conf = os.path.join(work,"rtklib.conf")
    out  = os.path.join(work,"output.pos")
    write_rtklib_conf(conf)
    subprocess.run([exe,"-k",conf,"-o",out,obs,nav],
                   capture_output=True, text=True, timeout=300)
    if not os.path.isfile(out) or os.path.getsize(out)==0:
        raise RuntimeError("RTKLIB produced no output.")
    return out

def parse_pos_file(path):
    rows=[]
    with open(path) as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("%"): continue
            p=line.split()
            if len(p)<8: continue
            try:
                rows.append({"date":p[0],"time":p[1],"lat":float(p[2]),
                    "lon":float(p[3]),"height":float(p[4]),"fix_status":int(p[5]),
                    "num_sats":int(p[6]),"pdop":float(p[7]),
                    "sdx":float(p[8]) if len(p)>8 else 0.0,
                    "sdy":float(p[9]) if len(p)>9 else 0.0,
                    "sdz":float(p[10]) if len(p)>10 else 0.0,
                })
            except: continue
    if not rows: raise ValueError("No epochs parsed from RTKLIB output.")
    return pd.DataFrame(rows)

def pos_to_channel_csv(pos_df):
    lat=pos_df["lat"].values; lon=pos_df["lon"].values
    height=pos_df["height"].values; pdop=pos_df["pdop"].values
    h_jump=np.abs(np.gradient(height))
    lat_j=np.abs(np.gradient(lat)); lon_j=np.abs(np.gradient(lon))
    records=[]
    for i in range(len(pos_df)):
        row={}
        for ch in range(RTKLIB_CHANNELS):
            prn=(ch+1)+(i%4)
            cn0=max(20.0, 45.0-pdop[i]*3.0+ch*0.5)
            row.update({
                f"ch{ch}_channel_id":ch, f"ch{ch}_prn":prn,
                f"ch{ch}_doppler_coarse":h_jump[i]*10+ch*0.1,
                f"ch{ch}_tracking_flag":1,
                f"ch{ch}_sampling_freq":SAMPLING_FREQ,
                f"ch{ch}_carrier_phase":lat[i]*1e6+lon[i]*1e6+ch*1000,
                f"ch{ch}_doppler_fine":lat_j[i]*1e4+lon_j[i]*1e4+ch*0.01,
                f"ch{ch}_cn0":cn0, f"ch{ch}_prompt_i":0.0,
                f"ch{ch}_prompt_q":0.0, f"ch{ch}_time_index":float(i),
            })
        records.append(row)
    expected=[f"ch{ch}_{f}" for ch in range(8)
              for f in ["channel_id","prn","doppler_coarse","tracking_flag",
                        "sampling_freq","carrier_phase","doppler_fine",
                        "cn0","prompt_i","prompt_q","time_index"]]
    return pd.DataFrame(records)[expected]


# ═══════════════════════════════════════════════════════════════════════════
# 11. MAIN APP
# ═══════════════════════════════════════════════════════════════════════════
def main():
    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/GPS_Satellite_NASA_art-iif.jpg/320px-GPS_Satellite_NASA_art-iif.jpg",
                 use_container_width=True)
        st.markdown("## 🛰️ GNSS Spoofing Detector")
        st.markdown("""
        Upload GNSS signal data (CSV, NPY, or RINEX).  
        The app runs **6 models simultaneously** and compares results.

        ---
        **🛰️ RTKLIB Pipeline**  
        Upload raw RINEX files in Section 0.

        ---
        **ML Models**  
        • SVM | • Random Forest  
        Input: 256 window statistics

        **DL Models**  
        • Attention-BiLSTM  
        • CNN-LSTM  
        • Transformer  
        • Transformer-Attention  
        Input: (30, 55) sequences

        ---
        **Labels** (if `label` column present):  
        `0` = Clean · `1` = Spoofed
        """)
        st.markdown("---")
        window_override = st.slider("Window size", 10, 60, WINDOW,
                                    help="Sliding-window length.")

    # ── Header ───────────────────────────────────────────────────────
    st.title("🛰️ GNSS Spoofing Detection — Six Model Comparison")
    st.markdown("*Supports CSV · NPY · RINEX — automatic preprocessing & six-model comparison*")

    # ── Load models ──────────────────────────────────────────────────
    with st.spinner("Loading models…"):
        models, load_errors, device = load_all_models()

    loaded = [n for n in ALL_MODELS if n in models]
    failed = [n for n in ALL_MODELS if n in load_errors]

    if loaded:
        st.success(f"✅ {len(loaded)}/6 models loaded: {', '.join(loaded)}", icon="🤖")
    if failed:
        with st.expander(f"⚠️ {len(failed)} model(s) failed to load"):
            for n in failed:
                st.error(f"**{n}**: {load_errors[n]}")
    if not loaded:
        st.error("No models loaded. Place model files in the app root folder.")
        st.stop()

    # ── SECTION 0 — RTKLIB ───────────────────────────────────────────
    st.markdown('<div class="section-header">🛰️ Section 0 — RTKLIB Preprocessing (RINEX → Features)</div>',
                unsafe_allow_html=True)
    with st.expander("📡 Convert RINEX files using RTKLIB (skip if uploading CSV)", expanded=False):
        if os.name == "nt":
            rtklib_path_input = st.text_input(
                "rnx2rtkp.exe path (blank = auto-detect)",
                placeholder=r"e.g. C:\RTKLIB-2.4.3b34\bin\rnx2rtkp.exe")
        else:
            rtklib_path_input = ""
            if is_rtklib_installed():
                st.success("✅ RTKLIB ready on server.")
            else:
                st.info("🔧 RTKLIB will auto-install on first run (~2 min).")


        st.markdown("---")
        st.markdown("#### 📂 Upload RINEX files")
        st.warning("⚠️ Rename `.23o`→`_obs.rnx` and `.23n`→`_nav.rnx` before uploading.")
        co, cn = st.columns(2)
        obs_file = co.file_uploader("📄 Observation (.rnx/.obs/.txt)",
                                    type=["rnx","obs","rnx3","txt"], key="rtklib_obs")
        nav_file = cn.file_uploader("📄 Navigation (.rnx/.nav/.txt)",
                                    type=["nav","rnx","rnx3","txt"], key="rtklib_nav")

        nav_ready = nav_file is not None

        if obs_file and nav_ready:
            if st.button("🚀 Run RTKLIB & Convert to Features", type="primary"):
                ph = st.empty()
                with st.spinner("Running RTKLIB…"):
                    try:
                        exe = (rtklib_path_input.strip() if rtklib_path_input.strip()
                               else get_rtklib_exe(ph))
                        ph.success(f"✅ RTKLIB at `{exe}`")
                        with tempfile.TemporaryDirectory() as tmp:
                            obs_path = os.path.join(tmp, obs_file.name)
                            open(obs_path,"wb").write(obs_file.read())
                            nav_path = os.path.join(tmp, nav_file.name)
                            open(nav_path,"wb").write(nav_file.read())
                            pos  = run_rnx2rtkp(obs_path, nav_path, exe, tmp)
                            pdf  = parse_pos_file(pos)
                            st.success(f"✅ {len(pdf)} epochs parsed")
                            with st.expander("RTKLIB solution preview"):
                                st.dataframe(pdf.head(20), use_container_width=True)
                            cdf = pos_to_channel_csv(pdf)
                            st.success(f"✅ Converted → {cdf.shape}")
                            st.session_state["rtklib_csv"] = cdf
                            st.download_button("⬇️ Download features CSV",
                                               data=cdf.to_csv(index=False).encode(),
                                               file_name="rtklib_features.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"❌ {e}")
        elif obs_file:
            st.warning("⚠️ Navigation file needed.")
        else:
            st.info("📋 1️⃣ Download nav → 2️⃣ Upload obs → 3️⃣ Run RTKLIB")

    # ── SECTION 1 — Upload ───────────────────────────────────────────
    st.markdown('<div class="section-header">📂 Section 1 — Upload Data</div>',
                unsafe_allow_html=True)
    rtklib_result = st.session_state.get("rtklib_csv")
    if rtklib_result is not None:
        st.success("✅ Using RTKLIB-converted features from Section 0.")
    uploaded = st.file_uploader("Upload a GNSS data file",
                                type=["csv","npy","obs","rnx","txt"],
                                help="CSV preferred. Skip if you ran RTKLIB above.")
    if uploaded is None and rtklib_result is None:
        st.info("👆 Run RTKLIB above or upload a CSV/NPY file.")
        st.stop()

    # ── SECTION 2 — Preprocessing ────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Section 2 — Preprocessing & Format Conversion</div>',
                unsafe_allow_html=True)
    with st.spinner("Parsing file…"):
        try:
            if rtklib_result is not None and uploaded is None:
                df = rtklib_result.copy()
                st.success(f"✅ RTKLIB features → {df.shape[0]} rows, {df.shape[1]} cols.")
            else:
                raw  = uploaded.read()
                name = uploaded.name.lower()
                if name.endswith(".npy"):
                    arr = np.load(io.BytesIO(raw), allow_pickle=True)
                    df  = npy_to_dataframe(arr)
                    st.success(f"✅ NPY {arr.shape} loaded.")
                elif name.endswith((".obs",".rnx")):
                    df  = rinex_to_dataframe(raw.decode("utf-8","ignore"))
                    st.success(f"✅ RINEX parsed → {df.shape}.")
                else:
                    df  = pd.read_csv(io.BytesIO(raw))
                    st.success(f"✅ CSV loaded → {df.shape[0]} rows, {df.shape[1]} cols.")
            df = remove_duplicate_cols(clean_columns(df))
        except Exception as e:
            st.error(f"❌ Parse error: {e}"); st.stop()

    st.download_button("⬇️ Download converted CSV",
                       data=df.to_csv(index=False).encode(),
                       file_name="converted_data.csv", mime="text/csv")
    with st.expander("🔍 Preview (first 50 rows)", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.metric("Label col", "Yes ✅" if "label" in df.columns else "No ⚠️")

    # Separate labels
    if "label" in df.columns:
        labels_raw = df["label"].values
        df_feat    = df.drop(columns=["label"])
    else:
        labels_raw = None
        df_feat    = df.copy()

    df_feat = df_feat.select_dtypes(include=[np.number])
    if df_feat.shape[1] == 0:
        st.error("No numeric feature columns found."); st.stop()

    # ── Drop constant/identifier columns that were excluded during training ──
    # Training used 64 features = 8 channels × 8 features
    # Dropped: channel_id, tracking_flag, sampling_freq (constant per channel)
    DROP_SUFFIXES = ["channel_id", "tracking_flag", "sampling_freq"]
    cols_to_drop = [c for c in df_feat.columns
                    if any(c.endswith(s) for s in DROP_SUFFIXES)]
    if cols_to_drop:
        df_feat = df_feat.drop(columns=cols_to_drop)

    st.info(f"ℹ️ Using {df_feat.shape[1]} features after dropping constant columns "
            f"({len(cols_to_drop)} dropped: channel_id, tracking_flag, sampling_freq)")

    scaler = StandardScaler()
    scaled = pd.DataFrame(scaler.fit_transform(df_feat), columns=df_feat.columns)

    # ── SECTION 3 — Predictions ──────────────────────────────────────
    st.markdown('<div class="section-header">🔮 Section 3 — Running Predictions (All 6 Models)</div>',
                unsafe_allow_html=True)

    win = window_override

    # Build ML features: (n_win, 256) — stats per window, padded/trimmed inside function
    X_ml = build_ml_features(scaled, window=win, target_features=ML_N_FEATURES)

    # Build DL sequences: (n_win, win, 64) — padded/trimmed inside function
    X_dl = build_sequences(scaled, window=win, target_dim=DL_INPUT_DIM)

    n_win = min(len(X_ml), len(X_dl))
    if n_win == 0:
        st.error(f"Not enough rows for window size {win}. "
                 f"Need at least {win*2} rows, got {len(scaled)}."); st.stop()

    X_ml = X_ml[:n_win]
    X_dl = X_dl[:n_win]

    if labels_raw is not None:
        y_win = np.array([int(np.any(labels_raw[i*win:(i+1)*win])) for i in range(n_win)])
    else:
        y_win = None

    st.info(f"📊 {n_win} windows of size {win} built from {scaled.shape[0]} rows · "
            f"ML input: {X_ml.shape} · DL input: {X_dl.shape}")

    # ── Run all models ────────────────────────────────────────────────
    results = {}
    progress = st.progress(0, text="Running models…")
    for idx, name in enumerate(ALL_MODELS):
        if name not in models:
            progress.progress((idx+1)/len(ALL_MODELS))
            continue
        try:
            if name in ML_MODELS:
                preds, proba = predict_ml(models[name], X_ml)
            else:
                preds, proba = predict_dl(models[name], X_dl, device)
            results[name] = {"preds": preds, "proba": proba}
        except Exception as e:
            st.warning(f"⚠️ {name} inference failed: {e}")
        progress.progress((idx+1)/len(ALL_MODELS), text=f"✅ {name} done")

    progress.empty()

    if not results:
        st.error("All model inferences failed."); st.stop()

    # ── CONSENSUS VERDICT BANNER ─────────────────────────────────────
    votes_spoof = sum(1 for res in results.values() if res["preds"].mean() > 0.4)
    votes_clean = len(results) - votes_spoof
    consensus   = votes_spoof > votes_clean
    avg_prob    = np.mean([res["proba"].mean() for res in results.values()]) * 100

    if consensus:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#c0392b,#e74c3c);
                    color:white;padding:24px 28px;border-radius:14px;
                    font-size:1.4rem;font-weight:700;text-align:center;margin:16px 0;">
            🔴 GNSS SPOOFING DETECTED<br>
            <span style="font-size:1rem;font-weight:400;opacity:.9;">
            {votes_spoof}/{len(results)} models agree · Avg spoofing probability: {avg_prob:.1f}%
            </span>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a7a4a,#27ae60);
                    color:white;padding:24px 28px;border-radius:14px;
                    font-size:1.4rem;font-weight:700;text-align:center;margin:16px 0;">
            🟢 SIGNAL APPEARS CLEAN<br>
            <span style="font-size:1rem;font-weight:400;opacity:.9;">
            {votes_clean}/{len(results)} models agree · Avg spoofing probability: {avg_prob:.1f}%
            </span>
        </div>""", unsafe_allow_html=True)

    # ── Summary table ─────────────────────────────────────────────────
    st.markdown("### 📋 Per-Model Results")
    summary_rows = []
    for name, res in results.items():
        spoof_n   = int(res["preds"].sum())
        spoof_pct = 100 * res["preds"].mean()
        avg_p     = 100 * res["proba"].mean()
        verdict   = "🔴 Spoofing" if spoof_pct > 40 else "🟢 Clean"
        summary_rows.append({
            "Model":             name,
            "Verdict":           verdict,
            "Spoofed Windows":   f"{spoof_n} / {n_win}",
            "Spoofed %":         f"{spoof_pct:.1f}%",
            "Avg P(Spoofed)":    f"{avg_p:.1f}%",
        })
    st.dataframe(pd.DataFrame(summary_rows).set_index("Model"),
                 use_container_width=True)

    # ── Per-model cards ───────────────────────────────────────────────
    cols = st.columns(3)
    for i, (name, res) in enumerate(results.items()):
        with cols[i % 3]:
            spoof_pct = 100 * res["preds"].mean()
            verdict   = "🔴 SPOOFING" if spoof_pct > 40 else "🟢 Clean"
            color     = "alert-spoof" if spoof_pct > 40 else "alert-clean"
            st.metric(label=name,
                      value=f"{res['preds'].sum()} / {n_win}",
                      delta=f"{spoof_pct:.1f}% spoofed")
            st.markdown(f'<span class="{color}">{verdict}</span>',
                        unsafe_allow_html=True)

    # ── SECTION 4 — Analysis ──────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Section 4 — Analysis</div>',
                unsafe_allow_html=True)

    # Only show tabs that always work (no label required)
    if y_win is not None and len(np.unique(y_win)) > 1:
        tabs = st.tabs(["📈 Metrics", "🗺️ Confusion Matrices",
                        "📉 Probabilities", "📋 Timeline", "⬇️ Export"])
        tab_metrics, tab_cm, tab_prob, tab_timeline, tab_export = tabs

        with tab_metrics:
            metrics_list = [compute_metrics(y_win, res["preds"], res["proba"], n)
                            for n, res in results.items()]
            mdf = pd.DataFrame(metrics_list).set_index("model")
            st.dataframe((mdf * 100).round(2).astype(str).add(" %"),
                         use_container_width=True)

            metric_names = ["accuracy","f1","precision","recall","auc_roc"]
            fig_bar = go.Figure()
            for m in metrics_list:
                fig_bar.add_trace(go.Bar(
                    name=m["model"],
                    x=[mn.replace("_"," ").title() for mn in metric_names],
                    y=[m[mn] for mn in metric_names],
                    marker_color=MODEL_COLORS.get(m["model"],"#888"),
                    text=[f"{m[mn]*100:.1f}%" for mn in metric_names],
                    textposition="outside",
                ))
            fig_bar.update_layout(barmode="group", title="Metric Comparison",
                                  yaxis=dict(range=[0,1.2]), height=420,
                                  margin=dict(t=50,b=20), legend=dict(orientation="h"))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.plotly_chart(metric_radar(metrics_list), use_container_width=True)

            # Best model
            best = max(metrics_list, key=lambda m: m["f1"])
            st.markdown(f"""
            <div class="winner-banner">
                🏆 Best Model: {best['model']} &nbsp;|&nbsp;
                F1 = {best['f1']*100:.1f}% · Accuracy = {best['accuracy']*100:.1f}%
            </div>""", unsafe_allow_html=True)

            with st.expander("📄 Classification Reports"):
                for name, res in results.items():
                    st.markdown(f"**{name}**")
                    st.text(classification_report(y_win, res["preds"],
                                                  target_names=["Clean","Spoofed"],
                                                  zero_division=0))

        with tab_cm:
            names = list(results.keys())
            for row_start in range(0, len(names), 3):
                cols_ = st.columns(3)
                for j, name in enumerate(names[row_start:row_start+3]):
                    with cols_[j]:
                        st.plotly_chart(
                            confusion_heatmap(y_win, results[name]["preds"], name),
                            use_container_width=True)

        with tab_prob:
            st.plotly_chart(proba_hist_fig(results), use_container_width=True)
            st.plotly_chart(proba_timeline_fig(results), use_container_width=True)

        with tab_timeline:
            st.plotly_chart(pred_timeline_fig(results), use_container_width=True)

        with tab_export:
            export = {"Window": list(range(n_win)),
                      "True Label": ["Spoofed" if y else "Clean" for y in y_win]}
            for name, res in results.items():
                safe = name.replace(" ","_").replace("-","_")
                export[f"{safe}_Prediction"] = ["Spoofed" if p else "Clean"
                                                for p in res["preds"]]
                export[f"{safe}_Probability"] = res["proba"].round(4)
            export_df = pd.DataFrame(export)
            st.dataframe(export_df, use_container_width=True, height=400)
            st.download_button("⬇️ Download All Predictions CSV",
                               data=export_df.to_csv(index=False).encode(),
                               file_name="all_predictions.csv", mime="text/csv")

    else:
        # No labels — show only probability and timeline analysis
        tabs = st.tabs(["📉 Probabilities", "📋 Timeline", "⬇️ Export"])
        tab_prob, tab_timeline, tab_export = tabs

        with tab_prob:
            st.plotly_chart(proba_hist_fig(results), use_container_width=True)
            st.plotly_chart(proba_timeline_fig(results), use_container_width=True)

        with tab_timeline:
            st.plotly_chart(pred_timeline_fig(results), use_container_width=True)

        with tab_export:
            export = {"Window": list(range(n_win))}
            for name, res in results.items():
                safe = name.replace(" ","_").replace("-","_")
                export[f"{safe}_Prediction"] = ["Spoofed" if p else "Clean"
                                                for p in res["preds"]]
                export[f"{safe}_Probability"] = res["proba"].round(4)
            export_df = pd.DataFrame(export)
            st.dataframe(export_df, use_container_width=True, height=400)
            st.download_button("⬇️ Download All Predictions CSV",
                               data=export_df.to_csv(index=False).encode(),
                               file_name="all_predictions.csv", mime="text/csv")

    st.markdown("---")
    st.caption(
        "GNSS Spoofing Detection · 6-Model Comparison · "
        "ML: SVM, Random Forest · "
        "DL: Attention-BiLSTM, CNN-LSTM, Transformer, Transformer-Attention"
    )


if __name__ == "__main__":
    main()
