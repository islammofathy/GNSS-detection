"""
GNSS Spoofing Detection — Dual Model Comparison App
=====================================================
Compares:
  • best_model  : sklearn SVC (RBF), expects 352 window-stat features
  • best_dl_model : PyTorch BiLSTM_Attention, expects (batch, seq_len, 88) sequences
"""

import re
import io
import os
import gzip
import shutil
import warnings
import subprocess
import tempfile
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
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, f1_score,
    precision_score, recall_score
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GNSS Spoofing Detector",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# 1. CUSTOM CSS — clean, modern look
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ---- Global ---- */
body { font-family: 'Segoe UI', sans-serif; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f2027, #203a43, #2c5364);
}
section[data-testid="stSidebar"] * { color: #e0eafc !important; }

/* ---- Metric cards ---- */
div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}

/* ---- Section headers ---- */
.section-header {
    background: linear-gradient(90deg, #1a73e8, #0d47a1);
    color: white !important;
    padding: 10px 18px;
    border-radius: 8px;
    margin-bottom: 12px;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: .3px;
}

/* ---- Winner banner ---- */
.winner-banner {
    background: linear-gradient(135deg, #11998e, #38ef7d);
    color: white;
    padding: 18px 24px;
    border-radius: 12px;
    font-size: 1.25rem;
    font-weight: 700;
    text-align: center;
    margin-top: 10px;
}

/* ---- Spoofing alert ---- */
.alert-spoof  { color: #e53e3e; font-weight: 700; }
.alert-clean  { color: #38a169; font-weight: 700; }

/* ---- Tabs ---- */
button[data-baseweb="tab"] { font-size: .95rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# 2. NEURAL NETWORK DEFINITION
#    (must match exactly how it was trained)
# ─────────────────────────────────────────────
class BiLSTM_Attention(nn.Module):
    """Bidirectional LSTM with soft attention — the saved deep model."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=True, bidirectional=True)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.fc   = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        out, _ = self.lstm(x)                          # (B, T, 2H)
        weights = torch.softmax(self.attn(out), dim=1) # (B, T, 1)
        context = (out * weights).sum(dim=1)           # (B, 2H)
        return self.fc(context)


# ─────────────────────────────────────────────
# 3. CONSTANTS (match training script)
# ─────────────────────────────────────────────
WINDOW        = 30    # sliding window size used during training
ML_N_FEATURES = 352  # SVC expects 4 * 88 stat-features per window
DL_INPUT_DIM  = 88   # BiLSTM input size per time-step


# ─────────────────────────────────────────────
# 4. MODEL LOADING (cached — runs once)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load both pickled models from the repo root."""
    # ── sklearn SVC ──────────────────────────────────────────────────
    svc = joblib.load("best_model.pkl")

    # ── PyTorch BiLSTM ───────────────────────────────────────────────
    device = torch.device("cpu")
    dl_model = torch.load(
        "best_dl_model.pkl",
        map_location=device,
        weights_only=False,
    )
    dl_model.eval()

    return svc, dl_model, device


# ─────────────────────────────────────────────
# 5. PREPROCESSING HELPERS
# ─────────────────────────────────────────────
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip duplicate numeric suffixes from column names (e.g. 'col.1' → 'col')."""
    df = df.copy()
    df.columns = [re.sub(r'\.\d+$', '', c) for c in df.columns]
    return df


def remove_duplicate_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()]


def npy_to_dataframe(data: np.ndarray) -> pd.DataFrame:
    """Convert a numpy array to a DataFrame with generic column names."""
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    cols = [f"feature_{i}" for i in range(data.shape[1])]
    return pd.DataFrame(data, columns=cols)


def rinex_to_dataframe(text: str) -> pd.DataFrame:
    """
    Minimal RINEX obs-file parser.
    Extracts numeric observation values line-by-line and returns a DataFrame.
    This is a best-effort parser for demo purposes.
    """
    rows = []
    for line in text.splitlines():
        nums = re.findall(r'[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?', line)
        if len(nums) >= 4:
            rows.append([float(v) for v in nums])
    if not rows:
        raise ValueError("No numeric data extracted from RINEX file.")
    max_len = max(len(r) for r in rows)
    padded  = [r + [np.nan] * (max_len - len(r)) for r in rows]
    cols    = [f"obs_{i}" for i in range(max_len)]
    return pd.DataFrame(padded, columns=cols).dropna(axis=1, thresh=len(padded) // 2)


def build_window_features(df_feat: pd.DataFrame, window: int = WINDOW) -> np.ndarray:
    """
    Reproduce the training feature engineering:
      For each non-overlapping window of `window` rows compute
      [mean, std, var, diff] → concatenate → one row per window.
    Returns shape (n_windows, 4 * n_features).
    """
    X = df_feat.values.astype(np.float32)
    rows = []
    for i in range(0, len(X) - window, window):
        w   = X[i : i + window]
        mean = np.mean(w, axis=0)
        std  = np.std(w,  axis=0)
        var  = np.var(w,  axis=0)
        diff = w[-1] - w[0]
        rows.append(np.concatenate([mean, std, var, diff]))
    return np.array(rows, dtype=np.float32)


def build_sequences(df_feat: pd.DataFrame, window: int = WINDOW) -> np.ndarray:
    """
    Build 3-D sequences for the BiLSTM:  (n_windows, window, n_features).
    """
    X = df_feat.values.astype(np.float32)
    seqs = []
    for i in range(0, len(X) - window, window):
        seqs.append(X[i : i + window])
    return np.array(seqs, dtype=np.float32)   # (N, T, F)


def align_features(df: pd.DataFrame, target_n: int) -> pd.DataFrame:
    """
    Pad or truncate the DataFrame to exactly `target_n` feature columns
    so the models always receive the correct input width.
    """
    n = df.shape[1]
    if n < target_n:
        for i in range(target_n - n):
            df[f"__pad_{i}"] = 0.0
    elif n > target_n:
        df = df.iloc[:, :target_n]
    return df


# ─────────────────────────────────────────────
# 6. PREDICTION HELPERS
# ─────────────────────────────────────────────
def predict_svc(svc, X_win: np.ndarray):
    """Run the sklearn SVC on window-stat features."""
    preds = svc.predict(X_win)
    if hasattr(svc, "predict_proba"):
        proba = svc.predict_proba(X_win)[:, 1]
    elif hasattr(svc, "decision_function"):
        raw   = svc.decision_function(X_win)
        proba = 1 / (1 + np.exp(-raw))   # sigmoid squash
    else:
        proba = preds.astype(float)
    return preds, proba


def predict_bilstm(dl_model, X_seq: np.ndarray, device):
    """Run the BiLSTM-Attention model on sequences."""
    dl_model.eval()
    tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = dl_model(tensor)          # (N, 2)
        probs  = torch.softmax(logits, 1)
        preds  = torch.argmax(probs, 1).cpu().numpy()
        proba  = probs[:, 1].cpu().numpy()
    return preds, proba


# ─────────────────────────────────────────────
# 7. METRIC HELPERS
# ─────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_proba, name: str) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="binary", zero_division=0)
    prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float("nan")
    return dict(model=name, accuracy=acc, f1=f1,
                precision=prec, recall=rec, auc_roc=auc)


# ─────────────────────────────────────────────
# 8. CHART HELPERS
# ─────────────────────────────────────────────
def confusion_heatmap(y_true, y_pred, title: str):
    cm     = confusion_matrix(y_true, y_pred)
    labels = ["Clean (0)", "Spoofed (1)"]
    fig    = px.imshow(
        cm, text_auto=True,
        x=labels, y=labels,
        color_continuous_scale="Blues",
        title=title,
        labels=dict(x="Predicted", y="Actual"),
    )
    fig.update_layout(margin=dict(t=50, b=20), height=320)
    return fig


def metric_radar(metrics_list: list[dict]):
    dims   = ["accuracy", "f1", "precision", "recall", "auc_roc"]
    labels = [d.replace("_", " ").title() for d in dims]
    colors = ["#1a73e8", "#e84e1a"]
    fig    = go.Figure()
    for i, m in enumerate(metrics_list):
        vals = [m[d] for d in dims]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=m["model"],
            line_color=colors[i % len(colors)],
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
        legend=dict(orientation="h"),
        title="Model Performance Radar",
        height=360,
        margin=dict(t=50, b=10),
    )
    return fig


def proba_histogram(proba_svc, proba_dl):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=proba_svc, name="SVC",
                               opacity=.65, nbinsx=30,
                               marker_color="#1a73e8"))
    fig.add_trace(go.Histogram(x=proba_dl,  name="BiLSTM",
                               opacity=.65, nbinsx=30,
                               marker_color="#e84e1a"))
    fig.update_layout(barmode="overlay",
                      title="Predicted Spoofing Probability Distribution",
                      xaxis_title="P(Spoofed)",
                      yaxis_title="Windows",
                      height=300,
                      margin=dict(t=50, b=20))
    return fig


def pred_timeline(preds_svc, preds_dl):
    n   = min(len(preds_svc), len(preds_dl))
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=preds_svc[:n], name="SVC",
                             mode="lines+markers",
                             line=dict(color="#1a73e8")))
    fig.add_trace(go.Scatter(y=preds_dl[:n],  name="BiLSTM",
                             mode="lines+markers",
                             line=dict(color="#e84e1a", dash="dash")))
    fig.update_layout(
        title="Predictions per Window (0=Clean, 1=Spoofed)",
        xaxis_title="Window index",
        yaxis=dict(tickvals=[0, 1], ticktext=["Clean", "Spoofed"]),
        height=300,
        margin=dict(t=50, b=20),
        legend=dict(orientation="h"),
    )
    return fig


# ─────────────────────────────────────────────
# 9. NAVIGATION FILE AUTO-DOWNLOADER
# ─────────────────────────────────────────────

def date_to_doy(year: int, month: int, day: int):
    """Convert calendar date to day-of-year."""
    return datetime(year, month, day).timetuple().tm_yday


def build_nav_urls(year: int, month: int, day: int) -> list:
    """
    Build a list of candidate URLs for the IGS broadcast navigation file
    (BRDC) for the given date. Tries multiple mirrors in order.
    """
    doy   = date_to_doy(year, month, day)
    yr2   = str(year)[2:]   # e.g. "23"
    yr4   = year            # e.g. 2023

    # Legacy RINEX 2 nav filename  e.g. brdc2780.23n.gz
    legacy_name = f"brdc{doy:03d}0.{yr2}n"

    # RINEX 3 mixed nav filename
    rinex3_name = f"BRDC00IGS_R_{yr4}{doy:03d}0000_01D_MN.rnx"

    mirrors = [
        # ── CDDIS NASA (most reliable, needs Earthdata but often open) ──
        f"https://cddis.nasa.gov/archive/gnss/data/daily/{yr4}/{doy:03d}/{yr2}n/{legacy_name}.gz",
        f"https://cddis.nasa.gov/archive/gnss/data/daily/{yr4}/brdc/{legacy_name}.gz",
        # ── BKG Germany ──
        f"https://igs.bkg.bund.de/root_ftp/IGS/BRDC/{yr4}/{doy:03d}/{rinex3_name}.gz",
        # ── IGN France ──
        f"https://igs.ign.fr/pub/igs/data/daily/{yr4}/{doy:03d}/{legacy_name}.gz",
        f"https://igs.ign.fr/pub/igs/data/daily/{yr4}/{doy:03d}/{legacy_name}.Z",
        # ── GFZ Germany ──
        f"https://ftp.gfz-potsdam.de/pub/GNSS/data/daily/{yr4}/{doy:03d}/{legacy_name}.gz",
        # ── EUREF ──
        f"https://igs.bkg.bund.de/root_ftp/EUREF/BRDC/{yr4}/{doy:03d}/{rinex3_name}.gz",
    ]
    return mirrors, legacy_name


def download_nav_file(year: int, month: int, day: int, save_dir: str) -> str:
    """
    Try each mirror in order until one succeeds.
    Decompresses .gz automatically.
    Returns the path to the decompressed nav file, or raises RuntimeError.
    """
    mirrors, legacy_name = build_nav_urls(year, month, day)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) RTKLIB/2.4.3",
        "Accept": "*/*",
    }

    for url in mirrors:
        try:
            resp = requests.get(url, headers=headers, timeout=30, stream=True)
            if resp.status_code == 200:
                # Save compressed file
                gz_path  = os.path.join(save_dir, legacy_name + ".gz")
                nav_path = os.path.join(save_dir, legacy_name)

                with open(gz_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        f.write(chunk)

                # Decompress
                with gzip.open(gz_path, "rb") as f_in, open(nav_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

                os.remove(gz_path)
                return nav_path
        except Exception:
            continue

    raise RuntimeError(
        "Could not download navigation file from any IGS mirror.\n"
        "Please download it manually from:\n"
        "https://cddis.nasa.gov/Data_and_Derived_Products/GNSS/broadcast_ephemeris_data.html\n"
        "Or ask your supervisor for the .23N file."
    )


# ─────────────────────────────────────────────
# 10. RTKLIB HELPERS
# ─────────────────────────────────────────────

RTKLIB_CHANNELS = 8   # ch0 → ch7, matching your training data
SAMPLING_FREQ   = 4000000  # default 4 MHz — adjust if needed

def find_rnx2rtkp() -> str:
    """
    Find rnx2rtkp.exe — checks PATH first, then common Windows install locations.
    Returns the full path or raises FileNotFoundError.
    """
    # 1. Check if it's already on PATH
    import shutil
    found = shutil.which("rnx2rtkp") or shutil.which("rnx2rtkp.exe")
    if found:
        return found

    # 2. Check common Windows locations the user might have extracted to
    common_paths = [
        r"C:\RTKLIB\bin\rnx2rtkp.exe",
        r"C:\RTKLIB-2.4.3b34\bin\rnx2rtkp.exe",
        r"C:\rtklib\bin\rnx2rtkp.exe",
        r"C:\Program Files\RTKLIB\bin\rnx2rtkp.exe",
        r"C:\Users\Public\RTKLIB\bin\rnx2rtkp.exe",
    ]
    for p in common_paths:
        if os.path.isfile(p):
            return p

    raise FileNotFoundError(
        "rnx2rtkp.exe not found. "
        "Please add your RTKLIB bin folder to Windows PATH "
        "or enter the full path below."
    )


def write_rtklib_conf(conf_path: str):
    """Write a minimal RTKLIB config file optimised for feature extraction."""
    conf_content = """\
pos1-posmode     =single
pos1-elmask      =15
pos1-snrmask_ena =off
pos1-snrmask_r   =0,0,0,0,0,0,0,0,0
pos1-dynamics    =off
pos1-tidecorr    =off
pos1-ionoopt     =brdc
pos1-tropopt     =saas
pos1-sateph      =brdc
pos1-exclsats    =
pos1-navsys      =1
out-solformat    =llh
out-outstat      =residual
out-timesys      =gpst
out-timeform     =tow
out-timendec     =3
out-degform      =deg
out-fieldsep     =
out-height       =ellipsoidal
out-geoid        =internal
out-solstatic    =all
out-nmeaintv1    =0
out-nmeaintv2    =0
out-outhead      =on
out-outopt       =on
out-outvel       =off
"""
    with open(conf_path, "w") as f:
        f.write(conf_content)


def run_rnx2rtkp(obs_path: str, nav_path: str, rtklib_exe: str, work_dir: str) -> str:
    """
    Run rnx2rtkp and return the path to the output .pos file.
    Raises RuntimeError if RTKLIB fails.
    """
    conf_path = os.path.join(work_dir, "rtklib.conf")
    out_path  = os.path.join(work_dir, "output.pos")
    write_rtklib_conf(conf_path)

    cmd = [rtklib_exe, "-k", conf_path, "-o", out_path, obs_path, nav_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

    # rnx2rtkp writes results to stderr as well as stdout — check both
    if not os.path.isfile(out_path) or os.path.getsize(out_path) == 0:
        raise RuntimeError(
            f"RTKLIB produced no output.\n"
            f"STDOUT: {result.stdout[:500]}\n"
            f"STDERR: {result.stderr[:500]}"
        )
    return out_path


def parse_pos_file(pos_path: str) -> pd.DataFrame:
    """
    Parse RTKLIB .pos solution file.
    Columns returned: date, time, lat, lon, height, fix_status, num_sats, pdop, sdx, sdy, sdz
    """
    rows = []
    with open(pos_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                row = {
                    "date":       parts[0],
                    "time":       parts[1],
                    "lat":        float(parts[2]),
                    "lon":        float(parts[3]),
                    "height":     float(parts[4]),
                    "fix_status": int(parts[5]),
                    "num_sats":   int(parts[6]),
                    "pdop":       float(parts[7]),
                    "sdx":        float(parts[8])  if len(parts) > 8  else 0.0,
                    "sdy":        float(parts[9])  if len(parts) > 9  else 0.0,
                    "sdz":        float(parts[10]) if len(parts) > 10 else 0.0,
                }
                rows.append(row)
            except (ValueError, IndexError):
                continue

    if not rows:
        raise ValueError("Could not parse any solution epochs from RTKLIB .pos file.")
    return pd.DataFrame(rows)


def pos_to_channel_csv(pos_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert RTKLIB position-solution DataFrame into the 88-column
    channel format (ch0..ch7 × 11 features) that your models expect.

    RTKLIB gives us per-epoch global values (lat, lon, height, pdop, …).
    We spread these across ch0→ch7 with per-channel variation added via
    small satellite-index offsets — a best-effort approximation when the
    raw SDR correlator values are unavailable from RINEX.

    Columns that CANNOT come from RINEX (prompt_i, prompt_q) are set to 0.
    tracking_flag is set to 1 (assumed tracking since epoch is in solution).
    """
    n_rows = len(pos_df)
    records = []

    # Derived epoch-level features
    lat    = pos_df["lat"].values
    lon    = pos_df["lon"].values
    height = pos_df["height"].values
    pdop   = pos_df["pdop"].values
    nsats  = pos_df["num_sats"].values
    fix    = pos_df["fix_status"].values
    sdx    = pos_df["sdx"].values
    sdy    = pos_df["sdy"].values
    sdz    = pos_df["sdz"].values
    t_idx  = np.arange(n_rows, dtype=np.float32)

    # Height / position jumps (strong spoofing indicators)
    h_jump  = np.abs(np.gradient(height))
    lat_jmp = np.abs(np.gradient(lat))
    lon_jmp = np.abs(np.gradient(lon))

    for i in range(n_rows):
        row = {}
        for ch in range(RTKLIB_CHANNELS):
            # PRN: distribute satellites 1..32 across channels round-robin
            prn = (ch + 1) + (i % 4)

            # CN0: approximate from PDOP (higher PDOP → lower CN0)
            # Add small per-channel variation to avoid identical columns
            cn0 = max(20.0, 45.0 - pdop[i] * 3.0 + ch * 0.5)

            # Doppler: use height jump as a proxy for Doppler anomaly
            doppler_coarse = h_jump[i] * 10.0 + ch * 0.1
            doppler_fine   = lat_jmp[i] * 1e4  + lon_jmp[i] * 1e4 + ch * 0.01

            # Carrier phase: use cumulative lat/lon displacement
            carrier_phase  = lat[i] * 1e6 + lon[i] * 1e6 + ch * 1000.0

            row.update({
                f"ch{ch}_channel_id":    ch,
                f"ch{ch}_prn":           prn,
                f"ch{ch}_doppler_coarse": doppler_coarse,
                f"ch{ch}_tracking_flag":  1,            # assumed tracking
                f"ch{ch}_sampling_freq":  SAMPLING_FREQ,
                f"ch{ch}_carrier_phase":  carrier_phase,
                f"ch{ch}_doppler_fine":   doppler_fine,
                f"ch{ch}_cn0":            cn0,
                f"ch{ch}_prompt_i":       0.0,          # not available in RINEX
                f"ch{ch}_prompt_q":       0.0,          # not available in RINEX
                f"ch{ch}_time_index":     t_idx[i],
            })
        records.append(row)

    result = pd.DataFrame(records)

    # Ensure exact column order matching your training data (ch0→ch7, 11 cols each)
    expected_cols = []
    for ch in range(RTKLIB_CHANNELS):
        for feat in ["channel_id","prn","doppler_coarse","tracking_flag",
                     "sampling_freq","carrier_phase","doppler_fine",
                     "cn0","prompt_i","prompt_q","time_index"]:
            expected_cols.append(f"ch{ch}_{feat}")
    result = result[expected_cols]
    return result


# ─────────────────────────────────────────────
# 10. MAIN APP
# ─────────────────────────────────────────────
def main():
    # ── Sidebar ──────────────────────────────────────────────────────
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/GPS_Satellite_NASA_art-iif.jpg/320px-GPS_Satellite_NASA_art-iif.jpg",
                 use_container_width=True)
        st.markdown("## 🛰️ GNSS Spoofing Detector")
        st.markdown("""
        Upload GNSS signal data (CSV, NPY, or RINEX).  
        The app runs **two independent models** and compares results.

        ---
        **🛰️ NEW — RTKLIB Pipeline**  
        Upload raw RINEX files in Section 0.  
        RTKLIB converts them to model-ready features automatically.

        ---
        **Model A — SVC (RBF)**  
        Sklearn Support Vector Classifier  
        Input: 352 window statistics

        **Model B — BiLSTM**  
        PyTorch Bidirectional LSTM + Attention  
        Input: (window, 88) sequences
        
        ---
        **Labels** (if column `label` present):  
        `0` = Clean signal  
        `1` = Spoofed signal
        """)
        st.markdown("---")
        window_override = st.slider(
            "Window size", 10, 60, WINDOW,
            help="Sliding-window length used to segment the signal."
        )

    # ── Header ───────────────────────────────────────────────────────
    st.title("🛰️ GNSS Spoofing Detection — Dual Model Comparison")
    st.markdown("*Supports CSV · NPY · RINEX — automatic preprocessing & side-by-side model comparison*")

    # ── Load models ──────────────────────────────────────────────────
    with st.spinner("Loading models…"):
        try:
            svc, dl_model, device = load_models()
            st.success("✅ Both models loaded successfully.", icon="🤖")
        except FileNotFoundError as exc:
            st.error(f"Model file not found: {exc}\n\n"
                     "Place `best_model.pkl` and `best_dl_model.pkl` "
                     "in the same folder as `app.py`.")
            st.stop()
        except Exception as exc:
            st.error(f"Error loading models: {exc}")
            st.stop()

    # ── SECTION 0 — RTKLIB RINEX Preprocessing ───────────────────────
    st.markdown('<div class="section-header">🛰️ Section 0 — RTKLIB Preprocessing (RINEX → Features)</div>',
                unsafe_allow_html=True)

    with st.expander("📡 Convert RINEX files using RTKLIB (optional — skip if uploading CSV directly)", expanded=False):
        st.markdown("""
        Upload a **RINEX Observation file** and a **RINEX Navigation file**.  
        RTKLIB will process them and generate a feature CSV compatible with your trained models.  
        The converted CSV will be passed automatically to Section 1.
        """)

        rtklib_path_input = st.text_input(
            "RTKLIB rnx2rtkp.exe path (leave blank to auto-detect from PATH)",
            value="",
            placeholder=r"e.g. C:\RTKLIB-2.4.3b34\bin\rnx2rtkp.exe",
        )

        # ── Navigation file auto-downloader ──────────────────────────
        st.markdown("---")
        st.markdown("#### 📥 Don't have a Navigation file? Download it automatically")
        st.markdown("Enter the **date of your Observation file** and we'll download the matching navigation file from IGS:")

        col_y, col_m, col_d = st.columns(3)
        with col_y:
            nav_year  = st.number_input("Year",  min_value=2000, max_value=2030, value=2023, step=1)
        with col_m:
            nav_month = st.number_input("Month", min_value=1,    max_value=12,   value=10,   step=1)
        with col_d:
            nav_day   = st.number_input("Day",   min_value=1,    max_value=31,   value=5,    step=1)

        if st.button("📥 Download Navigation File from IGS", type="secondary"):
            with st.spinner(f"Downloading navigation file for {int(nav_year)}-{int(nav_month):02d}-{int(nav_day):02d}…"):
                try:
                    tmp_nav_dir = tempfile.mkdtemp()
                    nav_dl_path = download_nav_file(
                        int(nav_year), int(nav_month), int(nav_day), tmp_nav_dir
                    )
                    with open(nav_dl_path, "rb") as f:
                        nav_bytes = f.read()

                    st.success(f"✅ Navigation file downloaded! ({len(nav_bytes)/1024:.1f} KB)")
                    st.download_button(
                        "⬇️ Save Navigation file (.nav) to your PC",
                        data=nav_bytes,
                        file_name=os.path.basename(nav_dl_path),
                        mime="application/octet-stream",
                    )
                    # Store in session for use in RTKLIB run
                    st.session_state["downloaded_nav_bytes"] = nav_bytes
                    st.session_state["downloaded_nav_name"]  = os.path.basename(nav_dl_path)
                    st.info("✅ Navigation file is ready — upload your Observation file below and click Run RTKLIB.")

                except RuntimeError as e:
                    st.error(f"❌ {e}")
                except Exception as e:
                    st.error(f"❌ Download failed: {e}")

        st.markdown("---")
        st.markdown("#### 📂 Upload RINEX files")

        st.warning(
            "⚠️ **Important — Rename your files before uploading!**\n\n"
            "Streamlit does not support extensions starting with numbers (`.23o`, `.23n`).\n\n"
            "Please rename your files like this:\n"
            "- `01052270.23o` → rename to → `01052270_obs.rnx`\n"
            "- `01052270.23n` → rename to → `01052270_nav.rnx`\n\n"
            "The file content stays exactly the same — only the extension changes."
        )

        col_obs, col_nav = st.columns(2)
        with col_obs:
            obs_file = st.file_uploader(
                "📄 RINEX Observation file (.rnx / .obs / .txt)",
                type=["rnx", "obs", "rnx3", "txt"],
                key="rtklib_obs",
            )
        with col_nav:
            # Show note if nav was auto-downloaded
            if st.session_state.get("downloaded_nav_bytes"):
                st.success(f"✅ Navigation file ready: `{st.session_state.get('downloaded_nav_name')}`  \nNo need to upload — it was downloaded above.")
                nav_file = None
            else:
                nav_file = st.file_uploader(
                    "📄 RINEX Navigation file (.rnx / .nav / .txt)",
                    type=["nav", "rnx", "rnx3", "txt"],
                    key="rtklib_nav",
                )

        # Determine if we have nav from download or upload
        nav_ready = (
            nav_file is not None or
            st.session_state.get("downloaded_nav_bytes") is not None
        )

        if obs_file and nav_ready:
            if st.button("🚀 Run RTKLIB & Convert to Features", type="primary"):
                with st.spinner("Running RTKLIB — this may take 10–60 seconds for large files…"):
                    try:
                        if rtklib_path_input.strip():
                            exe = rtklib_path_input.strip()
                            if not os.path.isfile(exe):
                                raise FileNotFoundError(f"Not found: {exe}")
                        else:
                            exe = find_rnx2rtkp()

                        st.info(f"✅ Using RTKLIB at: `{exe}`")

                        with tempfile.TemporaryDirectory() as tmp:
                            # Save observation file
                            obs_path = os.path.join(tmp, obs_file.name)
                            with open(obs_path, "wb") as f:
                                f.write(obs_file.read())

                            # Save navigation file (uploaded or downloaded)
                            if nav_file is not None:
                                nav_path = os.path.join(tmp, nav_file.name)
                                with open(nav_path, "wb") as f:
                                    f.write(nav_file.read())
                            else:
                                nav_name = st.session_state["downloaded_nav_name"]
                                nav_path = os.path.join(tmp, nav_name)
                                with open(nav_path, "wb") as f:
                                    f.write(st.session_state["downloaded_nav_bytes"])

                            st.info(f"📁 Obs: `{os.path.basename(obs_path)}`  |  Nav: `{os.path.basename(nav_path)}`")

                            # Run RTKLIB
                            pos_path = run_rnx2rtkp(obs_path, nav_path, exe, tmp)
                            st.success("✅ RTKLIB processing complete!")

                            pos_df = parse_pos_file(pos_path)
                            st.info(f"📊 Parsed {len(pos_df)} solution epochs from RTKLIB")

                            with st.expander("🔍 RTKLIB raw solution (first 20 epochs)"):
                                st.dataframe(pos_df.head(20), use_container_width=True)
                                c1, c2, c3 = st.columns(3)
                                c1.metric("Total epochs", len(pos_df))
                                c2.metric("Avg satellites", f"{pos_df['num_sats'].mean():.1f}")
                                c3.metric("Avg PDOP", f"{pos_df['pdop'].mean():.2f}")

                            channel_df = pos_to_channel_csv(pos_df)
                            st.success(f"✅ Converted → {channel_df.shape[0]} rows × {channel_df.shape[1]} cols")

                            st.session_state["rtklib_csv"] = channel_df

                            st.download_button(
                                "⬇️ Download converted features CSV",
                                data=channel_df.to_csv(index=False).encode(),
                                file_name="rtklib_features.csv",
                                mime="text/csv",
                            )
                            st.info("👇 Scroll down — the converted data has been loaded into Section 1 automatically.")

                    except FileNotFoundError as e:
                        st.error(f"❌ RTKLIB not found: {e}\n\n"
                                 "Please enter the full path to rnx2rtkp.exe in the field above.")
                    except RuntimeError as e:
                        st.error(f"❌ RTKLIB failed:\n{e}")
                    except Exception as e:
                        st.error(f"❌ Unexpected error: {e}")

        elif obs_file and not nav_ready:
            st.warning("⚠️ Navigation file needed — either download it above or upload it manually.")
        elif not obs_file and nav_ready:
            st.warning("⚠️ Please upload your Observation file (.obs / .23O).")
        else:
            st.info("📋 Steps:  1️⃣ Download nav file above (or upload it)  →  2️⃣ Upload your Observation file  →  3️⃣ Click Run RTKLIB")

    # ── SECTION 1 — Upload ───────────────────────────────────────────
    st.markdown('<div class="section-header">📂 Section 1 — Upload Data</div>',
                unsafe_allow_html=True)

    rtklib_result = st.session_state.get("rtklib_csv", None)
    if rtklib_result is not None:
        st.success("✅ Using RTKLIB-converted features from Section 0.")

    uploaded = st.file_uploader(
        "Upload a GNSS data file",
        type=["csv", "npy", "obs", "rnx", "txt"],
        help="Accepted: .csv (preferred), .npy (numpy array), .obs/.rnx/.txt (RINEX obs). "
             "Skip this if you already ran RTKLIB above.",
    )

    if uploaded is None and rtklib_result is None:
        st.info("👆 Either run RTKLIB in Section 0 to convert RINEX files, "
                "or upload a CSV/NPY file here directly.")
        st.stop()

    # ── SECTION 2 — Preprocessing ────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Section 2 — Preprocessing & Format Conversion</div>',
                unsafe_allow_html=True)

    with st.spinner("Parsing and converting file…"):
        try:
            # ── Priority: use RTKLIB output if available ──────────────
            if rtklib_result is not None and uploaded is None:
                df = rtklib_result.copy()
                st.success(f"✅ Using RTKLIB features → {df.shape[0]} rows, {df.shape[1]} cols.")

            elif uploaded is not None:
                raw_bytes = uploaded.read()
                fname     = uploaded.name.lower()

                # ── NPY → DataFrame ──────────────────────────────────
                if fname.endswith(".npy"):
                    arr = np.load(io.BytesIO(raw_bytes), allow_pickle=True)
                    df  = npy_to_dataframe(arr)
                    st.success(f"✅ NPY array {arr.shape} converted to DataFrame.")

                # ── RINEX obs → DataFrame ─────────────────────────────
                elif fname.endswith((".obs", ".rnx")):
                    text = raw_bytes.decode("utf-8", errors="ignore")
                    df   = rinex_to_dataframe(text)
                    st.success(f"✅ RINEX file parsed → {df.shape[0]} rows, {df.shape[1]} cols.")

                # ── CSV / TXT ─────────────────────────────────────────
                else:
                    df = pd.read_csv(io.BytesIO(raw_bytes))
                    st.success(f"✅ CSV loaded → {df.shape[0]} rows, {df.shape[1]} cols.")

            # Clean column names
            df = remove_duplicate_cols(clean_columns(df))

        except Exception as exc:
            st.error(f"❌ Could not parse file: {exc}")
            st.stop()

    # Offer converted CSV download
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button("⬇️ Download converted CSV",
                       data=csv_bytes,
                       file_name="converted_data.csv",
                       mime="text/csv")

    # Show data preview
    with st.expander("🔍 Preview converted data (first 50 rows)", expanded=False):
        st.dataframe(df.head(50), use_container_width=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows",    df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Label col present",
                    "Yes ✅" if "label" in df.columns else "No ⚠️")

    # Separate label column if present
    if "label" in df.columns:
        labels_raw = df["label"].values
        df_feat    = df.drop(columns=["label"])
    else:
        labels_raw = None
        df_feat    = df.copy()

    # Keep only numeric columns
    df_feat = df_feat.select_dtypes(include=[np.number])
    if df_feat.shape[1] == 0:
        st.error("No numeric feature columns found after preprocessing.")
        st.stop()

    # Scale features
    scaler  = StandardScaler()
    scaled  = pd.DataFrame(
        scaler.fit_transform(df_feat),
        columns=df_feat.columns
    )

    # ── SECTION 3 — Prediction ───────────────────────────────────────
    st.markdown('<div class="section-header">🔮 Section 3 — Running Predictions</div>',
                unsafe_allow_html=True)

    win = window_override   # user can adjust from sidebar

    # ── Feature engineering for SVC ──────────────────────────────────
    aligned_ml = align_features(scaled.copy(), DL_INPUT_DIM)   # 88 raw cols
    X_win      = build_window_features(aligned_ml, window=win)

    if X_win.shape[1] != ML_N_FEATURES:
        # Pad / trim to exactly 352
        if X_win.shape[1] < ML_N_FEATURES:
            pad   = np.zeros((X_win.shape[0], ML_N_FEATURES - X_win.shape[1]))
            X_win = np.hstack([X_win, pad])
        else:
            X_win = X_win[:, :ML_N_FEATURES]

    # ── Sequences for BiLSTM ─────────────────────────────────────────
    aligned_dl = align_features(scaled.copy(), DL_INPUT_DIM)
    X_seq      = build_sequences(aligned_dl, window=win)  # (N, win, 88)

    n_windows = min(len(X_win), len(X_seq))
    if n_windows == 0:
        st.error(f"Not enough rows to form even one window of size {win}. "
                 f"Upload a file with at least {win + 1} rows.")
        st.stop()

    X_win = X_win[:n_windows]
    X_seq = X_seq[:n_windows]
    if labels_raw is not None:
        # Align labels to windows
        y_win = np.array([
            int(np.any(labels_raw[i * win : (i + 1) * win]))
            for i in range(n_windows)
        ])
    else:
        y_win = None

    st.info(f"📊 {n_windows} windows of size {win} built from {df_feat.shape[0]} rows.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### 🔵 Model A — SVC (RBF)")
        with st.spinner("Running SVC inference…"):
            preds_svc, proba_svc = predict_svc(svc, X_win)
        spoof_pct_svc = 100 * preds_svc.mean()
        st.metric("Spoofed windows detected",
                  f"{preds_svc.sum()} / {n_windows}",
                  delta=f"{spoof_pct_svc:.1f}%")
        lbl = ("🔴 SPOOFING DETECTED" if spoof_pct_svc > 40
               else "🟢 Signal appears clean")
        st.markdown(f'<span class="{"alert-spoof" if spoof_pct_svc > 40 else "alert-clean"}">'
                    f'{lbl}</span>', unsafe_allow_html=True)

    with col_b:
        st.markdown("#### 🔴 Model B — BiLSTM Attention")
        with st.spinner("Running BiLSTM inference…"):
            preds_dl, proba_dl = predict_bilstm(dl_model, X_seq, device)
        spoof_pct_dl = 100 * preds_dl.mean()
        st.metric("Spoofed windows detected",
                  f"{preds_dl.sum()} / {n_windows}",
                  delta=f"{spoof_pct_dl:.1f}%")
        lbl = ("🔴 SPOOFING DETECTED" if spoof_pct_dl > 40
               else "🟢 Signal appears clean")
        st.markdown(f'<span class="{"alert-spoof" if spoof_pct_dl > 40 else "alert-clean"}">'
                    f'{lbl}</span>', unsafe_allow_html=True)

    # Prediction timeline
    st.plotly_chart(pred_timeline(preds_svc, preds_dl),
                    use_container_width=True)

    # ── SECTION 4 — Comparison ───────────────────────────────────────
    st.markdown('<div class="section-header">📊 Section 4 — Model Comparison</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Metrics", "🗺️ Confusion Matrices", "📉 Distributions", "📋 Raw Predictions"]
    )

    # ── Tab 1 — Metrics ──────────────────────────────────────────────
    with tab1:
        if y_win is not None and len(np.unique(y_win)) > 1:
            m_svc = compute_metrics(y_win, preds_svc, proba_svc, "SVC")
            m_dl  = compute_metrics(y_win, preds_dl,  proba_dl,  "BiLSTM")

            # Metrics table
            mdf = pd.DataFrame([m_svc, m_dl]).set_index("model")
            mdf_display = (mdf * 100).round(2).astype(str) + " %"
            st.dataframe(mdf_display, use_container_width=True)

            # Radar
            st.plotly_chart(metric_radar([m_svc, m_dl]),
                            use_container_width=True)

            # Bar chart
            fig_bar = go.Figure()
            metric_names = ["accuracy", "f1", "precision", "recall", "auc_roc"]
            for m, color in zip([m_svc, m_dl], ["#1a73e8", "#e84e1a"]):
                fig_bar.add_trace(go.Bar(
                    name=m["model"],
                    x=[mn.replace("_", " ").title() for mn in metric_names],
                    y=[m[mn] for mn in metric_names],
                    marker_color=color,
                    text=[f"{m[mn]*100:.1f}%" for mn in metric_names],
                    textposition="outside",
                ))
            fig_bar.update_layout(
                barmode="group", title="Side-by-Side Metric Comparison",
                yaxis=dict(range=[0, 1.15]), height=380,
                margin=dict(t=50, b=20),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ── Winner ───────────────────────────────────────────────
            st.markdown("---")
            if m_svc["f1"] > m_dl["f1"]:
                winner = f"🏆 SVC (F1 = {m_svc['f1']*100:.1f}% vs {m_dl['f1']*100:.1f}%)"
            elif m_dl["f1"] > m_svc["f1"]:
                winner = f"🏆 BiLSTM Attention (F1 = {m_dl['f1']*100:.1f}% vs {m_svc['f1']*100:.1f}%)"
            else:
                winner = "🤝 Tie — both models perform equally (F1)"
            st.markdown(f'<div class="winner-banner">{winner} performs better on this dataset</div>',
                        unsafe_allow_html=True)

            # Detailed classification reports
            with st.expander("📄 Detailed Classification Reports"):
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**SVC Report**")
                    st.text(classification_report(y_win, preds_svc,
                                                  target_names=["Clean", "Spoofed"],
                                                  zero_division=0))
                with c2:
                    st.markdown("**BiLSTM Report**")
                    st.text(classification_report(y_win, preds_dl,
                                                  target_names=["Clean", "Spoofed"],
                                                  zero_division=0))
        else:
            st.warning(
                "No `label` column found (or only one class present). "
                "Accuracy metrics require ground-truth labels. "
                "The prediction timeline and distribution charts are still available."
            )

    # ── Tab 2 — Confusion matrices ───────────────────────────────────
    with tab2:
        if y_win is not None and len(np.unique(y_win)) > 1:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(confusion_heatmap(y_win, preds_svc, "SVC Confusion Matrix"),
                                use_container_width=True)
            with c2:
                st.plotly_chart(confusion_heatmap(y_win, preds_dl, "BiLSTM Confusion Matrix"),
                                use_container_width=True)
        else:
            st.info("Confusion matrices require ground-truth labels in a `label` column.")

    # ── Tab 3 — Probability distributions ───────────────────────────
    with tab3:
        st.plotly_chart(proba_histogram(proba_svc, proba_dl),
                        use_container_width=True)

        # Avg probability per window (line chart)
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(y=proba_svc, name="SVC P(Spoofed)",
                                   line=dict(color="#1a73e8")))
        fig_p.add_trace(go.Scatter(y=proba_dl,  name="BiLSTM P(Spoofed)",
                                   line=dict(color="#e84e1a", dash="dash")))
        fig_p.update_layout(title="Spoofing Probability per Window",
                            xaxis_title="Window index",
                            yaxis_title="P(Spoofed)",
                            height=320,
                            margin=dict(t=50, b=20),
                            legend=dict(orientation="h"))
        st.plotly_chart(fig_p, use_container_width=True)

    # ── Tab 4 — Raw predictions table ────────────────────────────────
    with tab4:
        result_df = pd.DataFrame({
            "Window":         range(n_windows),
            "SVC Prediction": ["Spoofed" if p else "Clean" for p in preds_svc],
            "SVC P(Spoofed)": proba_svc.round(4),
            "BiLSTM Prediction": ["Spoofed" if p else "Clean" for p in preds_dl],
            "BiLSTM P(Spoofed)": proba_dl.round(4),
        })
        if y_win is not None:
            result_df.insert(1, "True Label",
                             ["Spoofed" if y else "Clean" for y in y_win])

        st.dataframe(result_df, use_container_width=True, height=400)
        st.download_button(
            "⬇️ Download predictions CSV",
            data=result_df.to_csv(index=False).encode(),
            file_name="predictions.csv",
            mime="text/csv",
        )

    # ── Footer ───────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "GNSS Spoofing Detection App · SVC (RBF) vs BiLSTM-Attention · "
        "Built with Streamlit · Model inputs: 352-dim stats (SVC) | (win × 88) sequences (BiLSTM)"
    )


if __name__ == "__main__":
    main()
