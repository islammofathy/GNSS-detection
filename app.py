"""
GNSS Spoofing Detection — Dual Model Comparison App
=====================================================
Compares:
  • best_model  : sklearn SVC (RBF), expects 352 window-stat features
  • best_dl_model : PyTorch BiLSTM_Attention, expects (batch, seq_len, 88) sequences
"""

import re
import io
import warnings
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
# 9. MAIN APP
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

    # ── SECTION 1 — Upload ───────────────────────────────────────────
    st.markdown('<div class="section-header">📂 Section 1 — Upload Data</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload a GNSS data file",
        type=["csv", "npy", "obs", "rnx", "txt"],
        help="Accepted: .csv (preferred), .npy (numpy array), .obs/.rnx/.txt (RINEX obs).",
    )

    if uploaded is None:
        st.info("👆 Upload a file to begin. "
                "A CSV with the same columns used during training gives the best results.")
        st.stop()

    # ── SECTION 2 — Preprocessing ────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Section 2 — Preprocessing & Format Conversion</div>',
                unsafe_allow_html=True)

    raw_bytes = uploaded.read()
    fname     = uploaded.name.lower()

    with st.spinner("Parsing and converting file…"):
        try:
            # ── NPY → DataFrame ──────────────────────────────────────
            if fname.endswith(".npy"):
                arr = np.load(io.BytesIO(raw_bytes), allow_pickle=True)
                df  = npy_to_dataframe(arr)
                st.success(f"✅ NPY array {arr.shape} converted to DataFrame.")

            # ── RINEX obs → DataFrame ─────────────────────────────────
            elif fname.endswith((".obs", ".rnx")):
                text = raw_bytes.decode("utf-8", errors="ignore")
                df   = rinex_to_dataframe(text)
                st.success(f"✅ RINEX file parsed → {df.shape[0]} rows, {df.shape[1]} cols.")

            # ── CSV / TXT ─────────────────────────────────────────────
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
