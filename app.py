from collections import Counter
import os
import pickle
import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from wordcloud import WordCloud

from model_arch import IndoBERTBiLSTMAttention

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ══════════════════════════════════════════════════════════════
# KONFIGURASI HALAMAN
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Analisis Sentimen KDMP",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════
# GENESIS DESIGN SYSTEM — CSS STYLING
# ══════════════════════════════════════════════════════════════
GENESIS_COLORS = {
    "primary": "#6366F1",
    "primary_hover": "#4F46E5",
    "secondary": "#20970B",
    "neutral": "#9C9C9C",
    "bg": "#0F0F12",
    "surface": "#17171A",
    "text_primary": "#F2F2F3",
    "text_secondary": "#A3A3A8",
    "border": "#2E2E33",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
}

COLORS = {
    "Positif": "#10B981",
    "Netral": "#9C9C9C",
    "Negatif": "#EF4444",
}

st.markdown("""
<style>
    @import url('https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600,700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --primary: #6366F1;
        --primary-hover: #4F46E5;
        --secondary: #20970B;
        --neutral: #9C9C9C;
        --bg: #0F0F12;
        --surface: #17171A;
        --text-primary: #F2F2F3;
        --text-secondary: #A3A3A8;
        --border: #2E2E33;
        --success: #10B981;
        --warning: #F59E0B;
        --error: #EF4444;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--text-primary);
    }

    .stApp, .main { background-color: var(--bg) !important; }
    .block-container { padding-top: 2rem; max-width: 1280px; }

    h1, h2, h3, h4, h5, h6,
    .hero-title, .metric-value {
        font-family: 'General Sans', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em;
        color: var(--text-primary) !important;
    }
    p, span, div, label { color: var(--text-primary); }

    code, pre, .stCodeBlock, .prob-label { font-family: 'JetBrains Mono', monospace !important; }

    section[data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
    section[data-testid="stSidebar"] .stRadio label { color: var(--text-secondary) !important; }

    .hero-header {
        background: var(--surface);
        border: 1px solid var(--border);
        padding: 32px 28px;
        border-radius: 12px;
        text-align: left;
        margin-bottom: 32px;
    }
    .hero-title {
        font-size: 2rem; font-weight: 700; color: var(--text-primary); margin: 0;
        letter-spacing: -0.03em;
    }
    .hero-subtitle {
        font-size: 0.95rem; color: var(--text-secondary); margin-top: 8px;
        font-family: 'DM Sans', sans-serif; line-height: 1.5;
    }

    .metric-card {
        background: var(--surface);
        border-radius: 12px;
        padding: 20px;
        text-align: left;
        border: 1px solid var(--border);
        margin: 5px 0;
        transition: box-shadow 200ms ease, transform 200ms ease;
    }
    .metric-card:hover {
        box-shadow: 0 8px 30px rgba(0,0,0,0.5), 0 0 0 1px rgba(99,102,241,0.15);
        transform: translateY(-2px);
    }
    .metric-value {
        font-size: clamp(1.1rem, 5vw, 1.8rem);
        font-weight: 700;
        margin: 4px 0;
        font-family: 'General Sans', sans-serif;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .metric-label {
        font-size: 0.72rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-weight: 500;
    }

    .badge-Positif, .badge-Negatif, .badge-Netral {
        padding: 6px 18px; border-radius: 9999px;
        font-weight: 600; font-size: 0.95rem; display: inline-block;
        font-family: 'General Sans', sans-serif;
    }
    .badge-Positif { background: rgba(16,185,129,0.18); color: var(--success); }
    .badge-Negatif { background: rgba(239,68,68,0.18); color: var(--error); }
    .badge-Netral  { background: rgba(156,156,156,0.22); color: #C4C4C9; }

    .prob-row { display: flex; align-items: center; margin: 8px 0; gap: 10px; }
    .prob-label { width: 80px; color: var(--text-secondary); font-size: 0.8rem; }
    .prob-bar-bg {
        flex: 1; background: #232327; border-radius: 8px; height: 20px; overflow: hidden;
        border: 1px solid var(--border);
    }
    .prob-bar-fill {
        height: 100%; border-radius: 8px;
        display: flex; align-items: center; padding-left: 8px;
        font-size: 0.72rem; font-weight: 600; color: white;
        transition: width 0.5s ease;
    }

    .info-box {
        background: var(--surface); border-left: 3px solid var(--primary);
        border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); border-right: 1px solid var(--border);
        padding: 12px 16px; border-radius: 0 8px 8px 0;
        margin: 10px 0;
    }

    .stButton > button {
        font-family: 'DM Sans', sans-serif;
        border-radius: 6px !important;
        font-weight: 500;
        transition: transform 120ms ease, box-shadow 120ms ease;
        border: 1px solid var(--border);
    }
    .stButton > button:hover { transform: translateY(-1px); }
    .stButton > button[kind="primary"] {
        background-color: var(--primary) !important;
        border: 1px solid var(--primary) !important;
        color: white !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: var(--primary-hover) !important;
        box-shadow: 0 4px 12px rgba(99,102,241,0.35);
    }
    .stButton > button[kind="secondary"] {
        background-color: transparent !important;
        color: var(--text-primary) !important;
    }

    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        border-radius: 6px !important;
        border: 1px solid var(--border) !important;
        background-color: var(--surface) !important;
        font-family: 'DM Sans', sans-serif;
        color: var(--text-primary) !important;
    }
    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
    }

    [data-baseweb="select"] > div {
        border-radius: 6px !important;
        border-color: var(--border) !important;
        background-color: var(--surface) !important;
    }
    [data-baseweb="tag"] {
        background-color: var(--primary) !important;
        border-radius: 9999px !important;
        color: white !important;
    }
    [data-baseweb="tag"] span { color: white !important; }

    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif; font-weight: 500;
        color: var(--text-secondary);
    }
    .stTabs [aria-selected="true"] { color: var(--primary) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background-color: var(--primary) !important; }

    [data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 8px; }
    hr { border-color: var(--border) !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# KONSTANTA & PREPROCESSING
# ══════════════════════════════════════════════════════════════
BERT_MODEL = "indobenchmark/indobert-base-p1"
MAX_LEN = 64
INV_LABEL_MAP = {0: "Negatif", 1: "Netral", 2: "Positif"}

SLANG_DICT = {
    "gak": "tidak", "ga": "tidak", "ngga": "tidak", "nggak": "tidak", "gk": "tidak",
    "tdk": "tidak", "tak": "tidak", "blm": "belum", "blum": "belum",
    "udah": "sudah", "udh": "sudah", "dah": "sudah",
    "bgt": "banget", "bngt": "banget", "yg": "yang", "dgn": "dengan",
    "utk": "untuk", "krn": "karena", "sdh": "sudah", "jd": "jadi",
    "tp": "tapi", "klo": "kalau", "kl": "kalau", "kalo": "kalau",
    "ttg": "tentang", "sm": "sama", "bs": "bisa", "msh": "masih",
    "trs": "terus", "jg": "juga", "lg": "lagi", "lbh": "lebih",
    "pd": "pada", "dr": "dari", "dlm": "dalam", "dg": "dengan",
    "kdmp": "koperasi desa merah putih",
    "wkwk": "", "haha": "", "hihi": "", "nih": "",
    "deh": "", "dong": "", "sih": "", "aja": "saja",
}


def normalize_sentiment_label(val):
    s = str(val).strip().lower()
    if s in {"positif", "positive", "pos", "2"}:
        return "Positif"
    if s in {"negatif", "negative", "neg", "0"}:
        return "Negatif"
    return "Netral"


def normalize_slang(text):
    return " ".join([SLANG_DICT.get(w, w) for w in text.split()])


def clean_for_bert(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    cleaned = normalize_slang(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned[:300] if cleaned else "tidak ada teks"


# ══════════════════════════════════════════════════════════════
# LOAD MODEL & DATA (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists("model_config.pkl"):
        with open("model_config.pkl", "rb") as f:
            cfg = pickle.load(f)
    else:
        cfg = {
            "pretrained_model": BERT_MODEL,
            "lstm_hidden": 128,
            "num_classes": 3,
            "dropout_rate": 0.3,
            "max_len": MAX_LEN,
            "inv_label_map": INV_LABEL_MAP,
        }

    tok_path = "tokenizer" if os.path.exists("tokenizer") else cfg.get("pretrained_model", BERT_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(tok_path)

    model = IndoBERTBiLSTMAttention(
        pretrained_path=tok_path,
        num_classes=cfg.get("num_classes", 3),
        lstm_hidden=cfg.get("lstm_hidden", 128),
        dropout_rate=cfg.get("dropout_rate", 0.3),
        freeze_bert=False,
    ).to(device)

    if os.path.exists("best_model.pt"):
        state = torch.load("best_model.pt", map_location=device)
        model.load_state_dict(state)

    model.eval()
    return model, tokenizer, device, cfg


@st.cache_data
def load_data():
    src = "df_labeled.csv" if os.path.exists("df_labeled.csv") else ("df_original.csv" if os.path.exists("df_original.csv") else None)
    if src is None:
        return None
    df = pd.read_csv(src)
    if "tweet" not in df.columns and "full_text" in df.columns:
        df["tweet"] = df["full_text"]
    if "clean_tweet" not in df.columns and "clean_text" in df.columns:
        df["clean_tweet"] = df["clean_text"]
    if "sentiment" in df.columns:
        df["sentiment"] = df["sentiment"].apply(normalize_sentiment_label)
    return df


@st.cache_data
def load_eval():
    if os.path.exists("eval_results.pkl"):
        with open("eval_results.pkl", "rb") as f:
            data = pickle.load(f)
        # Normalisasi kunci agar kompatibel dengan train_model.py maupun format alternatif
        if "f1" not in data and "f1_score" in data:
            data["f1"] = data["f1_score"]
        if "cm" not in data and "confusion_matrix" in data:
            data["cm"] = data["confusion_matrix"]
        if "report" not in data and "classification_report" in data:
            data["report"] = data["classification_report"]
        return data
    return None


def predict_sentiment(text, model, tokenizer, device, cfg):
    cleaned = clean_for_bert(text)
    max_len = cfg.get("max_len", MAX_LEN)
    inv_map = cfg.get("inv_label_map", INV_LABEL_MAP)

    enc = tokenizer(
        cleaned,
        return_tensors="pt",
        truncation=True,
        max_length=max_len,
        padding="max_length",
    )
    with torch.no_grad():
        logits = model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = normalize_sentiment_label(inv_map.get(pred_idx, "Netral"))
    return pred_label, probs, cleaned


# ══════════════════════════════════════════════════════════════
# SIDEBAR NAVIGASI
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 16px;'>
        <div style='font-size:2.2rem;'>🏛️</div>
        <div style='font-size:1.15rem; font-weight:700; color:#F2F2F3; margin-top:4px; letter-spacing:-0.02em;'>Sentimen KDMP</div>
        <div style='font-size:0.72rem; color:#A3A3A8; margin-top:2px;'>IndoBERT + BiLSTM + Attention</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    NAV_ITEMS = [
        "🏠 Dashboard",
        "🔮 Prediksi Sentimen",
        "📊 Evaluasi Model",
        "☁️ WordCloud",
        "📁 Data Explorer",
    ]

    if "page" not in st.session_state:
        st.session_state.page = NAV_ITEMS[0]

    st.markdown("""
    <style>
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        text-align: left !important;
        justify-content: flex-start !important;
        background-color: transparent !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 9px 14px !important;
        font-size: 0.92rem !important;
        font-weight: 500 !important;
        color: #A3A3A8 !important;
        margin-bottom: 2px !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background-color: rgba(99,102,241,0.10) !important;
        color: #F2F2F3 !important;
        transform: none !important;
    }
    section[data-testid="stSidebar"] .stButton > button p {
        text-align: left !important;
        font-size: 0.92rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

    for item in NAV_ITEMS:
        is_active = st.session_state.page == item
        if is_active:
            st.markdown(f"""
            <div style='background:rgba(99,102,241,0.16); border-radius:8px; padding:9px 14px;
                        margin-bottom:2px; font-size:0.92rem; font-weight:600; color:#818CF8;'>
                {item}
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button(item, key=f"nav_{item}", use_container_width=True):
                st.session_state.page = item
                st.rerun()

    page = st.session_state.page

    st.divider()
    st.markdown("""
    <div class='info-box'>
        <div style='font-size:0.8rem; color:#A3A3A8;'>
        <b style='color:#F2F2F3;'>Arsitektur Model</b><br>
        IndoBERT → BiLSTM → Attention → Softmax
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# LOAD RESOURCES
# ══════════════════════════════════════════════════════════════
model_loaded = False
model, tokenizer, device, cfg = None, None, torch.device("cpu"), {}
try:
    with st.spinner("🔄 Loading model IndoBERT + BiLSTM + Attention..."):
        model, tokenizer, device, cfg = load_model()
    model_loaded = os.path.exists("best_model.pt")
except Exception as e:
    st.sidebar.error(f"⚠️ Model belum tersedia\n\n{str(e)[:120]}")

df = load_data()
eval_res = load_eval()

# ══════════════════════════════════════════════════════════════
# PAGE 1: DASHBOARD
# ══════════════════════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>🏛️ Analisis Sentimen Program KDMP</div>
        <div class='hero-subtitle'>
            Perbandingan Sentimen Masyarakat terhadap Koperasi Desa Merah Putih<br>
            menggunakan Deep Learning IndoBERT + BiLSTM + Attention pada Data X/Twitter
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df is not None:
        total = len(df)
        counts = df["sentiment"].value_counts() if "sentiment" in df.columns else {}
        pos = int(counts.get("Positif", 0))
        net = int(counts.get("Netral", 0))
        neg = int(counts.get("Negatif", 0))

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Total Tweet</div>
                <div class='metric-value' style='color:#6366F1;'>{total:,}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>😊 Positif</div>
                <div class='metric-value' style='color:#10B981;'>{pos:,}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>😐 Netral</div>
                <div class='metric-value' style='color:#9C9C9C;'>{net:,}</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>😠 Negatif</div>
                <div class='metric-value' style='color:#EF4444;'>{neg:,}</div>
            </div>""", unsafe_allow_html=True)

        st.divider()

        col1, col2 = st.columns([3, 2])
        labels_list = ["Positif", "Netral", "Negatif"]
        values_list = [pos, net, neg]
        bar_colors = [COLORS[l] for l in labels_list]

        with col1:
            st.subheader("📊 Distribusi Label Sentimen")
            fig, ax = plt.subplots(figsize=(8, 4), facecolor="#17171A")
            ax.set_facecolor("#17171A")
            bars = ax.bar(
                labels_list, values_list,
                color=bar_colors, edgecolor="#17171A", linewidth=2, width=0.5
            )
            for bar, v in zip(bars, values_list):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    v + max(total * 0.005, 1),
                    f"{v:,}",
                    ha="center", fontweight="bold", color="#F2F2F3", fontsize=11,
                )
            ax.set_ylabel("Jumlah Tweet", color="#A3A3A8")
            ax.tick_params(colors="#F2F2F3")
            for spine in ax.spines.values():
                spine.set_color("#2E2E33")
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with col2:
            st.subheader("🥧 Proporsi Sentimen")
            fig2, ax2 = plt.subplots(figsize=(5, 4), facecolor="#17171A")
            ax2.set_facecolor("#17171A")
            if sum(values_list) > 0:
                wedges, texts, autotexts = ax2.pie(
                    values_list,
                    labels=labels_list,
                    colors=bar_colors,
                    autopct="%1.1f%%",
                    startangle=90,
                    pctdistance=0.75,
                    wedgeprops={"linewidth": 2, "edgecolor": "#17171A"},
                )
                for t in texts:
                    t.set_color("#F2F2F3")
                for at in autotexts:
                    at.set_color("white")
                    at.set_fontweight("bold")
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        if eval_res:
            st.divider()
            st.subheader("🎯 Performa Model")
            m1, m2, m3, m4 = st.columns(4)
            metrics = [
                ("Accuracy", eval_res.get("accuracy", 0), "#6366F1"),
                ("Precision", eval_res.get("precision", 0), "#10B981"),
                ("Recall", eval_res.get("recall", 0), "#F59E0B"),
                ("F1-Score", eval_res.get("f1", 0), "#EF4444"),
            ]
            for col, (name, val, color) in zip([m1, m2, m3, m4], metrics):
                with col:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{name}</div>
                        <div class='metric-value' style='color:{color};'>{val*100:.2f}%</div>
                    </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("💬 Contoh Tweet per Sentimen")
        tab1, tab2, tab3 = st.tabs(["😊 Positif", "😐 Netral", "😠 Negatif"])
        for tab, sent in zip([tab1, tab2, tab3], ["Positif", "Netral", "Negatif"]):
            with tab:
                subset = df[df["sentiment"] == sent].head(5) if "sentiment" in df.columns else pd.DataFrame()
                if not subset.empty:
                    for _, row in subset.iterrows():
                        tweet = str(row.get("tweet", ""))[:200]
                        loc = row.get("user_location", "")
                        meta_txt = f"lokasi: {loc}" if pd.notna(loc) and str(loc).strip() else ""
                        st.markdown(f"""
                        <div style='background:#17171A; border:1px solid #2E2E33; border-left:3px solid {COLORS[sent]};
                             padding:10px 14px; margin:6px 0; border-radius:0 8px 8px 0;'>
                            <span style='color:#F2F2F3; font-size:0.9rem;'>{tweet}</span>
                            <span style='color:#A3A3A8; font-size:0.75rem; float:right;'>{meta_txt}</span>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.info(f"Belum ada contoh data berlabel {sent}.")
    else:
        st.info("📂 Jalankan `scrape_large_data.py` dan `preprocess.py` untuk melihat statistik dataset KDMP.")

# ══════════════════════════════════════════════════════════════
# PAGE 2: PREDIKSI SENTIMEN
# ══════════════════════════════════════════════════════════════
elif page == "🔮 Prediksi Sentimen":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>🔮 Prediksi Sentimen Real-Time</div>
        <div class='hero-subtitle'>Masukkan teks opini untuk dianalisis menggunakan IndoBERT + BiLSTM + Attention</div>
    </div>
    """, unsafe_allow_html=True)

    if not model_loaded or model is None or tokenizer is None:
        st.error("❌ Model belum berhasil dimuat. Pastikan file `best_model.pt` dan folder `tokenizer/` tersedia.")
        st.stop()

    st.markdown("**💡 Atau coba contoh kalimat:**")
    examples = [
        "Program Koperasi Desa Merah Putih sangat membantu warga desa",
        "Pengelolaan KDMP lambat dan banyak kendala di lapangan",
        "Sosialisasi pembentukan koperasi desa hari ini berjalan",
        "Alhamdulillah koperasi desa bikin distribusi makin lancar dan transparan",
        "Anggaran koperasi desa rawan mangkrak kalau tidak diawasi",
    ]
    if "tweet_input" not in st.session_state:
        st.session_state.tweet_input = ""

    cols = st.columns(len(examples))
    for idx, (col, ex) in enumerate(zip(cols, examples)):
        with col:
            if st.button(ex[:32] + "...", use_container_width=True, key=f"ex_{idx}"):
                st.session_state.tweet_input = ex
                st.rerun()

    user_input = st.text_area(
        "✍️ Masukkan teks tweet / opini:",
        height=120,
        placeholder="Ketik atau tempel opini seputar KDMP di sini...",
        key="tweet_input",
    )

    def clear_tweet_input():
        st.session_state.tweet_input = ""

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        analyze = st.button("🔍 Analisis", type="primary", use_container_width=True)
    with col_btn2:
        st.button(
            "🗑️ Hapus",
            use_container_width=True,
            on_click=clear_tweet_input,
        )

    if analyze and user_input.strip():
        with st.spinner("🧠 Model sedang menganalisis..."):
            pred, probs, cleaned = predict_sentiment(user_input, model, tokenizer, device, cfg)

        st.divider()
        emoji = {"Positif": "✅", "Netral": "😐", "Negatif": "❌"}[pred]
        badge_class = f"badge-{pred}"
        conf_pct = float(probs.max()) * 100

        # Urutan probabilitas mengikuti INV_LABEL_MAP {0: Negatif, 1: Netral, 2: Positif}
        prob_by_label = {
            "Positif": float(probs[2]) * 100 if len(probs) > 2 else 0.0,
            "Netral": float(probs[1]) * 100 if len(probs) > 1 else 0.0,
            "Negatif": float(probs[0]) * 100 if len(probs) > 0 else 0.0,
        }

        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.markdown(f"""
            <div style='background:#17171A; border:1px solid #2E2E33; border-radius:12px; padding:25px; text-align:center;'>
                <div style='font-size:4rem;'>{emoji}</div>
                <div class='{badge_class}' style='margin-top:10px;'>{pred.upper()}</div>
                <div style='color:#A3A3A8; margin-top:10px; font-size:0.85rem;'>
                    Keyakinan model: <b style='color:#F2F2F3;'>{conf_pct:.1f}%</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            st.markdown("**📊 Probabilitas per Kelas:**")
            for label, color in [("Positif", "#10B981"), ("Netral", "#9C9C9C"), ("Negatif", "#EF4444")]:
                pct = prob_by_label[label]
                st.markdown(f"""
                <div class='prob-row'>
                    <div class='prob-label'>{label}</div>
                    <div class='prob-bar-bg'>
                        <div class='prob-bar-fill' style='width:{pct:.1f}%; background:{color};'>
                            {pct:.1f}%
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class='info-box' style='margin-top:15px;'>
                <span style='color:#A3A3A8; font-size:0.8rem;'>Teks setelah preprocessing:</span><br>
                <span style='color:#F2F2F3; font-size:0.85rem;'><i>{cleaned[:150]}</i></span>
            </div>
            """, unsafe_allow_html=True)

    elif analyze:
        st.warning("⚠️ Masukkan teks terlebih dahulu!")

    st.divider()
    st.subheader("📋 Analisis Multi-Tweet (Batch)")
    batch_input = st.text_area(
        "Masukkan beberapa tweet (pisahkan dengan baris baru):",
        height=150,
        placeholder="Tweet 1\nTweet 2\nTweet 3\n...",
    )
    if st.button("🔍 Analisis Semua", type="secondary"):
        tweets = [t.strip() for t in batch_input.split("\n") if t.strip()]
        if tweets:
            results = []
            prog = st.progress(0)
            for i, tw in enumerate(tweets):
                p, prb, _ = predict_sentiment(tw, model, tokenizer, device, cfg)
                results.append({
                    "Tweet": tw[:80] + "..." if len(tw) > 80 else tw,
                    "Sentimen": p,
                    "Confidence": f"{float(prb.max())*100:.1f}%",
                    "Positif": f"{float(prb[2])*100:.1f}%",
                    "Netral": f"{float(prb[1])*100:.1f}%",
                    "Negatif": f"{float(prb[0])*100:.1f}%",
                })
                prog.progress((i + 1) / len(tweets))
            df_res = pd.DataFrame(results)
            st.dataframe(df_res, use_container_width=True)
            st.download_button(
                "⬇️ Download Hasil CSV",
                df_res.to_csv(index=False).encode("utf-8"),
                "hasil_prediksi_kdmp.csv",
                "text/csv",
            )
        else:
            st.warning("⚠️ Masukkan minimal satu baris teks untuk analisis batch.")

# ══════════════════════════════════════════════════════════════
# PAGE 3: EVALUASI MODEL
# ══════════════════════════════════════════════════════════════
elif page == "📊 Evaluasi Model":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>📊 Evaluasi Model IndoBERT + BiLSTM + Attention</div>
        <div class='hero-subtitle'>Hasil performa model dari <code>eval_results.pkl</code></div>
    </div>
    """, unsafe_allow_html=True)

    if eval_res is None:
        st.info("📂 File `eval_results.pkl` belum tersedia. Jalankan `python train_model.py` terlebih dahulu.")
        st.stop()

    st.subheader("🎯 Metrik Evaluasi")
    m1, m2, m3, m4 = st.columns(4)
    metrics_data = [
        ("Accuracy", eval_res.get("accuracy", 0), "#6366F1"),
        ("Precision", eval_res.get("precision", 0), "#10B981"),
        ("Recall", eval_res.get("recall", 0), "#F59E0B"),
        ("F1-Score", eval_res.get("f1", 0), "#EF4444"),
    ]
    for col, (name, val, color) in zip([m1, m2, m3, m4], metrics_data):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{name}</div>
                <div class='metric-value' style='color:{color};'>{val*100:.2f}%</div>
            </div>""", unsafe_allow_html=True)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔲 Confusion Matrix")
        cm_data = eval_res.get("cm", None)
        if cm_data is not None:
            cm_array = np.array(cm_data)
            class_names = ["Negatif", "Netral", "Positif"]
            fig, ax = plt.subplots(figsize=(6, 5), facecolor="#17171A")
            ax.set_facecolor("#17171A")
            sns.heatmap(
                cm_array,
                annot=True,
                fmt="d",
                ax=ax,
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues",
                linewidths=0.5,
                linecolor="#2E2E33",
                annot_kws={"size": 14, "weight": "bold", "color": "white"},
            )
            ax.set_title("Confusion Matrix", color="#F2F2F3", fontweight="bold", pad=10)
            ax.set_xlabel("Prediksi", color="#A3A3A8")
            ax.set_ylabel("Aktual", color="#A3A3A8")
            ax.tick_params(colors="#F2F2F3")
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with col2:
        st.subheader("📈 Training History")
        history = eval_res.get("history", None)
        if history and "train_loss" in history:
            epochs_ran = range(1, len(history["train_loss"]) + 1)
            fig2, axes = plt.subplots(2, 1, figsize=(6, 5), facecolor="#17171A")
            for ax in axes:
                ax.set_facecolor("#17171A")

            axes[0].plot(epochs_ran, history["train_loss"], color="#6366F1", marker="o", label="Train", lw=2, ms=5)
            if "val_loss" in history:
                axes[0].plot(epochs_ran, history["val_loss"], color="#F59E0B", marker="o", label="Val", lw=2, ms=5)
            axes[0].set_title("Loss", color="#F2F2F3", fontsize=10)
            axes[0].legend(labelcolor="#F2F2F3", facecolor="#17171A", edgecolor="#2E2E33")
            axes[0].tick_params(colors="#F2F2F3")
            axes[0].grid(alpha=0.2)
            for sp in axes[0].spines.values():
                sp.set_color("#2E2E33")

            if "train_acc" in history and "val_acc" in history:
                axes[1].plot(epochs_ran, [a * 100 for a in history["train_acc"]], color="#6366F1", marker="o", label="Train", lw=2, ms=5)
                axes[1].plot(epochs_ran, [a * 100 for a in history["val_acc"]], color="#F59E0B", marker="o", label="Val", lw=2, ms=5)
                axes[1].set_title("Accuracy (%)", color="#F2F2F3", fontsize=10)
            elif "val_acc" in history:
                axes[1].plot(epochs_ran, [a * 100 for a in history["val_acc"]], color="#10B981", marker="o", label="Val Acc", lw=2, ms=5)
                axes[1].set_title("Validation Accuracy (%)", color="#F2F2F3", fontsize=10)

            axes[1].legend(labelcolor="#F2F2F3", facecolor="#17171A", edgecolor="#2E2E33")
            axes[1].tick_params(colors="#F2F2F3")
            axes[1].grid(alpha=0.2)
            for sp in axes[1].spines.values():
                sp.set_color("#2E2E33")

            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)
            plt.close()
        else:
            st.info("Training history tidak tersedia.")

    st.divider()
    st.subheader("📋 Classification Report Detail")
    report = eval_res.get("report", None)
    if report:
        rows = []
        for cls in ["Negatif", "Netral", "Positif"]:
            if cls in report:
                r = report[cls]
                rows.append({
                    "Kelas": cls,
                    "Precision": f"{r.get('precision', 0)*100:.2f}%",
                    "Recall": f"{r.get('recall', 0)*100:.2f}%",
                    "F1-Score": f"{r.get('f1-score', 0)*100:.2f}%",
                    "Support": int(r.get("support", 0)),
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("🏗️ Arsitektur Model")
    st.markdown("""
    <div style='background:#17171A; border:1px solid #2E2E33; border-radius:12px; padding:24px; box-shadow:0 8px 24px rgba(0,0,0,0.35);'>
    <svg viewBox="0 0 640 670" width="100%" style="max-width:520px; display:block; margin:0 auto; font-family:'DM Sans',sans-serif;">
      <defs>
        <marker id="archArrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M2 1L8 5L2 9" fill="none" stroke="#6B6B70" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </marker>
      </defs>

      <rect x="170" y="10" width="300" height="44" rx="8" fill="#1E1E22" stroke="#3A3A40" stroke-width="1"/>
      <text x="320" y="37" text-anchor="middle" font-size="14" font-weight="700" fill="#F2F2F3">Input Teks Opini KDMP</text>
      <line x1="320" y1="54" x2="320" y2="92" stroke="#6B6B70" stroke-width="1.5" marker-end="url(#archArrow)"/>

      <rect x="120" y="92" width="400" height="76" rx="8" fill="rgba(99,102,241,0.10)" stroke="#6366F1" stroke-width="1.2"/>
      <text x="320" y="118" text-anchor="middle" font-size="14" font-weight="700" fill="#F2F2F3">IndoBERT Encoder</text>
      <text x="320" y="140" text-anchor="middle" font-size="11.5" fill="#A3A3A8">indobenchmark/indobert-base-p1</text>
      <text x="320" y="157" text-anchor="middle" font-size="11.5" fill="#A3A3A8">768-dim contextual embedding</text>
      <line x1="320" y1="168" x2="320" y2="206" stroke="#6B6B70" stroke-width="1.5" marker-end="url(#archArrow)"/>

      <rect x="120" y="206" width="400" height="76" rx="8" fill="rgba(16,185,129,0.10)" stroke="#10B981" stroke-width="1.2"/>
      <text x="320" y="232" text-anchor="middle" font-size="14" font-weight="700" fill="#F2F2F3">BiLSTM</text>
      <text x="320" y="254" text-anchor="middle" font-size="11.5" fill="#A3A3A8">Bidirectional sequence encoder</text>
      <text x="320" y="271" text-anchor="middle" font-size="11.5" fill="#A3A3A8">Forward + Backward hidden states</text>
      <line x1="320" y1="282" x2="320" y2="320" stroke="#6B6B70" stroke-width="1.5" marker-end="url(#archArrow)"/>

      <rect x="120" y="320" width="400" height="76" rx="8" fill="rgba(245,158,11,0.10)" stroke="#F59E0B" stroke-width="1.2"/>
      <text x="320" y="346" text-anchor="middle" font-size="14" font-weight="700" fill="#F2F2F3">Attention Mechanism</text>
      <text x="320" y="368" text-anchor="middle" font-size="11.5" fill="#A3A3A8">Context vector + attention weights</text>
      <text x="320" y="385" text-anchor="middle" font-size="11.5" fill="#A3A3A8">Fokus pada token penentu sentimen</text>
      <line x1="320" y1="396" x2="320" y2="434" stroke="#6B6B70" stroke-width="1.5" marker-end="url(#archArrow)"/>

      <rect x="120" y="434" width="400" height="76" rx="8" fill="rgba(239,68,68,0.10)" stroke="#EF4444" stroke-width="1.2"/>
      <text x="320" y="460" text-anchor="middle" font-size="14" font-weight="700" fill="#F2F2F3">Classifier + Softmax</text>
      <text x="320" y="482" text-anchor="middle" font-size="11.5" fill="#A3A3A8">Dropout + Linear projection</text>
      <text x="320" y="499" text-anchor="middle" font-size="11.5" fill="#A3A3A8">Distribusi probabilitas 3 kelas</text>
      <line x1="320" y1="510" x2="320" y2="548" stroke="#6B6B70" stroke-width="1.5" marker-end="url(#archArrow)"/>

      <rect x="120" y="548" width="118" height="56" rx="8" fill="rgba(16,185,129,0.14)" stroke="#10B981" stroke-width="1.2"/>
      <text x="179" y="580" text-anchor="middle" font-size="13" font-weight="700" fill="#10B981">Positif</text>
      <rect x="261" y="548" width="118" height="56" rx="8" fill="rgba(156,156,156,0.14)" stroke="#9C9C9C" stroke-width="1.2"/>
      <text x="320" y="580" text-anchor="middle" font-size="13" font-weight="700" fill="#C4C4C9">Netral</text>
      <rect x="402" y="548" width="118" height="56" rx="8" fill="rgba(239,68,68,0.14)" stroke="#EF4444" stroke-width="1.2"/>
      <text x="461" y="580" text-anchor="middle" font-size="13" font-weight="700" fill="#EF4444">Negatif</text>
      <text x="320" y="632" text-anchor="middle" font-size="11.5" fill="#A3A3A8">Output probabilitas 3 kelas sentimen</text>
    </svg>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PAGE 4: WORDCLOUD
# ══════════════════════════════════════════════════════════════
elif page == "☁️ WordCloud":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>☁️ WordCloud Analisis Sentimen KDMP</div>
        <div class='hero-subtitle'>Perbandingan sebelum/sesudah stopword serta kata dominan per sentimen</div>
    </div>
    """, unsafe_allow_html=True)

    if os.path.exists("wordcloud_before_after.png"):
        st.subheader("🔄 Perbandingan Sebelum vs Sesudah Stopword Removal")
        st.image("wordcloud_before_after.png", use_container_width=True)
        st.divider()

    if df is None or "sentiment" not in df.columns:
        st.info("📂 Dataset berlabel (`df_labeled.csv`) belum tersedia untuk membuat WordCloud per sentimen.")
        st.stop()

    col_wc = st.columns([1, 1, 1])
    wc_cfg = [
        ("Positif", "Greens_r", "😊 Sentimen Positif"),
        ("Netral", "Blues_r", "😐 Sentimen Netral"),
        ("Negatif", "Reds_r", "😠 Sentimen Negatif"),
    ]
    text_col = "clean_tweet" if "clean_tweet" in df.columns else "tweet"

    for col, (sent, cmap, title) in zip(col_wc, wc_cfg):
        with col:
            st.markdown(f"<h4 style='text-align:center;color:{COLORS[sent]};'>{title}</h4>", unsafe_allow_html=True)
            corpus = " ".join(df[df["sentiment"] == sent][text_col].dropna().tolist())
            if corpus.strip():
                wc = WordCloud(
                    width=500,
                    height=350,
                    background_color="#17171A",
                    colormap=cmap,
                    max_words=100,
                    collocations=False,
                    prefer_horizontal=0.9,
                ).generate(corpus)
                fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#17171A")
                ax.imshow(wc, interpolation="bilinear")
                ax.axis("off")
                st.pyplot(fig, use_container_width=True)
                plt.close()
                cnt = len(df[df["sentiment"] == sent])
                st.markdown(f"<p style='text-align:center;color:#A3A3A8;'>{cnt:,} tweet</p>", unsafe_allow_html=True)
            else:
                st.info("Tidak ada data")

    st.divider()
    st.subheader("📊 Top 10 Kata per Sentimen")
    cols_top = st.columns(3)
    for col, sent in zip(cols_top, ["Positif", "Netral", "Negatif"]):
        with col:
            corpus = " ".join(df[df["sentiment"] == sent][text_col].dropna().tolist())
            if corpus.strip():
                words = [w for w in corpus.split() if len(w) > 3]
                top10 = Counter(words).most_common(10)
                if top10:
                    df_top = pd.DataFrame(top10, columns=["Kata", "Frekuensi"])
                    fig, ax = plt.subplots(figsize=(5, 4), facecolor="#17171A")
                    ax.set_facecolor("#17171A")
                    ax.barh(df_top["Kata"][::-1], df_top["Frekuensi"][::-1], color=COLORS[sent], alpha=0.85)
                    ax.set_title(f"Top Kata — {sent}", color="#F2F2F3", fontweight="bold")
                    ax.tick_params(colors="#F2F2F3")
                    for sp in ax.spines.values():
                        sp.set_color("#2E2E33")
                    st.pyplot(fig, use_container_width=True)
                    plt.close()

# ══════════════════════════════════════════════════════════════
# PAGE 5: DATA EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📁 Data Explorer":
    st.markdown("""
    <div class='hero-header'>
        <div class='hero-title'>📁 Data Explorer</div>
        <div class='hero-subtitle'>Jelajahi dataset opini Koperasi Desa Merah Putih (KDMP)</div>
    </div>
    """, unsafe_allow_html=True)

    if df is None:
        st.info("📂 Dataset belum tersedia untuk dijelajahi.")
        st.stop()

    col_f1, col_f2, col_f3 = st.columns([2, 2, 2])
    with col_f1:
        sent_filter = st.multiselect(
            "🎭 Filter Sentimen",
            options=["Positif", "Netral", "Negatif"],
            default=["Positif", "Netral", "Negatif"],
        )
    with col_f2:
        search = st.text_input("🔍 Cari kata kunci:", placeholder="Ketik kata kunci...")
    with col_f3:
        if "user_location" in df.columns:
            locs = sorted(df["user_location"].dropna().unique().tolist())
            loc_filter = st.multiselect("📍 Filter Lokasi", options=locs, default=[])
        else:
            loc_filter = []

    df_show = df.copy()
    if sent_filter and "sentiment" in df_show.columns:
        df_show = df_show[df_show["sentiment"].isin(sent_filter)]
    if search:
        df_show = df_show[df_show["tweet"].astype(str).str.contains(search, case=False, na=False)]
    if loc_filter and "user_location" in df_show.columns:
        df_show = df_show[df_show["user_location"].isin(loc_filter)]

    st.markdown(f"**Menampilkan {len(df_show):,} dari {len(df):,} tweet**")

    show_cols = [c for c in ["tweet", "clean_tweet", "sentiment", "user_location", "likes_count", "retweet_count", "created_at", "username"] if c in df_show.columns]
    df_display = df_show[show_cols].head(500).reset_index(drop=True)
    st.dataframe(df_display, use_container_width=True, height=450)

    st.download_button(
        "⬇️ Download Data yang Difilter (CSV)",
        df_show.to_csv(index=False).encode("utf-8"),
        "data_kdmp_filtered.csv",
        "text/csv",
    )

    st.divider()
    st.subheader("📊 Statistik Dataset")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if "keyword_source" in df.columns:
            st.markdown("**Distribusi per Keyword Source (Top 10):**")
            kw_counts = df["keyword_source"].value_counts().head(10).reset_index()
            kw_counts.columns = ["Keyword", "Jumlah"]
            st.dataframe(kw_counts, use_container_width=True, hide_index=True)
        else:
            st.info("Kolom `keyword_source` tidak tersedia.")
    with col_s2:
        st.markdown("**Statistik Panjang Tweet:**")
        tweet_lens = df["tweet"].astype(str).str.len()
        stats = tweet_lens.describe()
        stat_df = pd.DataFrame({
            "Statistik": ["Rata-rata", "Minimum", "Maximum", "Std Dev"],
            "Nilai": [
                f"{stats['mean']:.1f}",
                f"{stats['min']:.0f}",
                f"{stats['max']:.0f}",
                f"{stats['std']:.1f}",
            ],
        })
        st.dataframe(stat_df, use_container_width=True, hide_index=True)