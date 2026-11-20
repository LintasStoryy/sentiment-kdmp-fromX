import os
import pickle
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from model_arch import IndoBERTBiLSTMAttention

warnings.filterwarnings("ignore")

print("=== Tahap 3: Training Model IndoBERT + BiLSTM + Attention (KDMP) ===")

# ---------------------------------------------------------
# KONFIGURASI TRAINING (Sesuaikan dengan spesifikasi PC/Laptop)
# ---------------------------------------------------------
DATA_FILE = "df_labeled.csv"
TOKENIZER_DIR = "tokenizer"
MAX_SAMPLES = 3000      # Ubah ke None jika ingin melatih seluruh data (perhatikan waktu komputasi)
MAX_LEN = 64
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 1e-3
LSTM_HIDDEN = 128
DROPOUT = 0.3
FREEZE_BERT = True      # True = cepat di CPU; False = fine-tune seluruh parameter IndoBERT

LABEL_MAP = {"Negatif": 0, "Netral": 1, "Positif": 2}
INV_LABEL_MAP = {0: "Negatif", 1: "Netral", 2: "Positif"}

if not os.path.exists(DATA_FILE) or not os.path.exists(TOKENIZER_DIR):
    print("Error: Pastikan 'df_labeled.csv' dan folder 'tokenizer/' sudah tersedia!")
    raise SystemExit(1)

# 1. Memuat & Menyiapkan Dataset
df = pd.read_csv(DATA_FILE)
df = df.dropna(subset=["clean_tweet", "sentiment"]).copy()
df["clean_tweet"] = df["clean_tweet"].astype(str).str.strip()
df = df[df["clean_tweet"] != ""]
df["label"] = df["sentiment"].map(LABEL_MAP)
df = df.dropna(subset=["label"]).copy()
df["label"] = df["label"].astype(int)

if MAX_SAMPLES is not None and len(df) > MAX_SAMPLES:
    # Usahakan pengambilan sampel tetap mempertahankan variasi label jika tersedia
    if df["label"].nunique() > 1:
        n_per_class = max(1, MAX_SAMPLES // df["label"].nunique())
        sampled_parts = []
        for lbl, grp in df.groupby("label"):
            take_n = min(len(grp), n_per_class)
            sampled_parts.append(grp.sample(n=take_n, random_state=42))
        df_sampled = pd.concat(sampled_parts, ignore_index=True)
        if len(df_sampled) < MAX_SAMPLES:
            remaining = df.drop(df_sampled.index, errors="ignore")
            extra_n = min(len(remaining), MAX_SAMPLES - len(df_sampled))
            if extra_n > 0:
                df_sampled = pd.concat(
                    [df_sampled, remaining.sample(n=extra_n, random_state=42)],
                    ignore_index=True,
                )
        df = df_sampled.sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
    print(f"Menggunakan {len(df)} baris sampel untuk pelatihan.")
else:
    print(f"Menggunakan total {len(df)} baris data untuk pelatihan.")

print("Distribusi label pada data terpilih:", df["label"].map(INV_LABEL_MAP).value_counts().to_dict())

can_stratify = df["label"].nunique() > 1 and (df["label"].value_counts().min() >= 2)
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"] if can_stratify else None,
)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tok, max_len):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


train_dataset = SentimentDataset(train_df["clean_tweet"], train_df["label"], tokenizer, MAX_LEN)
val_dataset = SentimentDataset(val_df["clean_tweet"], val_df["label"], tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 2. Inisialisasi Device & Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Perangkat komputasi yang digunakan: {device}")

model = IndoBERTBiLSTMAttention(
    pretrained_path=TOKENIZER_DIR,
    num_classes=3,
    lstm_hidden=LSTM_HIDDEN,
    dropout_rate=DROPOUT,
    freeze_bert=FREEZE_BERT,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
)

# 3. Loop Pelatihan & Evaluasi
best_val_f1 = -1.0
history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "val_f1": [],
}
best_eval_payload = {}

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()
        preds_train = torch.argmax(logits, dim=1)
        train_correct += (preds_train == labels).sum().item()
        train_total += labels.size(0)

    avg_train_loss = total_train_loss / max(1, len(train_loader))
    train_acc = train_correct / max(1, train_total)

    # Evaluasi pada Validation Set
    model.eval()
    total_val_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_val_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_val_loss = total_val_loss / max(1, len(val_loader))
    val_acc = float(accuracy_score(all_labels, all_preds))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    try:
        y_bin = label_binarize(all_labels, classes=[0, 1, 2])
        probs_arr = np.array(all_probs)
        if y_bin.shape[1] == 3 and len(np.unique(all_labels)) > 1:
            macro_auc = float(
                roc_auc_score(y_bin, probs_arr, average="macro", multi_class="ovr")
            )
        else:
            macro_auc = 0.0
    except Exception:
        macro_auc = 0.0

    history["train_loss"].append(float(avg_train_loss))
    history["val_loss"].append(float(avg_val_loss))
    history["train_acc"].append(float(train_acc))
    history["val_acc"].append(val_acc)
    history["val_f1"].append(float(f1))

    print(
        f"Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
        f"| Val Acc: {val_acc:.4f} | Val F1: {f1:.4f} | Macro-AUC: {macro_auc:.4f}"
    )

    if f1 >= best_val_f1:
        best_val_f1 = float(f1)
        torch.save(model.state_dict(), "best_model.pt")
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2])
        report = classification_report(
            all_labels,
            all_preds,
            labels=[0, 1, 2],
            target_names=["Negatif", "Netral", "Positif"],
            output_dict=True,
            zero_division=0,
        )
        best_eval_payload = {
            "accuracy": val_acc,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "f1_score": float(f1),
            "macro_auc": macro_auc,
            "cm": cm.tolist(),
            "confusion_matrix": cm.tolist(),
            "report": report,
            "classification_report": report,
            "history": history,
            "num_train": int(len(train_df)),
            "num_val": int(len(val_df)),
        }
        print("  -> Bobot model terbaik diperbarui ke 'best_model.pt'")

# 4. Simpan Konfigurasi Model & Hasil Evaluasi (.pkl)
model_config = {
    "architecture": "IndoBERT -> BiLSTM -> Attention -> Softmax",
    "pretrained_model": "indobenchmark/indobert-base-p1",
    "tokenizer_dir": TOKENIZER_DIR,
    "max_len": MAX_LEN,
    "num_classes": 3,
    "lstm_hidden": LSTM_HIDDEN,
    "dropout_rate": DROPOUT,
    "label_map": LABEL_MAP,
    "inv_label_map": INV_LABEL_MAP,
}

with open("model_config.pkl", "wb") as f:
    pickle.dump(model_config, f)

with open("eval_results.pkl", "wb") as f:
    pickle.dump(best_eval_payload, f)

print("\n=== Tahap 3 Selesai! ===")
print("Artefak yang diperbarui: best_model.pt, model_config.pkl, eval_results.pkl")