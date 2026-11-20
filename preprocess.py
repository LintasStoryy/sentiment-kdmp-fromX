import pandas as pd
import re
import os
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

print("=== Memulai Tahap 2: Cleaning, Stopwords & Labeling ===")

INPUT_FILE = "df_original.csv"
OUTPUT_FILE = "df_labeled.csv"

if not os.path.exists(INPUT_FILE):
    print(f"Error: File '{INPUT_FILE}' tidak ditemukan!")
    exit()

df = pd.read_csv(INPUT_FILE)
print(f"Berhasil memuat {len(df)} baris data dari '{INPUT_FILE}'.")

factory = StopWordRemoverFactory()
stopword_remover = factory.create_stop_word_remover()

def basic_clean_text(text):
    """Pembersihan dasar (tanpa buang stopwords) untuk perbandingan WordCloud 'Sebelum'."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("\n[Proses 1/3] Melakukan cleaning dasar (sebelum stopword removal)...")
df["raw_clean_tweet"] = df["tweet"].apply(basic_clean_text)

print("[Proses 2/3] Melakukan penghapusan stopwords (sesudah stopword removal)...")
df["clean_tweet"] = df["raw_clean_tweet"].apply(lambda x: stopword_remover.remove(x).strip())

# Kamus Leksikon Pelabelan Otomatis
positive_words = [
    "membantu", "mantap", "lancar", "bagus", "sukses", "apresiasi",
    "mudah", "positif", "hebat", "bermanfaat", "terbantu", "transparan", "baik"
]
negative_words = [
    "mangkrak", "rusak", "gagal", "kritik", "berbelit", "kecewa",
    "buruk", "masalah", "korupsi", "rugi", "lambat", "susah", "protes"
]

def label_sentiment(text):
    score = 0
    for word in text.split():
        if word in positive_words:
            score += 1
        elif word in negative_words:
            score -= 1
    if score > 0:
        return "Positif"
    elif score < 0:
        return "Negatif"
    else:
        return "Netral"

print("[Proses 3/3] Melakukan pelabelan otomatis (Positif, Netral, Negatif)...")
df["sentiment"] = df["clean_tweet"].apply(label_sentiment)

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
print(f"\nSelesai! File '{OUTPUT_FILE}' tersimpan dengan kolom 'raw_clean_tweet', 'clean_tweet', dan 'sentiment'.")
print("\nDistribusi Sentimen:")
print(df["sentiment"].value_counts())