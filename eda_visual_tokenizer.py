import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from transformers import AutoTokenizer
import os

print("=== Inisialisasi WordCloud (Before/After & Per Sentimen) + Tokenizer ===")

INPUT_FILE = "df_labeled.csv"

if not os.path.exists(INPUT_FILE):
    print(f"Error: File '{INPUT_FILE}' belum tersedia! Jalankan preprocess.py terlebih dahulu.")
    exit()

df = pd.read_csv(INPUT_FILE)
df["raw_clean_tweet"] = df["raw_clean_tweet"].fillna("")
df["clean_tweet"] = df["clean_tweet"].fillna("")
df["sentiment"] = df["sentiment"].fillna("Netral")

def save_wc_image(text, output_path, title, colormap="viridis"):
    if not text.strip():
        print(f"Peringatan: Teks kosong untuk '{title}', melewati pembuatan {output_path}.")
        return
    wc = WordCloud(
        width=900,
        height=500,
        background_color="white",
        colormap=colormap,
        max_words=120
    ).generate(text)
    plt.figure(figsize=(10, 5.5))
    plt.imshow(wc, interpolation="bilinear")
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# ---------------------------------------------------------
# 1. WORDCLOUD SEBELUM vs SESUDAH STOPWORD REMOVAL
# ---------------------------------------------------------
print("\n[1/3] Membuat WordCloud perbandingan Sebelum vs Sesudah Stopwords...")
before_text = " ".join(df["raw_clean_tweet"].tolist())
after_text = " ".join(df["clean_tweet"].tolist())

wc_before = WordCloud(
    width=800, height=450, background_color="white", colormap="Greys", max_words=100
).generate(before_text)

wc_after = WordCloud(
    width=800, height=450, background_color="white", colormap="viridis", max_words=100
).generate(after_text)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(wc_before, interpolation="bilinear")
axes[0].set_title("Sebelum Stopword Removal", fontsize=13, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(wc_after, interpolation="bilinear")
axes[1].set_title("Sesudah Stopword Removal", fontsize=13, fontweight="bold")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("wordcloud_before_after.png", dpi=300)
plt.close()
print("-> Tersimpan: wordcloud_before_after.png")

# ---------------------------------------------------------
# 2. WORDCLOUD PER SENTIMEN (POSITIF, NETRAL, NEGATIF)
# ---------------------------------------------------------
print("[2/3] Membuat WordCloud untuk sentimen Positif, Netral, dan Negatif...")

sentiment_configs = [
    ("Positif", "Greens", "wordcloud_positif.png"),
    ("Netral", "Blues", "wordcloud_netral.png"),
    ("Negatif", "Reds", "wordcloud_negatif.png"),
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

for idx, (label, cmap, single_file) in enumerate(sentiment_configs):
    sub_text = " ".join(df[df["sentiment"] == label]["clean_tweet"].tolist())
    save_wc_image(sub_text, single_file, f"WordCloud Sentimen: {label}", colormap=cmap)

    if sub_text.strip():
        wc_sub = WordCloud(
            width=700, height=450, background_color="white", colormap=cmap, max_words=100
        ).generate(sub_text)
        axes[idx].imshow(wc_sub, interpolation="bilinear")
    else:
        axes[idx].text(0.5, 0.5, f"Tidak ada data {label}", ha="center", va="center", fontsize=12)

    axes[idx].set_title(f"Sentimen: {label}", fontsize=13, fontweight="bold")
    axes[idx].axis("off")

plt.tight_layout()
plt.savefig("wordcloud_sentiments.png", dpi=300)
plt.close()
print("-> Tersimpan: wordcloud_positif.png, wordcloud_netral.png, wordcloud_negatif.png, dan wordcloud_sentiments.png")

# ---------------------------------------------------------
# 3. PERSIAPAN & PENYIMPANAN TOKENIZER (INDOBERT)
# ---------------------------------------------------------
print("[3/3] Menyimpan Tokenizer 'indobenchmark/indobert-base-p1' ke folder 'tokenizer/'...")
MODEL_NAME = "indobenchmark/indobert-base-p1"
TOKENIZER_DIR = "tokenizer"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(TOKENIZER_DIR)

if len(df) > 0 and df["clean_tweet"].iloc[0].strip():
    sample_sentence = df["clean_tweet"].iloc[0]
    tokens = tokenizer.tokenize(sample_sentence)
    token_ids = tokenizer.encode(sample_sentence)
    print("\n--- Contoh Hasil Kerja Tokenizer ---")
    print(f"Teks Bersih : {sample_sentence}")
    print(f"Tokens      : {tokens[:10]}")
    print(f"Token IDs   : {token_ids[:10]}")

print("\nSelesai! Semua gambar WordCloud dan folder tokenizer sudah siap.")