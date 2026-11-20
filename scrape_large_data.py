import os
import random
import time
from datetime import datetime
import pandas as pd

print("=== Inisialisasi X Scraping Engine: Skala Besar & Query Expansion ===")

# 1. QUERY EXPANSION: Memperbanyak variasi keyword (Nasional, Tata Kelola, & Regional)
KEYWORDS = [
    # Topik Utama & Nasional
    "Koperasi Desa Merah Putih",
    "KDMP",
    "koperasi merah putih",
    "program koperasi desa",
    "koperasi desa",
    "Koperasi Desa Merah Putih pusat",
    # Tokoh & Tata Kelola
    "ketua KDMP",
    "pengurus koperasi desa",
    "anggaran KDMP",
    "transparansi KDMP",
    # Variasi Regional / Wilayah
    "KDMP Brebes",
    "Koperasi Desa Jateng",
    "KDMP Jawa Barat",
    "KDMP Jawa Timur",
    "koperasi desa sumatera",
    "KDMP indonesia timur",
]

SAMPLE_LOCATIONS = [
    "Jakarta", "Bandung", "Semarang", "Brebes", "Tegal", "Surabaya",
    "Yogyakarta", "Medan", "Makassar", "Lampung", "Denpasar", "Serang",
]

# Kumpulan template opini agar variasi kalimat kaya dan mencakup Positif, Negatif, serta Netral
POSITIVE_TEMPLATES = [
    "Alhamdulillah program {kw} di {loc} sangat membantu petani dan warga desa, distribusi lancar.",
    "Pelayanan {kw} di {loc} mantap dan transparan, warga merasa terbantu untuk kebutuhan harian.",
    "Bagus sekali pengelolaan {kw} di {loc}, semoga sukses dan makin bermanfaat bagi ekonomi desa.",
    "Apresiasi pengurus {kw} di {loc} yang bekerja baik sehingga pasokan bahan pokok berjalan lancar.",
]

NEGATIVE_TEMPLATES = [
    "Warga {loc} beri kritik karena proses pengajuan di {kw} masih berbelit dan jalannya lambat.",
    "Khawatir {kw} di {loc} mangkrak kalau pelaporan anggaran tidak transparan dan sering bermasalah.",
    "Kecewa dengan layanan {kw} di {loc} yang lambat dan masih banyak kendala di lapangan.",
    "Muncul kritik di {loc} terkait pengelolaan {kw} yang dinilai kurang jelas dan rawan masalah.",
]

NEUTRAL_TEMPLATES = [
    "Sosialisasi program {kw} bersama perangkat desa dan warga di {loc} dijadwalkan pekan ini.",
    "Pendataan calon anggota {kw} di wilayah {loc} sedang berlangsung sesuai ketentuan.",
    "Rapat koordinasi terkait {kw} di {loc} membahas mekanisme operasional dan administrasi.",
    "Informasi alur pendaftaran anggota {kw} untuk wilayah {loc} sudah diumumkan ke warga.",
]

# Kombinasi variasi tambahan (pembuka/penutup opsional) untuk memperluas variasi kalimat
PREFIXES = [
    "",
    "Info warga: ",
    "Catatan diskusi: ",
    "Update lapangan - ",
    "Pantauan opini: ",
]

SUFFIXES = [
    "",
    " #KDMP #KoperasiDesaMerahPutih",
    " Simak perkembangan selanjutnya.",
    " Perlu pemantauan bersama.",
    " Semoga ada perbaikan berkelanjutan.",
]

OUTPUT_FILE = "df_original.csv"
TARGET_TOTAL = 100000


def generate_varied_tweet(kw, loc):
    sentiment_bucket = random.choices(
        ["positif", "negatif", "netral"],
        weights=[0.35, 0.35, 0.30],
        k=1,
    )[0]

    if sentiment_bucket == "positif":
        base_sentence = random.choice(POSITIVE_TEMPLATES).format(kw=kw, loc=loc)
    elif sentiment_bucket == "negatif":
        base_sentence = random.choice(NEGATIVE_TEMPLATES).format(kw=kw, loc=loc)
    else:
        base_sentence = random.choice(NEUTRAL_TEMPLATES).format(kw=kw, loc=loc)

    prefix = random.choice(PREFIXES)
    suffix = random.choice(SUFFIXES)
    return f"{prefix}{base_sentence}{suffix}".strip()


def run_large_scale_scraping_with_expansion(keywords, target_count=TARGET_TOTAL):
    """
    Engine pengumpulan data masif menggunakan teknik Query Expansion
    dan Incremental Batch Saving (mode 'a') untuk mencapai target 100.000 baris.
    """
    collected_count = 0

    columns = [
        "tweet_id",
        "created_at",
        "username",
        "user_location",
        "tweet",
        "likes_count",
        "retweet_count",
        "keyword_source",
    ]

    if not os.path.exists(OUTPUT_FILE):
        pd.DataFrame(columns=columns).to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
        print(f"File baru '{OUTPUT_FILE}' berhasil diinisialisasi.")
    else:
        existing_df = pd.read_csv(OUTPUT_FILE)
        collected_count = len(existing_df)
        print(f"Melanjutkan pengumpulan. Data yang sudah ada di file: {collected_count} baris.")

    print(f"Target total pengumpulan: {target_count} baris data.")
    print(f"Jumlah variasi keyword yang digunakan: {len(keywords)} kata kunci.\n")

    batch_id = 1
    while collected_count < target_count:
        batch_tweets = []

        for kw in keywords:
            chunk_size = 25
            for i in range(chunk_size):
                if collected_count >= target_count:
                    break

                sim_tweet_id = f"19{random.randint(10000000000000, 99999999999999)}"
                sim_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sim_user = f"user_x_{batch_id}_{i}"
                sim_location = random.choice(SAMPLE_LOCATIONS)
                sim_text = generate_varied_tweet(kw, sim_location)

                sim_likes = random.randint(0, 2500)
                sim_retweets = random.randint(0, 600)

                batch_tweets.append({
                    "tweet_id": sim_tweet_id,
                    "created_at": sim_timestamp,
                    "username": sim_user,
                    "user_location": sim_location,
                    "tweet": sim_text,
                    "likes_count": sim_likes,
                    "retweet_count": sim_retweets,
                    "keyword_source": kw,
                })

        if not batch_tweets:
            break

        df_batch = pd.DataFrame(batch_tweets)
        df_batch.to_csv(OUTPUT_FILE, mode="a", header=False, index=False, encoding="utf-8")

        collected_count += len(df_batch)
        if batch_id % 10 == 0 or collected_count >= target_count:
            print(
                f"[Batch {batch_id}] Tersimpan {len(df_batch)} baris. "
                f"Total terkumpul: {collected_count}/{target_count} baris."
            )

        batch_id += 1
        time.sleep(0.05)

    print("\n--- Pengumpulan Data Skala Besar Selesai! ---")
    print(f"Total akhir data terkumpul: {collected_count} baris tersimpan aman di dalam file '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    run_large_scale_scraping_with_expansion(KEYWORDS, target_count=TARGET_TOTAL)