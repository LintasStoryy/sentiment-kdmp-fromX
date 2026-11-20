import pandas as pd
import datetime

print("--- Memulai Pengumpulan Data X (Simulasi Skala Lebih Besar) untuk KDMP ---")

# Daftar data mentah (raw data) tweet terkait Koperasi Desa Merah Putih (KDMP)
raw_tweets = [
    {
        "created_at": "2026-06-01 08:30:00",
        "username": "warga_desa_01",
        "tweet": "Program Koperasi Desa Merah Putih atau KDMP ini sangat membantu warga desa kami dalam mendapatkan pupuk dan sembako dengan harga terjangkau. Mantap!"
    },
    {
        "created_at": "2026-06-01 09:15:00",
        "username": "pengamat_ekonomi",
        "tweet": "KDMP harus diawasi ketat transparansinya agar tidak salah kelola seperti program-program desa sebelumnya. Jangan sampai mangkrak."
    },
    {
        "created_at": "2026-06-01 10:00:00",
        "username": "netral_guy",
        "tweet": "Lagi ramai dibahas soal peluncuran Koperasi Desa Merah Putih hari ini di berbagai kabupaten."
    },
    {
        "created_at": "2026-06-01 10:45:00",
        "username": "tani_makmur",
        "tweet": "Alhamdulillah adanya koperasi merah putih bikin petani kecil lebih mudah akses modal usaha. Semoga lancar terus programnya."
    },
    {
        "created_at": "2026-06-01 11:20:00",
        "username": "kritikus_kebijakan",
        "tweet": "Birokrasi pembentukan KDMP ini berbelit-belit di tingkat lokal, bikin warga lelah mengurusnya. Evaluasi dong!"
    },
    {
        "created_at": "2026-06-01 12:05:00",
        "username": "IbuRumahtangga99",
        "tweet": "Bagus banget ada KDMP, harga kebutuhan pokok di desa jadi lebih stabil dan gak terlalu jauh belinya ke kota."
    },
    {
        "created_at": "2026-06-01 12:50:00",
        "username": "pemuda_desa",
        "tweet": "Transparansi pengurus koperasi desa merah putih harus diutamakan, libatkan anak muda lokal biar digitalisasi jalannya lancar."
    },
    {
        "created_at": "2026-06-01 13:30:00",
        "username": "sceptical_id",
        "tweet": "Udah sering lihat program mirip gini, ujung-ujungnya mangkrak kalau pengurusnya gak profesional. Kita lihat saja nanti."
    },
    {
        "created_at": "2026-06-01 14:10:00",
        "username": "umkm_maju",
        "tweet": "Produk UMKM desa kami akhirnya punya tempat pasarkan lewat jaringan KDMP. Terima kasih pemerintah!"
    },
    {
        "created_at": "2026-06-01 15:00:00",
        "username": "wartawan_lokal",
        "tweet": "Peluncuran resmi Koperasi Desa Merah Putih diresmikan oleh pejabat daerah setempat pagi tadi."
    },
    {
        "created_at": "2026-06-01 15:40:00",
        "username": "andi_prasetyo",
        "tweet": "Gak jelas aturan penyertaan modal awalnya gimana, sosialisasi ke warga masih kurang merata."
    },
    {
        "created_at": "2026-06-01 16:20:00",
        "username": "dewi_lestari",
        "tweet": "Wah keren sih kalau KDMP bisa motong rantai pasok tengkulak yang selama ini merugikan petani."
    },
    {
        "created_at": "2026-06-01 17:00:00",
        "username": "pencari_fakta",
        "tweet": "Ada yang punya info syarat gabung jadi anggota Koperasi Desa Merah Putih? Mohon infonya."
    },
    {
        "created_at": "2026-06-01 17:45:00",
        "username": "hendra_stwn",
        "tweet": "Kecewa sama pelayanan tahap awal pendaftaran kemarin, antrean panjang dan panitia kurang sigap."
    },
    {
        "created_at": "2026-06-01 18:30:00",
        "username": "desa_mandiri",
        "tweet": "Optimis KDMP bisa jadi motor penggerak ekonomi baru di perdesaan Indonesia. Maju terus!"
    },
    {
        "created_at": "2026-06-01 19:15:00",
        "username": "ekonom_indonesia",
        "tweet": "Keberhasilan Koperasi Desa Merah Putih sangat bergantung pada kualitas SDM pengelolanya dan bebas dari intervensi politik praktis."
    },
    {
        "created_at": "2026-06-01 20:00:00",
        "username": "siti_aminah",
        "tweet": "Alhamdulillah sembako murah dari koperasi merah putih sangat membantu meringankan pengeluaran bulan ini."
    },
    {
        "created_at": "2026-06-01 20:45:00",
        "username": "rizky_cyber",
        "tweet": "Topik hangat hari ini: perdebatan efektivitas Koperasi Desa Merah Putih di linimasa."
    },
    {
        "created_at": "2026-06-01 21:30:00",
        "username": "budi_sanjaya",
        "tweet": "Jangan sampai program bagus ini cuma jadi ajang pencitraan sesaat tanpa dampak nyata ke masyarakat bawah."
    },
    {
        "created_at": "2026-06-01 22:10:00",
        "username": "lumbung_pangan",
        "tweet": "Dukungan penuh untuk KDMP! Semoga distribusi pupuk bersubsidi bisa tepat sasaran ke tangan petani yang benar-benar membutuhkan."
    }
]

# Mengubah data mentah menjadi DataFrame Pandas
df_original = pd.DataFrame(raw_tweets)

# Menyimpan ke file df_original.csv
output_filename = "df_original.csv"
df_original.to_csv(output_filename, index=False, encoding='utf-8')

print(f"Sukses! Berhasil mengumpulkan {len(df_original)} data tweet.")
print(f"File dataset mentah dengan data yang lebih kaya telah diperbarui pada '{output_filename}'.")