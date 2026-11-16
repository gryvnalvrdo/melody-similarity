# Product Requirements Document (PRD)
## Melody Similarity Detector: AI Audio & Melody Search Web Application

**Nama Sistem**: Melody Similarity Detector  
**Kategori Produk**: AI Audio Processing, Vector Similarity Search & Web Inspection Tool  
**Stack Utama**: Python, Flask, PyTorch (CNN + BiLSTM + Attention Pooling), Demucs Source Separation, FAISS Vector Index, CQT & YIN Pitch Contour Analysis

---

## 1. Gambaran Umum & Tujuan Sistem

**Melody Similarity Detector** adalah aplikasi web berbasis kecerdasan buatan yang menganalisis berkas audio atau tautan YouTube untuk mendeteksi kesamaan melodi terhadap database lagu referensi. Sistem ini dirancang untuk membantu kreator musik, tim hak cipta, dan penilai non-teknis memverifikasi kemiripan melodi secara transparan, bukan sekadar menampilkan angka skor tanpa penjelasan.

---

## 2. Arsitektur & Fitur Utama

### 2.1 Pemrosesan Audio & Model Deep Learning (`script/`)
- **Pemisahan Sumber Audio (Demucs)**: Memisahkan komponen vokal dan melodi dari instrumen drum/perkusi agar ritme latar tidak mengaburkan kemiripan melodi utama.
- **Ekstraksi Fitur CQT & Pencarian FAISS**: Mengubah segmen audio menjadi representasi *Constant-Q Transform* yang tahan terhadap perubahan kunci nada dan mencocokkan embedding melalui indeks vektor FAISS berkecepatan tinggi.
- **Validasi Dunia Nyata**: Diuji terhadap 15 kasus sengketa hak cipta musik nyata dengan tingkat kesesuaian 80%.

### 2.2 Antarmuka Web Interaktif (`web/index.html`)
- **Dual Input**: Unggah file audio lokal (`mp3`, `wav`, `flac`, `m4a`, `ogg`, `aac`, `opus`, `wma`, `aiff`) atau masukkan tautan YouTube resmi.
- **Parameter Pencarian**: Pengaturan ambang batas skor kemiripan minimum (`30%` hingga `90%`), jumlah kandidat terbaik per segmen, serta pilihan perangkat komputasi (`Auto`, `GPU CUDA`, `CPU`).
- **Dual Audio & Pitch Contour Inspector (`/api/play_compare`)**: Memutar potongan audio kueri dan lagu pembanding secara berdampingan lengkap dengan visualisasi grafik kontur nada (algoritma YIN).
- **Ringkasan Lagu & Segmen**: Menampilkan daftar segmen yang cocok maupun segmen di bawah ambang batas beserta alasannya.
- **Mode Gelap & Terang**: Dukungan pilihan tema Dark/Light serta tombol cepat kembali ke atas halaman.

---

## 3. Keamanan, Error Handling & Skalabilitas (`script/app.py`)

- **Validasi Input Ketat**: Memeriksa ekstensi audio yang diizinkan (menolak format video), membatasi ukuran upload maksimal 50 MB (`MAX_CONTENT_LENGTH`), serta membatasi sumber URL khusus domain YouTube resmi (`youtube.com`, `youtu.be`).
- **Security Headers & Rate Limiting**: Menambahkan header keamanan HTTP (`X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, `Permissions-Policy`) serta pembatasan frekuensi permintaan per IP pada endpoint komputasi berat (`/api/query` dan `/api/play_compare`).
- **Manajemen Memori & Pembersihan File Sementara**: Pembersihan otomatis direktori sementara pada blok `finally` dan pemangkasan sesi/progres yang kedaluwarsa (`_prune_sessions`).
- **Sanitasi Error**: Stack trace hanya dicatat di log server, sedangkan respons HTTP 500 ke pengguna selalu menampilkan pesan aman yang terstruktur.
