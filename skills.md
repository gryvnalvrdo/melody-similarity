# AI Audio & Similarity Web UI/UX Engineering Skill (`skills.md`)

> **Target Application**: **Melody Similarity Detector** (`melody-similarity`)  
> **Product Domain**: AI Audio Analysis, Music Plagiarism Verification & Interactive Pitch-Contour Inspector.

---

## 1. Domain Identity & Visual Standards

Berbeda dengan aplikasi SaaS CRM atau reservasi hotel, **Melody Similarity Detector** adalah alat analisis audio teknis yang dipakai oleh produser musik, peneliti, maupun penilai hukum non-teknis. Antarmuka harus mengutamakan **keterbacaan grafik kontur nada, kejelasan skor kemiripan per segmen waktu, dan kenyamanan inspeksi audio**:

1. **Dual Theme Support (Dark & Light Mode)**:
   - Mendukung mode Gelap (`#0d1117` / `#161b22`) dan mode Terang (`#f8fafc` / `#ffffff`) menggunakan CSS custom properties tanpa hardcoded warna gelap pada input/select.
2. **Anti-AI-Slop & Clean UI**:
   - Dilarang memakai teks gradient pelangi pada judul `<h1>` atau emoji dekoratif di tombol dan tab.
   - Dilarang menggunakan tanda hubung panjang (*em-dash* `—`) pada judul halaman dan label antarmuka.
   - Gunakan bahasa yang jelas agar hasil analisis AI mudah diverifikasi oleh pihak non-teknis.
3. **Mobile-First Audio Inspection**:
   - Pemutar audio ganda (*Query vs Match*) dan grafik perbandingan kontur nada wajib tersusun rapi dalam 1 kolom pada layar smartphone (`< 600px`) tanpa overflow horizontal, dilengkapi tombol mengambang kembali ke atas.

---

## 2. Backend Security & High-Concurrency Guardrails (`script/app.py`)

1. **Strict Input & Upload Protection**:
   - Batasi ukuran unggahan (`MAX_CONTENT_LENGTH = 50 MB`), tolak ekstensi video, dan validasi hostname YouTube untuk mencegah penyalahgunaan downloader (`SSRF`).
2. **Inference Rate Limiting & Session Pruning**:
   - Endpoint pemrosesan model PyTorch/Demucs (`/api/query` & `/api/play_compare`) wajib dilindungi pembatasan laju permintaan per IP dan pembersihan sesi/progres berkala agar memori server tidak bocor saat banyak pengguna mengakses bersamaan.
3. **Security Headers & Safe Error Responses**:
   - Setiap respons HTTP wajib menyertakan `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`, dan `Permissions-Policy`, serta tidak pernah mengirimkan *Python traceback* atau path server ke sisi client saat terjadi error `500`.
