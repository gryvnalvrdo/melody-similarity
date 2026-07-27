# 🎵 Melody Similarity

A melody similarity search system that finds songs with similar melodies using a **Demucs + CQT-based CRNN** (CNN + Bidirectional LSTM) model trained with **NT-Xent contrastive loss**.

---

## 🧠 How It Works

```
Audio Input
    │
    ▼
[Demucs] → Source Separation (remove drums)
    │         bass + other + vocals
    ▼
[CQT Feature Extraction] → 168-bin Constant-Q Transform
    │                        22050 Hz, 15s segments
    ▼
[CRNN Model]
    ├─ Input Conv (1 → 64 channels)
    ├─ ResBlocks (64 → 128 → 256)
    ├─ Bottleneck Linear (192-dim)
    ├─ Bidirectional LSTM (128 hidden × 2 layers)
    ├─ Attention Pooling
    └─ Projection Head → 256-dim L2-normalized embedding
    │
    ▼
[FAISS Index] → Nearest Neighbor Search
    │
    ▼
Similar Songs 🎶
```

---

## ✨ Features

- 🥁 **Source separation** via [Demucs](https://github.com/facebookresearch/demucs) — removes drums for melody-focused embeddings
- 🎼 **CQT-based features** — musically meaningful frequency representation
- 🔁 **Bidirectional LSTM** with attention pooling — captures temporal melodic patterns
- 📐 **NT-Xent contrastive loss** — robust similarity learning
- 🚀 **Hard Negative Mining** — improves training efficiency
- ⚡ **FAISS index** — fast nearest-neighbor retrieval at scale
- 🌐 **Flask web app** — upload audio and query similar songs via browser

---

## 🏗️ Architecture

| Component | Detail |
|-----------|--------|
| Input | CQT spectrogram — 168 bins × T frames |
| CNN | ResBlocks: 64 → 128 → 256 channels, InstanceNorm |
| Bottleneck | Linear 192-dim + LayerNorm |
| Temporal | Bidirectional LSTM — 128 hidden × 2 layers |
| Pooling | Attention Pooling |
| Output | 256-dim L2-normalized embedding |
| Loss | NT-Xent (Temperature = 0.07) |
| Parameters | ~13M |

---

## 📁 Project Structure

```
melody-similarity/
├── script/
│   ├── model.py              # CRNN model architecture
│   ├── config.py             # Hyperparameters & paths
│   ├── train.py              # Training loop (NT-Xent + Hard Negative Mining)
│   ├── auto_eval_train.py    # Automated training with evaluation
│   ├── extract_features.py   # CQT feature extraction pipeline
│   ├── dataset_triplet.py    # Dataset & DataLoader
│   ├── evaluate.py           # Model evaluation metrics
│   ├── build_index.py        # Build FAISS search index
│   ├── query.py              # Query engine & inference
│   ├── app.py                # Flask web application
│   ├── audio_utils.py        # Audio processing utilities
│   ├── demucs_utils.py       # Demucs source separation wrapper
│   ├── music_theory.py       # Music theory helpers
│   ├── midi_utils.py         # MIDI utilities
│   └── scraper.py            # Dataset scraper (YouTube via yt-dlp)
├── web/
│   └── index.html            # Web UI frontend
├── dataset.csv               # Scraped dataset index (song_id, title, artist, YouTube URL)
├── spotify_songs.csv         # Source song list from Spotify dataset
├── requirements.txt
└── README.md
```

---

## 📂 Dataset

The dataset is built from a **two-step pipeline**: first collecting song metadata, then downloading the actual audio files.

---

### Step 1 — Song List Collection (`scraper.py`)

**Source file:** [`spotify_songs.csv`](spotify_songs.csv)

This file contains ~32,841 tracks from Spotify with rich metadata:

| Column | Description |
|--------|-------------|
| `track_id` | Spotify track ID |
| `track_name` | Song title |
| `track_artist` | Artist name |
| `track_popularity` | Spotify popularity score (0–100) |
| `playlist_genre` | Genre (pop, rock, EDM, etc.) |
| `danceability`, `energy`, `tempo`, ... | Spotify audio features |

**Script:** [`script/scraper.py`](script/scraper.py)

The scraper reads `spotify_songs.csv` and for each song:
1. Searches YouTube for the **original version** and multiple **cover versions**
2. Uses the YouTube Data API to find the best matching video
3. Saves the results to [`dataset.csv`](dataset.csv)

```bash
# Run the scraper to collect YouTube URLs
python script/scraper.py
```

> ⚠️ **Requires a YouTube Data API key:**
> ```bash
> # Windows
> set YOUTUBE_API_KEY=your_api_key_here
>
> # Linux / macOS
> export YOUTUBE_API_KEY=your_api_key_here
> ```
> Get a free API key at [Google Cloud Console](https://console.cloud.google.com/).

**Output — [`dataset.csv`](dataset.csv)** — the scraped index with columns:

| Column | Description | Example |
|--------|-------------|---------|
| `song_id` | Unique song identifier | `001` |
| `track_id` | Version-specific ID | `001_1`, `001_2` |
| `title` | Song title | `Mantan Terindah` |
| `artist` | Artist name | `Kahitna` |
| `version` | Version label | `original`, `cover`, `cover_2`, `cover_3` |
| `url` | YouTube URL | `https://youtu.be/...` |

Each `song_id` groups one song with its original and cover versions — this pairing is critical for training the similarity model (positives = same song, negatives = different songs).

---

### Step 2 — Audio Download (`download_dataset.py`)

**Script:** [`script/download_dataset.py`](script/download_dataset.py)

This script reads `dataset.csv` and downloads all audio files from YouTube using **yt-dlp**. It is designed to be robust and production-ready with many smart features.

```bash
# Download the entire dataset
python script/download_dataset.py

# Download a specific range of songs (useful for splitting across machines)
python script/download_dataset.py --start-song 1 --end-song 1000

# Control the number of parallel download threads
python script/download_dataset.py --threads 8

# Download a single song interactively (for testing)
python script/download_dataset.py --single
```

**Key features of the downloader:**

| Feature | Description |
|---------|-------------|
| 🔄 **Resume support** | Already-downloaded files are automatically skipped — safe to re-run |
| ⚡ **Parallel downloads** | Up to 12 threads by default (`--threads` to override) |
| 🚀 **aria2c acceleration** | Automatically uses aria2c (16 connections) if installed |
| 🤖 **Bot detection handling** | Detects YouTube bot-checks and pauses automatically; stops after 3 detections |
| ⏱️ **Rate limit backoff** | Handles HTTP 429 (too many requests) with exponential backoff |
| 🗑️ **Auto-cleanup** | Permanently unavailable videos are removed from `dataset.csv` automatically |
| 🔍 **Duplicate check** | Checks for duplicate `(title, original artist)` pairs before downloading |
| ✅ **Verification** | Verifies the folder structure after download and reports incomplete songs |

**Output structure:**
```
dataset/
├── 001/
│   ├── original.wav    ← original artist
│   ├── cover.wav       ← cover version 1
│   └── cover_2.wav     ← cover version 2 (if available)
├── 002/
│   ├── original.wav
│   └── cover.wav
└── ...
```

> 💡 **Tip:** If YouTube bot-detection stops the download, wait 30–60 minutes and re-run. Progress is saved automatically.

---

### Dataset Stats

| File | Description | Size |
|------|-------------|------|
| `spotify_songs.csv` | Source Spotify track list | ~32,841 tracks |
| `dataset.csv` | Scraped YouTube index (original + covers) | ~12,495 entries |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/gryvnalvrdo/melody-similarity.git
cd melody-similarity
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Demucs requires `ffmpeg` to be installed on your system.
> - Windows: `winget install Gyan.FFmpeg`
> - Linux: `sudo apt install ffmpeg`
> - macOS: `brew install ffmpeg`

### 3. Download & Prepare Dataset

Use the provided scripts to build the dataset (see [Dataset](#-dataset) section for full details):

```bash
# Step A: Scrape YouTube URLs from spotify_songs.csv
python script/scraper.py

# Step B: Download audio files from YouTube
python script/download_dataset.py
```

The resulting `dataset/` folder will have this structure:
```
dataset/
├── 001/
│   ├── original.wav    ← original artist recording
│   ├── cover.wav       ← cover version 1
│   └── cover_2.wav     ← cover version 2 (if available)
├── 002/
│   ├── original.wav
│   └── cover.wav
└── ...
```

### 4. Extract CQT Features

This step separates drums using Demucs, then computes CQT spectrograms for each audio segment and saves them to an HDF5 file:

```bash
python script/extract_features.py
```

> 💡 Features are stored in `dataset.h5` (not tracked by git due to size). This file is required for training.

### 5. Train the Model

```bash
python script/train.py
```

The training uses **NT-Xent contrastive loss** with **Hard Negative Mining** starting from epoch 5. The best model is saved to `models/best_model.pt`.

### 6. Build Search Index

After training, generate embeddings for all songs and save them into a searchable index:

```bash
python script/build_index.py
```

This loads `models/best_model.pt`, runs all extracted features through the model, and saves a FAISS-compatible index to the `index/` directory.

### 7. Query — Find Similar Songs

```bash
# Query using a local audio file
python script/query.py --input path/to/song.mp3

# Query using a YouTube URL directly
python script/query.py --input https://youtu.be/xxxx

# Control number of results
python script/query.py --input path/to/song.mp3 --top-k 20
```

### 8. Run the Web App

```bash
python script/app.py
```

Then open your browser at `http://localhost:5000`

---

## ⚙️ Configuration

Key hyperparameters can be adjusted in [`script/config.py`](script/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 22050 | Audio sample rate |
| `SEGMENT_DURATION` | 15.0s | Segment length for feature extraction |
| `SEGMENT_HOP` | 5.0s | Hop between segments |
| `EMBEDDING_DIM` | 256 | Output embedding dimension |
| `CNN_CHANNELS` | [64, 128, 256] | CNN channel progression |
| `LSTM_HIDDEN` | 128 | LSTM hidden size |
| `LSTM_LAYERS` | 2 | Number of LSTM layers |
| `BOTTLENECK_DIM` | 192 | Bottleneck linear dimension |
| `TEMPERATURE` | 0.07 | NT-Xent temperature |
| `LEARNING_RATE` | 5e-5 | AdamW learning rate |
| `BATCH_SIZE` | 8 | Training batch size |
| `EPOCHS` | 50 | Max training epochs |

---

## 📊 Training Details

- **Loss:** NT-Xent (Normalized Temperature-scaled Cross Entropy)
- **Optimizer:** AdamW with cosine annealing scheduler
- **Regularization:** Dropout (0.2), Weight Decay (1e-4)
- **Hard Negative Mining:** Starts at epoch 5, updates every 2 epochs
- **Early Stopping:** Patience = 10 epochs
- **Gradient Accumulation:** 16 steps (effective batch size = 128)

---

## 🔍 Query — Finding Similar Songs

[`script/query.py`](script/query.py) is the main inference engine. It accepts a local audio file **or** a YouTube URL and returns the most similar songs from the index.

```bash
# From a local file
python script/query.py --input song.mp3

# From a YouTube URL (auto-downloaded)
python script/query.py --input https://youtu.be/xxxx

# Show top 20 results
python script/query.py --input song.mp3 --top-k 20
```

**How the query pipeline works:**
1. Load and normalize the audio
2. Apply Demucs source separation (remove drums)
3. Compute CQT features on 15-second sliding windows
4. Run each window through the CRNN model → embedding
5. Search the FAISS index using **MaxSim** (max similarity across all windows)
6. Filter results using **consecutive window matching** to reduce false positives
7. Return ranked list of similar songs with similarity scores

---

## 🌐 Web Interface

[`script/app.py`](script/app.py) is a **Flask** web server that exposes the full query pipeline via a browser UI.

```bash
python script/app.py
# Open http://localhost:5000
```

**Features:**
- 🎵 **Upload audio** (MP3, WAV, M4A) and find similar songs
- 🔗 **Paste a YouTube URL** to query directly without downloading
- 📊 **Real-time progress** tracking during feature extraction
- 🎯 **Ranked results** with similarity scores and song metadata
- 🔊 **Session-based** request handling with TTL cleanup

---

## 📦 Model Weights

Pre-trained model weights are available on Hugging Face Hub:

> 🤗 Coming soon...

---

## 📋 Requirements

- Python 3.9+
- PyTorch 2.0+ (CUDA recommended)
- See [`requirements.txt`](requirements.txt) for full list

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Demucs](https://github.com/facebookresearch/demucs) — Music source separation
- [librosa](https://librosa.org/) — Audio feature extraction
- [FAISS](https://github.com/facebookresearch/faiss) — Efficient similarity search
- [PyTorch](https://pytorch.org/) — Deep learning framework
