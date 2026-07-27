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
│   └── midi_utils.py         # MIDI utilities
├── web/
│   └── index.html            # Web UI frontend
├── requirements.txt
└── README.md
```

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

### 3. Prepare Dataset

Place your audio files in `dataset/` organized by song ID:
```
dataset/
├── song_001/
│   └── audio.mp3
├── song_002/
│   └── audio.mp3
└── ...
```

### 4. Extract Features

```bash
python script/extract_features.py
```

### 5. Train the Model

```bash
python script/train.py
```

Or use the auto-evaluation training:
```bash
python script/auto_eval_train.py
```

### 6. Build Search Index

```bash
python script/build_index.py
```

### 7. Run the Web App

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

## 🔍 Query Example

```python
from script.query import find_similar_songs

results = find_similar_songs("path/to/query.mp3", top_k=10)
for song_id, similarity in results:
    print(f"{song_id}: {similarity:.4f}")
```

---

## 🌐 Web Interface

Upload an audio file (MP3/WAV/M4A) and discover songs with similar melodies in real time.

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
