from __future__ import annotations
import os, sys, io, json, tempfile, threading, traceback, uuid, time, gc, random
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.dirname(_SCRIPT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from flask import Flask, request, jsonify, send_from_directory, send_file, Response
from flask_cors import CORS
import logging
logging.getLogger('werkzeug').setLevel(logging.ERROR)

_model_cache: dict = {}
_index  = None
_meta   = None
_lock   = threading.Lock()

_SESSION_STORE: dict = {}
_SESSION_TTL  = 3600          

_PROGRESS: dict = {}

_BENCH_LOCK   = threading.Lock()
_BENCH_STATUS = {"running": False, "results": None, "error": None, "log": []}

def _set_progress(session_id: str, stage: str, detail: str = ""):
    if session_id:
        _PROGRESS[session_id] = {'stage': stage, 'detail': detail}

def _prune_sessions():
    now = time.time()
    expired = [k for k, v in _SESSION_STORE.items()
               if now - v.get('created', now) > _SESSION_TTL]
    for k in expired:
        sess = _SESSION_STORE.pop(k, {})
                                          
        import shutil
        for path in sess.get('dl_cache', {}).values():
            try:
                shutil.rmtree(os.path.dirname(path), ignore_errors=True)
            except Exception:
                pass
                                   
        qp = sess.get('query_path', '')
        if qp and os.path.exists(qp):
            try:
                os.remove(qp)
            except Exception:
                pass

def _resolve_device(device_pref: str = "auto") -> str:
    import torch
    pref = (device_pref or "auto").strip().lower()
    if pref in ("gpu", "cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("GPU diminta tapi CUDA tidak tersedia di server ini.")
        return "cuda"
    elif pref == "cpu":
        return "cpu"
    else:        
        return "cuda" if torch.cuda.is_available() else "cpu"

def _ensure_loaded(device_key: str = None, model_path=None):
    global _model_cache, _index, _meta
    import torch
    if device_key is None:
        device_key = "cuda" if torch.cuda.is_available() else "cpu"
    with _lock:
        if device_key not in _model_cache or _index is None:
            from melody_first import load_model, load_index, load_song_metadata
            if device_key not in _model_cache:
                                                                                      
                import melody_first as _mf
                import config as _cfg
                _orig_dev = _cfg.DEVICE
                _cfg.DEVICE = torch.device(device_key)
                _mf.DEVICE  = torch.device(device_key)
                try:
                    loaded = load_model(model_path)
                    loaded = loaded.to(torch.device(device_key))
                    _model_cache[device_key] = loaded
                finally:
                    _cfg.DEVICE = _orig_dev
                    _mf.DEVICE  = _orig_dev
            if _index is None:
                _index = load_index()
            if _meta is None:
                _meta = load_song_metadata()
    return _model_cache[device_key], _index, _meta

app = Flask(__name__,
            static_folder=os.path.join(_ROOT_DIR, "web"),
            static_url_path="")
CORS(app)

@app.route("/")
def index():
    return send_from_directory(os.path.join(_ROOT_DIR, "web"), "index.html")

@app.route("/api/check_gpu", methods=["GET"])
def api_check_gpu():
    import torch
    return jsonify({"has_gpu": torch.cuda.is_available()})

@app.route("/api/progress/<session_id>", methods=["GET"])
def api_progress(session_id):
    prog = _PROGRESS.get(session_id, {'stage': 'starting', 'detail': ''})
    return jsonify(prog)

@app.route("/api/device-info", methods=["GET"])
def api_device_info():
    import torch
    info = {"cpu_threads": torch.get_num_threads()}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_available"]   = True
        info["gpu_name"]        = props.name
        info["gpu_vram_mb"]     = props.total_memory // 1024**2
        info["gpu_used_mb"]     = torch.cuda.memory_allocated(0) // 1024**2
        info["gpu_reserved_mb"] = torch.cuda.memory_reserved(0)  // 1024**2
        info["cuda_version"]    = torch.version.cuda or "N/A"
    else:
        info["gpu_available"] = False
    info["torch_version"] = torch.__version__
    return jsonify(info)

@app.route("/api/benchmark", methods=["POST"])
def api_benchmark():
    global _BENCH_STATUS
    with _BENCH_LOCK:
        if _BENCH_STATUS["running"]:
            return jsonify({"error": "Benchmark sedang berjalan. Tunggu selesai."}), 409
        _BENCH_STATUS = {"running": True, "results": None, "error": None, "log": []}

    body = request.get_json(silent=True) or {}
    n_batches_gpu = int(body.get("n_batches_gpu", 20))
    n_batches_cpu = int(body.get("n_batches_cpu", 10))
    batch_size    = int(body.get("batch_size",    8))
    n_epochs      = int(body.get("n_epochs",      2))
    warmup        = int(body.get("warmup_batches", 3))

    def _log(msg: str):
        print(f"[BENCH] {msg}", flush=True)
        with _BENCH_LOCK:
            _BENCH_STATUS["log"].append(msg)

    def _run_benchmark():
        global _BENCH_STATUS
        import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
        import h5py

        h5_path = os.path.join(_ROOT_DIR, "dataset.h5")
        if not os.path.exists(h5_path):
            with _BENCH_LOCK:
                _BENCH_STATUS["running"] = False
                _BENCH_STATUS["error"]   = f"dataset.h5 tidak ditemukan di {h5_path}"
            return

        try:
            from config import TEMPERATURE
            from model import MelodySimilarityModel
        except ImportError:
            from script.config import TEMPERATURE
            from script.model import MelodySimilarityModel

        class _SupConLoss(nn.Module):
            def __init__(self, temp=0.07):
                super().__init__()
                self.temp = temp
            def forward(self, a, p, s=None):
                N = a.size(0)
                dev = a.device
                a = F.normalize(a, p=2, dim=1)
                p = F.normalize(p, p=2, dim=1)
                feats = torch.cat([a, p], dim=0)
                lbl = torch.cat([s, s]) if s is not None else torch.cat(
                    [torch.arange(N, device=dev), torch.arange(N, device=dev)])
                sim = torch.matmul(feats, feats.T) / self.temp
                mask = (lbl.unsqueeze(1) == lbl.unsqueeze(0)).float()
                eye = torch.eye(2*N, device=dev)
                mask = mask * (1 - eye)
                sim_max, _ = sim.max(dim=1, keepdim=True)
                sim = sim - sim_max.detach()
                exp_sim = torch.exp(sim) * (1 - eye)
                log_den = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)
                log_p = sim - log_den
                num_p = mask.sum(dim=1)
                valid = num_p > 0
                per = -(mask * log_p).sum(dim=1)
                per = torch.where(valid, per / (num_p + 1e-9), torch.zeros_like(per))
                return per[valid].mean() if valid.any() else torch.tensor(0.0, device=dev)

        with h5py.File(h5_path, 'r') as f:
            all_keys = sorted(f.keys(), key=lambda x: int(x))
                                                   
        n_songs = max(batch_size * 4, round(len(all_keys) * 0.20))
        song_ids = all_keys[:n_songs]
        _log(f"Benchmark menggunakan {n_songs} songs (20% dari {len(all_keys)} total)")

        pairs = []
        with h5py.File(h5_path, 'r') as f:
            for sid in song_ids:
                if sid not in f: continue
                g = f[sid]
                vers = list(g.keys())
                seg_counts = {v: g[v].shape[0] for v in vers}
                for i, va in enumerate(vers):
                    for vp in vers[i+1:]:
                        n_p = min(seg_counts[va], seg_counts[vp])
                        for idx in range(n_p):
                            pairs.append((sid, va, idx, vp, idx))
        song_id_map = {s: i for i, s in enumerate(song_ids)}
        _log(f"Total pairs tersedia: {len(pairs):,}")

        def make_batches(n_batches):
            pool = list(pairs)
            random.shuffle(pool)
            pool_idx = 0
            batches = []
            with h5py.File(h5_path, 'r') as f:
                for _ in range(n_batches):
                    if pool_idx + batch_size > len(pool):
                        random.shuffle(pool)
                        pool_idx = 0
                    bp = pool[pool_idx : pool_idx + batch_size]
                    pool_idx += batch_size
                    ancs, poss, sids = [], [], []
                    for (sid, va, ia, vp, ip) in bp:
                        ancs.append(torch.from_numpy(f[sid][va][ia].astype(np.float32)))
                        poss.append(torch.from_numpy(f[sid][vp][ip].astype(np.float32)))
                        sids.append(song_id_map[sid])
                    batches.append((
                        torch.stack(ancs),
                        torch.stack(poss),
                        torch.tensor(sids, dtype=torch.long),
                    ))
            return batches

        def bench_device(device, n_batches):
            _log(f"--- {str(device).upper()} ({n_batches} batches x {n_epochs} epochs) ---")
            if device.type == 'cuda':
                gc.collect(); torch.cuda.empty_cache()
            model     = MelodySimilarityModel().to(device)
            criterion = _SupConLoss(TEMPERATURE).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=1e-4)
            scaler    = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

            _log(f"  Memuat {n_batches + warmup} batches dari HDF5...")
            all_batches = make_batches(n_batches + warmup)
            warmup_batches = all_batches[:warmup]
            bench_batches  = all_batches[warmup:]

            model.train()
            for a, p, s in warmup_batches:
                a, p, s = a.to(device), p.to(device), s.to(device)
                if scaler:
                    with torch.amp.autocast('cuda'):
                        loss = criterion(model(a), model(p), s)
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss = criterion(model(a), model(p), s)
                    loss.backward(); optimizer.step()
                optimizer.zero_grad()
            if device.type == 'cuda': torch.cuda.synchronize()

            epoch_times = []
            for ep in range(n_epochs):
                t0 = time.perf_counter()
                model.train()
                for a, p, s in bench_batches:
                    a, p, s = a.to(device), p.to(device), s.to(device)
                    if scaler:
                        with torch.amp.autocast('cuda'):
                            loss = criterion(model(a), model(p), s)
                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer); scaler.update()
                    else:
                        loss = criterion(model(a), model(p), s)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad()
                if device.type == 'cuda': torch.cuda.synchronize()
                t_ep = time.perf_counter() - t0
                epoch_times.append(t_ep)
                _log(f"  Epoch {ep+1}/{n_epochs}: {t_ep:.2f}s ({t_ep/n_batches:.3f}s/batch)")

            avg = float(np.mean(epoch_times))
            spb = avg / n_batches
            est_full = (spb * 6250) / 60                             
            _log(f"  Avg: {avg:.2f}s ± {float(np.std(epoch_times)):.2f}s | est. full epoch: ~{est_full:.0f} min")

            del model, criterion, optimizer, scaler, all_batches
            if device.type == 'cuda':
                gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()

            return {
                "device":          str(device),
                "n_batches":       n_batches,
                "n_epochs":        n_epochs,
                "avg_epoch_sec":   round(avg, 3),
                "std_epoch_sec":   round(float(np.std(epoch_times)), 3),
                "sec_per_batch":   round(spb, 4),
                "est_full_epoch_min": round(est_full, 1),
                "epoch_times":     [round(t, 3) for t in epoch_times],
            }

        results = {"n_songs": n_songs, "n_pairs": len(pairs),
                   "batch_size": batch_size}
        try:
            if torch.cuda.is_available():
                results["gpu"] = bench_device(torch.device("cuda"), n_batches_gpu)
            else:
                results["gpu"] = None
                _log("GPU tidak tersedia, skip GPU benchmark.")
        except Exception as e:
            _log(f"GPU error: {e}")
            results["gpu"] = {"error": str(e)}

        try:
            results["cpu"] = bench_device(torch.device("cpu"), n_batches_cpu)
        except Exception as e:
            _log(f"CPU error: {e}")
            results["cpu"] = {"error": str(e)}

        if (results.get("gpu") and results.get("cpu")
                and "sec_per_batch" in results["gpu"]
                and "sec_per_batch" in results["cpu"]):
            results["speedup_x"] = round(
                results["cpu"]["sec_per_batch"] / results["gpu"]["sec_per_batch"], 2)
        else:
            results["speedup_x"] = None

        _log("Benchmark selesai!")
        with _BENCH_LOCK:
            _BENCH_STATUS["running"] = False
            _BENCH_STATUS["results"] = results

    t = threading.Thread(target=_run_benchmark, daemon=True)
    t.start()
    return jsonify({"status": "started",
                    "message": "Benchmark GPU vs CPU dimulai di background. Poll /api/benchmark/status untuk hasil."})

@app.route("/api/benchmark/status", methods=["GET"])
def api_benchmark_status():
    with _BENCH_LOCK:
        status = {
            "running": _BENCH_STATUS["running"],
            "results": _BENCH_STATUS["results"],
            "error":   _BENCH_STATUS["error"],
            "log":     list(_BENCH_STATUS["log"]),
        }
    return jsonify(status)

@app.route("/api/query", methods=["POST"])
def api_query():
    try:
        top_k      = int(request.form.get("top_k",      5)  or (request.json or {}).get("top_k",      5))
        candidates = int(request.form.get("candidates", 80) or (request.json or {}).get("candidates", 80))
        min_score  = float(request.form.get("min_score", 0.70) or (request.json or {}).get("min_score", 0.70))
        device_pref = (request.form.get("device", "auto")
                       or (request.json or {}).get("device", "auto") or "auto")

        tmp_path   = None
        session_id = request.form.get("session_id") or (request.json or {}).get("session_id") or str(uuid.uuid4())

        if request.files and "file" in request.files:
            f      = request.files["file"]
            suffix = os.path.splitext(f.filename)[-1].lower() or ".wav"

            ALLOWED_AUDIO = {".mp3", ".wav", ".m4a", ".flac", ".ogg",
                             ".aac", ".opus", ".wma", ".aiff", ".aif"}
            REJECTED_VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".webm",
                              ".flv", ".wmv", ".mpeg", ".mpg", ".ts"}
            _AUDIO_LIST = "mp3, wav, m4a, flac, ogg, aac, opus, wma, aiff"

            if suffix in REJECTED_VIDEO:
                return jsonify({
                    "error": (
                        f"Format '{suffix}' adalah file video dan tidak didukung. "
                        f"Aplikasi ini hanya menerima file audio.\n"
                        f"Format yang didukung: {_AUDIO_LIST}."
                    )
                }), 415
            elif suffix not in ALLOWED_AUDIO:
                return jsonify({
                    "error": (
                        f"Format file '{suffix}' tidak didukung.\n"
                        f"Format audio yang diterima: {_AUDIO_LIST}."
                    )
                }), 415

            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            f.save(tmp_path)
            input_src = tmp_path

        else:
            url = (request.form.get("url", "").strip()
                   or (request.json or {}).get("url", "").strip())
            if not url:
                return jsonify({"error": "No URL or file provided"}), 400

            import urllib.parse as _up
            _parsed = _up.urlparse(url)
            _host   = _parsed.netloc.lower().lstrip("www.")
            _YOUTUBE_HOSTS = {"youtube.com", "youtu.be", "m.youtube.com",
                              "youtube-nocookie.com"}
            if _parsed.scheme not in ("http", "https"):
                return jsonify({
                    "error": (
                        "URL tidak valid. Harap masukkan link YouTube yang benar.\n"
                        "Contoh: https://www.youtube.com/watch?v=... "
                        "atau https://youtu.be/..."
                    )
                }), 400
            if _host not in _YOUTUBE_HOSTS:
                return jsonify({
                    "error": (
                        f"URL dari '{_parsed.netloc}' tidak didukung.\n"
                        "Aplikasi ini hanya menerima link dari YouTube.\n"
                        "Contoh yang valid:\n"
                        "  • https://www.youtube.com/watch?v=xxxxx\n"
                        "  • https://youtu.be/xxxxx"
                    )
                }), 415

            input_src = url

        try:
            device_key = _resolve_device(device_pref)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 400

        _prune_sessions()
        _SESSION_STORE[session_id] = {
            'query_path': tmp_path or '',                                  
            'input_src':  input_src,
            'dl_cache':   {},
            'created':    time.time(),
        }
        _set_progress(session_id, 'loading_model', 'Memuat model')

        result = _run_query(input_src, top_k, candidates, min_score,
                            session_id=session_id, device_key=device_key)
        _set_progress(session_id, 'done', 'Selesai!')
        result['session_id'] = session_id
        return jsonify(result)

    except Exception as e:
        tb = traceback.format_exc()
        try:
            print("[API ERROR]\n" + tb, flush=True)
        except Exception:
            pass
        return jsonify({"error": str(e), "traceback": tb}), 500
    finally:
                                                                         
        pass

@app.route("/api/play_compare", methods=["GET"])
def api_play_compare():
    import base64, shutil

    session_id = request.args.get("session_id", "")
    q_start    = float(request.args.get("q_start", 0))
    q_end      = float(request.args.get("q_end",   15))
    song_id    = request.args.get("song_id",  "")
    version    = request.args.get("version",  "original").lower()
    m_start    = float(request.args.get("m_start", 0))
    m_end      = float(request.args.get("m_end",   15))

    sess = _SESSION_STORE.get(session_id)
    if sess is None:
        return jsonify({"error": "Session expired or not found. Please re-run the query."}), 404

    try:
        from audio_utils import (extract_melody_segment,
                                  download_and_extract_proof_melody,
                                  get_pitch_contour,
                                  _normalize_song_id, _load_csv_urls, _CSV_URL_CACHE)
    except ImportError:
        from script.audio_utils import (extract_melody_segment,
                                         download_and_extract_proof_melody,
                                         get_pitch_contour,
                                         _normalize_song_id, _load_csv_urls, _CSV_URL_CACHE)

    tmp_out_dir = tempfile.mkdtemp(prefix="play_compare_")
    try:
        import librosa, soundfile as sf
        try:
            from demucs_utils import remove_drums
        except ImportError:
            from script.demucs_utils import remove_drums

        def _demucs_clean(raw_path: str, out_name: str, label: str) -> str:
            """Jalankan Demucs drum-removal pada satu klip WAV.
            Mengembalikan path hasil (sudah dinormalisasi -3 dBFS),
            atau path mentah (raw_path) sebagai fallback jika Demucs gagal."""
            try:
                print(f"   🥁  Demucs: removing drums from {label} clip…")
                audio_raw, sr_raw = librosa.load(raw_path, sr=None, mono=True)
                audio_nd = remove_drums(audio_raw, sr_raw)
                if audio_nd is None:
                    print(f"   ⚠️  Demucs returned None — using raw {label} clip")
                    return raw_path
                peak = np.max(np.abs(audio_nd))
                if peak > 1e-6:
                    audio_nd = audio_nd * (10 ** (-3.0 / 20.0) / peak)
                nd_path = os.path.join(tmp_out_dir, f"{out_name}_nd.wav")
                sf.write(nd_path, audio_nd, sr_raw, subtype="PCM_16")
                print(f"   ✅  {label.capitalize()} clip: drums removed")
                return nd_path
            except Exception as e:
                print(f"   ⚠️  Demucs drum removal failed for {label}: {e} — using raw")
                return raw_path

        # ── Query clip ────────────────────────────────────────────────────────
        q_wav_raw  = os.path.join(tmp_out_dir, "query_clip_raw.wav")
        query_path = sess.get('query_path', '')
        query_ok   = False
        if query_path and os.path.exists(query_path):
            query_ok = extract_melody_segment(query_path, q_start, q_end, q_wav_raw)
        q_wav = _demucs_clean(q_wav_raw, "query_clip", "query") if query_ok else q_wav_raw

        # ── Match clip (raw download) ─────────────────────────────────────────
        m_wav_raw = os.path.join(tmp_out_dir, "match_clip_raw.wav")
        match_ok  = download_and_extract_proof_melody(
            song_id    = song_id,
            version    = version,
            start_time = m_start,
            end_time   = m_end,
            output_path= m_wav_raw,
            session_cache= sess['dl_cache'],
        )
        m_wav = _demucs_clean(m_wav_raw, "match_clip", "match") if match_ok else m_wav_raw

        def wav_to_b64(path):
            if not path or not os.path.exists(path):
                return None
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')

        query_contour = get_pitch_contour(q_wav)  if query_ok else {"times": [], "notes": []}
        match_contour = get_pitch_contour(m_wav)  if match_ok else {"times": [], "notes": []}

        return jsonify({
            "query_audio_b64": wav_to_b64(q_wav) if query_ok else None,
            "match_audio_b64": wav_to_b64(m_wav) if match_ok else None,
            "query_contour": query_contour,
            "match_contour": match_contour,
            "query_ok":  query_ok,
            "match_ok":  match_ok,
            "q_start": q_start, "q_end": q_end,
            "m_start": m_start, "m_end": m_end,
        })

    except Exception as e:
        tb = traceback.format_exc()
        print("[PLAY_COMPARE ERROR]\n" + tb, flush=True)
        return jsonify({"error": str(e), "traceback": tb}), 500
    finally:
        shutil.rmtree(tmp_out_dir, ignore_errors=True)


def _run_query(input_src: str, top_k: int, n_candidates: int,
               min_score: float, session_id: str = None,
               device_key: str = None) -> dict:
    import torch
    from melody_first import (
        melody_query_core,
        is_youtube_url, download_youtube, normalize_query_audio,
        cleanup_normalized_audio,
        format_time,
    )
    import melody_first as _mf
    import config as _cfg

    if device_key is None:
        device_key = "cuda" if torch.cuda.is_available() else "cpu"
    _device_obj = torch.device(device_key)

    _orig_dev = _cfg.DEVICE
    _cfg.DEVICE = _device_obj
    _mf.DEVICE  = _device_obj

    model, index, meta_db = _ensure_loaded(device_key=device_key)

    _original    = input_src
    _tmp_norm    = None
    _query_title  = ''
    _query_artist = ''

    _t_total_start = time.perf_counter()
    _timing: dict  = {}

    if is_youtube_url(input_src):
                                                                    
        try:
            import yt_dlp as _yt_meta
            with _yt_meta.YoutubeDL({'quiet': True, 'no_warnings': True}) as _ydl:
                _info = _ydl.extract_info(input_src, download=False) or {}
            _query_title  = str(_info.get('title',    '') or '')
            _query_artist = str(_info.get('uploader', '') or _info.get('channel', '') or '')
            print(f"   🎬 YouTube: {_query_title} — {_query_artist}")
        except Exception as _e:
            print(f"   ⚠️  Could not fetch YouTube metadata: {_e}")

        _set_progress(session_id, 'downloading', f'Mengunduh audio dari YouTube…')
        _t0 = time.perf_counter()
        downloaded = download_youtube(input_src)
        _timing['download_sec'] = round(time.perf_counter() - _t0, 2)
        if downloaded is None:
            raise RuntimeError(
                "Gagal download audio dari YouTube.\n"
                "Kemungkinan penyebab:\n"
                "  • yt-dlp perlu diperbarui → jalankan: pip install -U yt-dlp\n"
                "  • Video tidak tersedia atau dibatasi di region ini\n"
                "  • Koneksi internet tidak stabil\n"
                "Coba gunakan tab 'File Audio' dan upload file secara langsung."
            )
        input_src = downloaded
                                                                    
        if session_id and session_id in _SESSION_STORE:
            _SESSION_STORE[session_id]['query_path'] = downloaded
    else:
                                                                 
        _set_progress(session_id, 'normalizing', 'Menormalisasi audio yang diunggah…')
        _t0 = time.perf_counter()
        _tmp_norm = normalize_query_audio(input_src)
        _timing['normalize_sec'] = round(time.perf_counter() - _t0, 2)
        if _tmp_norm != input_src:
            input_src = _tmp_norm
                                                        
            if session_id and session_id in _SESSION_STORE:
                _SESSION_STORE[session_id]['query_path'] = _tmp_norm

    _set_progress(session_id, 'extracting', 'Mengekstrak fitur CQT dari audio…')
    _t0 = time.perf_counter()
    core = melody_query_core(
        input_src, model, index, meta_db,
        top_k=top_k, n_candidates=n_candidates, min_score=min_score,
        progress_cb=lambda stage, detail='': _set_progress(session_id, stage, detail),
    )
    _timing['query_core_sec'] = round(time.perf_counter() - _t0, 2)
    _timing['cqt_extract_sec'] = core.get('cqt_sec')
    _timing['embed_infer_sec'] = core.get('infer_sec')
    _timing['total_sec']      = round(time.perf_counter() - _t_total_start, 2)

    _cfg.DEVICE = _orig_dev
    _mf.DEVICE  = _orig_dev

    results_per_seg = core["results_per_seg"]
    seg_meta        = core["seg_meta"]
    query_key       = core["query_key"]
    key_conf        = core["key_conf"]
    query_bpm       = core.get("query_bpm", 0.0)
    query_notes     = core["query_notes"]
    warnings        = core["warnings"]

    segments_out = []
    skipped_out  = []

    for i, seg in enumerate(seg_meta):
        res_list = results_per_seg.get(i, [])
        good     = [r for r in res_list if r["embed_sim"] >= min_score][:top_k]

        q_notes  = np.asarray(
            query_notes.get(i, np.array([], dtype=np.int8)),
            dtype=np.int8
        ).ravel()

        q_start = float(seg.get("start_time", 0.0))
        q_end   = float(seg.get("end_time",   q_start + 15.0))

        if not good:
            best_info = None
            if res_list:
                best  = res_list[0]
                smeta = meta_db.get(best["song_id"], {})
                best_info = {
                    "title":  smeta.get("title",  f"Song {best['song_id']}"),
                    "artist": smeta.get("artist", "Unknown"),
                    "score":  round(float(best["embed_sim"]) * 100, 1),
                }
            skipped_out.append({
                "seg_id":     i + 1,
                "start":      round(q_start, 2),
                "end":        round(q_end,   2),
                "time_label": f"{format_time(q_start)} – {format_time(q_end)}",
                "n_notes":    int(q_notes.size),
                "reason": (
                    f"Skor tertinggi {best_info['score']:.0f}% "
                    f"({best_info['title']}) — di bawah threshold {min_score*100:.0f}%"
                ) if best_info else "Tidak ada kandidat yang ditemukan",
                "best_candidate": best_info,
            })
            continue

        matches_out = []
        for rank, res in enumerate(good, start=1):
            s_id  = res["song_id"]
            smeta = meta_db.get(s_id, {})
            title = smeta.get("title", f"Song {s_id}")
            ver   = res["version"]
            artist = (smeta.get("artist_cover", "") if ver == "cover"
                      else smeta.get("artist_original", ""))
            if not artist:
                artist = smeta.get("artist", "Unknown")

            m_key = str(res.get("global_key", "C Major") or "C Major")
            _KEY_SEMI = {"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,
                         "F":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,
                         "A#":10,"Bb":10,"B":11}
            _q_root = query_key.split()[0] if query_key and query_key != "Unknown" else ""
            _m_root = m_key.split()[0]     if m_key   and m_key   != "Unknown"  else ""
            _q_semi = _KEY_SEMI.get(_q_root, -1)
            _m_semi = _KEY_SEMI.get(_m_root, -1)
            if _q_semi >= 0 and _m_semi >= 0:
                _diff    = abs(_q_semi - _m_semi)
                key_diff = min(_diff, 12 - _diff)
            else:
                key_diff = -1

            matches_out.append({
                "rank":      rank,
                "song_id":   s_id,
                "title":     title,
                "artist":    artist,
                "version":   ver,
                "ver_label": "versi cover" if ver == "cover" else "versi asli",
                "m_start":   round(float(res["start_time"]), 2),
                "m_end":     round(float(res["end_time"]),   2),
                "m_time":    f"{format_time(res['start_time'])} – {format_time(res['end_time'])}",
                "m_key":     m_key,
                "embed_pct": round(float(res["embed_sim"]) * 100, 1),
                "key_diff":  key_diff,
            })

        segments_out.append({
            "seg_id":     i + 1,
            "start":      round(q_start, 2),
            "end":        round(q_end,   2),
            "time_label": f"{format_time(q_start)} – {format_time(q_end)}",
            "matches":    matches_out,
        })

    return {
        "query_key":        query_key,
        "query_bpm":        query_bpm,
        "key_confidence":   round(float(key_conf), 3),
        "total_segments":   len(seg_meta),
        "matched_segments": len(segments_out),
        "skipped_segments": skipped_out,
        "segments":         segments_out,
        "warnings":         warnings,
        "query_title":      _query_title,
        "query_artist":     _query_artist,
                                                                              
        "device_used":      device_key,
        "device_label":     "GPU (CUDA)" if device_key == "cuda" else "CPU",
        "timing_sec":       _timing,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default="127.0.0.1")
    parser.add_argument("--port",  type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"\nStarting Melody Similarity Web Server")
    print(f"  URL : http://{args.host}:{args.port}")
    print(f"  Press Ctrl+C to stop\n")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)