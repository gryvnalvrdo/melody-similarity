import os
import numpy as np
import torch
from tqdm import tqdm
import pickle
from collections import defaultdict

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from config import (
        FEATURES_DIR, INDEX_DIR, MODELS_DIR, DEVICE, BATCH_SIZE,
        HDF5_DATASET_PATH, SEGMENT_HOP, SEGMENT_DURATION, SAMPLE_RATE, HOP_LENGTH
    )
    from model import MelodySimilarityModel
except ImportError:
    from script.config import (
        FEATURES_DIR, INDEX_DIR, MODELS_DIR, DEVICE, BATCH_SIZE,
        HDF5_DATASET_PATH, SEGMENT_HOP, SEGMENT_DURATION, SAMPLE_RATE, HOP_LENGTH
    )
    from script.model import MelodySimilarityModel

def load_model(model_path=None):
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = MelodySimilarityModel().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {model_path}")
    return model

def extract_embeddings(model, cqt_features):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(cqt_features), BATCH_SIZE):
            batch_data = cqt_features[i:i + BATCH_SIZE]
            batch_normalized = np.array(batch_data, dtype=np.float32)
            batch = torch.FloatTensor(batch_normalized).to(DEVICE)
            batch_emb = model(batch)
            embeddings.append(batch_emb.cpu().numpy())
    if not embeddings:
        return np.array([])
    return np.vstack(embeddings)

def _count_valid_segments(h5f, song_ids):
    total = 0
    for song_id in tqdm(song_ids, desc="  Pass 1/2 – counting segs", leave=False):
        grp = h5f[song_id]
        for version in ['original', 'cover', 'cover_2', 'cover_3']:
            if version not in grp:
                continue
            dset = grp[version]
            valid_mask = dset.attrs.get('valid_mask', None)
            if valid_mask is not None:
                total += int(np.sum(valid_mask))
            else:
                                                                                    
                total += dset.shape[0]
    return total

def build_index(model_path=None):
    print("=" * 70)
    print("BUILDING EMBEDDING INDEX  (memmap mode - OOM-safe)")
    print("=" * 70)
    model = load_model(model_path)
    use_h5 = H5PY_AVAILABLE and os.path.exists(HDF5_DATASET_PATH)
    print(f"Data source: {'HDF5 (' + HDF5_DATASET_PATH + ')' if use_h5 else 'NPY (' + FEATURES_DIR + ')'}")

    _hop_time = SEGMENT_HOP                                                                 
    all_metadata = []
    global_idx   = 0

    os.makedirs(INDEX_DIR, exist_ok=True)
    import tempfile
    mmap_path = os.path.join(tempfile.gettempdir(), "melody_sim_embeddings_tmp.mmap")

    emb_dim = 256                
    if use_h5:
        try:
            with h5py.File(HDF5_DATASET_PATH, 'r') as _hf:
                for _sid in sorted(_hf.keys()):
                    for _ver in ['original', 'cover']:
                        if _ver in _hf[_sid]:
                            _dummy = _hf[_sid][_ver][:1].astype(np.float32)
                            with torch.no_grad():
                                _t = torch.FloatTensor(_dummy).to(DEVICE)
                                emb_dim = model(_t).shape[-1]
                            break
                    break
        except Exception as e:
            print(f"   [!] Could not infer emb_dim ({e}), using default {emb_dim}")

    if use_h5:
        with h5py.File(HDF5_DATASET_PATH, 'r') as h5f:
            song_ids = sorted(h5f.keys())
            print(f"Found {len(song_ids):,} songs in HDF5")

            total_segs_est = _count_valid_segments(h5f, song_ids)
            est_gb = total_segs_est * emb_dim * 4 / 1e9
            print(f"   Estimated segments : {total_segs_est:,}")
            print(f"   Embedding dim      : {emb_dim}")
            print(f"   Memmap size (est.) : {est_gb:.2f} GB  ->  {mmap_path}")

            emb_mmap = np.memmap(mmap_path, dtype=np.float32, mode='w+',
                                 shape=(total_segs_est, emb_dim))

            write_ptr = 0
            for song_id in tqdm(song_ids, desc="  Pass 2/2 – extracting  "):
                grp = h5f[song_id]
                for version in ['original', 'cover', 'cover_2', 'cover_3']:
                    if version not in grp:
                        continue
                    dset       = grp[version]
                    n_segs     = dset.shape[0]
                    global_key = dset.attrs.get('global_key', 'Unknown')
                    cqt_data   = dset[:].astype(np.float32)
                    valid_mask = dset.attrs.get('valid_mask', None)

                    valid_indices = []
                    for seg_idx in range(n_segs):
                        if valid_mask is not None:
                            if valid_mask[seg_idx]:
                                valid_indices.append(seg_idx)
                        else:
                            if cqt_data[seg_idx].mean() > 1e-4:
                                valid_indices.append(seg_idx)

                    if not valid_indices:
                        continue

                    valid_cqt  = cqt_data[valid_indices]
                    embeddings = extract_embeddings(model, valid_cqt)                      
                    n          = len(valid_indices)

                    emb_mmap[write_ptr: write_ptr + n] = embeddings

                    for emb_i, seg_idx in enumerate(valid_indices):
                        all_metadata.append({
                            'song_id':     str(song_id),
                            'version':     version,
                            'segment_idx': seg_idx,
                            'start_time':  seg_idx * _hop_time,
                            'end_time':    seg_idx * _hop_time + SEGMENT_DURATION,
                            'global_idx':  global_idx,
                            'global_key':  str(global_key),
                        })
                        global_idx += 1

                    write_ptr += n

            actual_segs = write_ptr
            emb_mmap.flush()
            del emb_mmap                                                 

        print(f"\n   Extracted {actual_segs:,} segments total")

    if not all_metadata:
        print("No embeddings extracted!")
        _try_remove(mmap_path)
        return None

    emb_mmap = np.memmap(mmap_path, dtype=np.float32, mode='r',
                         shape=(actual_segs, emb_dim))

    print("Computing song-level embeddings (chunked)...")
    sv_index = defaultdict(list)                                          
    for idx, m in enumerate(all_metadata):
        sv_index[f"{m['song_id']}|{m['version']}"].append(idx)

    song_emb_list  = []
    song_meta_list = []
    for sv_key, indices in sv_index.items():
        chunk = emb_mmap[sorted(indices)]                                    
        mean  = np.mean(chunk, axis=0).astype(np.float32)
        norm  = np.linalg.norm(mean)
        if norm > 1e-8:
            mean /= norm
        sid, ver = sv_key.rsplit('|', 1)
        song_emb_list.append(mean)
        song_meta_list.append({'song_id': sid, 'version': ver})

    song_emb_array = np.array(song_emb_list, dtype=np.float32)

    print("Saving embeddings.npy ...")
    embeddings_npy_path = os.path.join(INDEX_DIR, "embeddings.npy")
    np.save(embeddings_npy_path, emb_mmap)                                      
    np.save(os.path.join(INDEX_DIR, "song_embeddings.npy"), song_emb_array)

    print("Saving embedding_index.pkl ...")
    embeddings_array = np.load(embeddings_npy_path)                                  
    index = {
        'embeddings':     embeddings_array,
        'metadata':       all_metadata,
        'song_ids':       list(set(m['song_id'] for m in all_metadata)),
        'song_embeddings': song_emb_array,
        'song_metadata':  song_meta_list,
    }
    index_path = os.path.join(INDEX_DIR, "embedding_index.pkl")
    with open(index_path, 'wb') as f:
        pickle.dump(index, f)

    try:
        import faiss as _faiss
        _fi = _faiss.IndexFlatIP(emb_dim)
        _fi.add(embeddings_array)
        _faiss.write_index(_fi, os.path.join(INDEX_DIR, 'faiss_seg.bin'))
        _fsi = _faiss.IndexFlatIP(emb_dim)
        _fsi.add(song_emb_array)
        _faiss.write_index(_fsi, os.path.join(INDEX_DIR, 'faiss_song.bin'))
        print("FAISS indexes saved.")
    except ImportError:
        pass

    del emb_mmap                                  
    _try_remove(mmap_path)

    print(f"Total segments: {len(all_metadata):,}, Unique songs: {len(index['song_ids']):,}")
    print(f"Index saved to: {index_path}")
    return index

def _try_remove(path):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"   [!] Could not remove temp file {path}: {e}")

def load_index(index_path=None):
    if index_path is None:
        index_path = os.path.join(INDEX_DIR, "embedding_index.pkl")
    with open(index_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()
    build_index(model_path=args.model)
