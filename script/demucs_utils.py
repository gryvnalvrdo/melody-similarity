import numpy as np
import torch

                                                                                 
                                                                 
                                               
_DEMUCS_MODEL_NAME = "htdemucs"

                                                          
_demucs_model_cache = None
_demucs_device_cache = None


def _get_demucs_model():
    """Load (dan cache) model Demucs. Return (model, device)."""
    global _demucs_model_cache, _demucs_device_cache
    
    import sys
    cfg_device = None
    if 'script.config' in sys.modules:
        cfg_device = sys.modules['script.config'].DEVICE
    elif 'config' in sys.modules:
        cfg_device = sys.modules['config'].DEVICE
    else:
        cfg_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_device = cfg_device

    if _demucs_model_cache is not None:
        if _demucs_device_cache != target_device:
            try:
                _demucs_model_cache.to(target_device)
                _demucs_device_cache = target_device
            except RuntimeError as e:
                if 'out of memory' in str(e).lower() and target_device.type == 'cuda':
                    print("   ⚠️  Demucs OOM on GPU → falling back to CPU")
                    torch.cuda.empty_cache()
                    target_device = torch.device('cpu')
                    _demucs_model_cache.to(target_device)
                    _demucs_device_cache = target_device
                else:
                    raise
        return _demucs_model_cache, _demucs_device_cache

    try:
        from demucs.pretrained import get_model
    except ImportError:
        raise ImportError(
            "Demucs tidak ditemukan. Install dengan: pip install demucs"
        )

    device = target_device
    model = get_model(_DEMUCS_MODEL_NAME)
    model.eval()

    try:
        model.to(device)
    except RuntimeError as e:
                                                    
        if 'out of memory' in str(e).lower() and device.type == 'cuda':
            print("   ⚠️  Demucs OOM on GPU → falling back to CPU (slower but safe)")
            torch.cuda.empty_cache()
            device = torch.device('cpu')
            model.to(device)
        else:
            raise

    _demucs_model_cache = model
    _demucs_device_cache = device
    return model, device


def _run_demucs(audio: np.ndarray, sr: int) -> dict:
    try:
        from demucs.apply import apply_model
    except ImportError:
        raise ImportError("Demucs tidak terinstall.")

    model, device = _get_demucs_model()
    target_sr = model.samplerate                       

                                                                                
    import torchaudio
    if sr != target_sr:
        audio_t = torch.from_numpy(audio).float()
        if audio_t.dim() == 1:
            audio_t = audio_t.unsqueeze(0)          
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio_t = resampler(audio_t)
    else:
        audio_t = torch.from_numpy(audio).float()
        if audio_t.dim() == 1:
            audio_t = audio_t.unsqueeze(0)

                                                            
    if audio_t.shape[0] == 1:
        audio_t = audio_t.expand(2, -1)

                                    
    audio_t = audio_t.unsqueeze(0).to(device)

    with torch.no_grad():
                                                         
        sources = apply_model(
            model, audio_t,
            device=device,
            shifts=0,          # 0 = no test-time augmentation, ~2x faster, quality still fine for CQT
            split=True,        # split long audio into chunks to avoid OOM
            overlap=0.25,
            progress=False,
        )

                                                                        
                                      
    stem_names = model.sources                                        
    sources_np = sources[0].cpu().numpy()             

    result = {}
    for i, name in enumerate(stem_names):
                                 
        stem_mono = sources_np[i].mean(axis=0)        

                                                
        if target_sr != sr:
            stem_t = torch.from_numpy(stem_mono).float().unsqueeze(0)
            resamp_back = torchaudio.transforms.Resample(target_sr, sr)
            stem_mono = resamp_back(stem_t).squeeze(0).numpy()

        result[name] = stem_mono.astype(np.float32)

    return result


def remove_drums(audio: np.ndarray, sr: int) -> np.ndarray:
    try:
        stems = _run_demucs(audio, sr)

                                                       
        audio_no_drums = (
            stems.get("bass",  np.zeros_like(audio)) +
            stems.get("other", np.zeros_like(audio)) +
            stems.get("vocals", np.zeros_like(audio))
        )

                                              
        audio_no_drums = np.clip(audio_no_drums, -1.0, 1.0)

                                                                                         
        min_len = min(len(audio), len(audio_no_drums))
        audio_no_drums = audio_no_drums[:min_len]

        return audio_no_drums.astype(np.float32)

    except Exception as e:
                                                                    
                                                                              
        if 'out of memory' in str(e).lower() or 'cuda error' in str(e).lower():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        print(f"   ⚠️  Demucs remove_drums gagal: {e} — menggunakan audio original")
        return None


def separate_vocals(audio: np.ndarray, sr: int) -> np.ndarray:
    try:
        stems = _run_demucs(audio, sr)
        vocals = stems.get("vocals")
        if vocals is None:
            return None

              
        vocals = np.clip(vocals, -1.0, 1.0)

                      
        min_len = min(len(audio), len(vocals))
        return vocals[:min_len].astype(np.float32)

    except Exception as e:
        print(f"   ⚠️  Demucs separate_vocals gagal: {e}")
        return None


def separate_melody_stems(audio: np.ndarray, sr: int) -> np.ndarray:
    return remove_drums(audio, sr)


def clear_cache():
    """Bebaskan model Demucs dari memori (berguna setelah batch processing)."""
    global _demucs_model_cache, _demucs_device_cache
    if _demucs_model_cache is not None:
        del _demucs_model_cache
        _demucs_model_cache = None
        _demucs_device_cache = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
