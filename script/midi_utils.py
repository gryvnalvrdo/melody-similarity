"""
midi_utils.py — Synthesis & plotting utilities for melody proof generation.
Falls back gracefully if optional dependencies (scipy, matplotlib) are missing.
"""
import numpy as np


def is_synthesis_available() -> bool:
    """Return True if WAV synthesis from MIDI notes is supported."""
    try:
        import scipy.io.wavfile              
        return True
    except ImportError:
        return False


def is_plot_available() -> bool:
    """Return True if matplotlib pitch-contour plotting is supported."""
    try:
        import matplotlib              
        return True
    except ImportError:
        return False


def notes_to_wav(notes: np.ndarray, output_path: str,
                 bpm: float = 120.0, sr: int = 22050) -> bool:
    """
    Synthesize a simple piano-like melody from a MIDI note array and write to WAV.

    Parameters
    ----------
    notes       : 1-D array of MIDI note numbers (int8 / int)
    output_path : destination .wav file path
    bpm         : tempo for note duration estimate
    sr          : sample rate (must match the rest of the pipeline)

    Returns True on success, False on failure / missing deps.
    """
    if not is_synthesis_available():
        return False
    if notes is None or len(notes) == 0:
        return False
    try:
        import librosa
        import scipy.io.wavfile

        note_dur = (60.0 / bpm) * 0.5                                         
        n_samples = int(sr * note_dur)
        t = np.linspace(0.0, note_dur, n_samples, endpoint=False)
        envelope = np.exp(-4.0 * t / note_dur)                

        segments = []
        for midi in notes:
            freq = float(librosa.midi_to_hz(int(midi)))
            wave = 0.35 * np.sin(2.0 * np.pi * freq * t) * envelope
                                                   
            wave += 0.10 * np.sin(4.0 * np.pi * freq * t) * envelope
            segments.append(wave)

        audio = np.concatenate(segments)
        audio_i16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
        scipy.io.wavfile.write(output_path, sr, audio_i16)
        return True
    except Exception:
        return False


def generate_comparison_plot(
    q_notes: np.ndarray,
    m_notes: np.ndarray,
    q_key: str = "C Major",
    m_key: str = "C Major",
    output_path: str = "comparison.png",
    title: str = "Melody Comparison",
) -> bool:
    """
    Generate a side-by-side pitch-contour comparison PNG.

    Returns True on success, False on failure / missing deps.
    """
    if not is_plot_available():
        return False
    if q_notes is None or m_notes is None:
        return False
    if len(q_notes) == 0 or len(m_notes) == 0:
        return False
    try:
        import matplotlib
        matplotlib.use("Agg")                                   
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=False)
        fig.suptitle(title or "Melody Comparison", fontsize=11, fontweight="bold")

        axes[0].plot(q_notes, "b-o", markersize=3, linewidth=1.5,
                     label=f"Query  ({q_key})")
        axes[0].set_ylabel("MIDI Note")
        axes[0].set_title("Query")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(m_notes, "r-o", markersize=3, linewidth=1.5,
                     label=f"Match  ({m_key})")
        axes[1].set_ylabel("MIDI Note")
        axes[1].set_title("Match")
        axes[1].legend(loc="upper right", fontsize=8)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception:
        return False
