"""Bat-ball contact from sound: a sharp, broadband, high-frequency transient ("crack")."""
import numpy as np
from scipy.signal import find_peaks


def onset_strength(y, sr, hop=256, n_fft=1024, fmin=1500.0):
    """High-frequency spectral flux, returned as robust z-score per hop."""
    if y is None or len(y) < n_fft * 4:
        return None, None
    win = np.hanning(n_fft).astype(np.float32)
    n_hops = 1 + (len(y) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_hops)[:, None]
    frames = y[idx] * win
    mag = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mag = np.log1p(100.0 * mag[:, freqs >= fmin])
    flux = np.maximum(np.diff(mag, axis=0, prepend=mag[:1]), 0).sum(axis=1)
    med = np.median(flux)
    mad = np.median(np.abs(flux - med)) + 1e-9
    z = (flux - med) / (1.4826 * mad)
    t = (np.arange(n_hops) * hop + n_fft / 2) / sr
    return t, z


def onsets(y, sr, z_min=6.0, min_sep_s=0.25):
    """List of (time_s, z) of strong transients."""
    t, z = onset_strength(y, sr)
    if t is None:
        return []
    dt = t[1] - t[0]
    pk, _ = find_peaks(z, height=z_min, distance=max(1, int(min_sep_s / dt)))
    return [(float(t[i]), float(z[i])) for i in pk]
