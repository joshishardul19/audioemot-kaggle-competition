"""Phase 3: clean every clip and turn it into one fixed-length feature vector.

Reads data/preprocessed/metadata.csv and test_metadata.csv (from build_metadata.py) and writes:
    data/preprocessed/features.csv       one row per training clip (calm included; filter later)
    data/preprocessed/test_features.csv  one row per Kaggle test clip

Run from the project root (takes a few minutes; pitch tracking is the slow part):
    python src/features.py
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED = PROJECT_ROOT / "data" / "preprocessed"


# Why is the sample rate 16 kHz? Why are we trimming at 30 dB? Why are we normalising to -20 dB? What is the significance of these numbers?
# Preprocessing settings, chosen from the Phase 2 findings.
SR = 16_000           # lowest sample rate in the test set; higher rates are resampled down to it
TOP_DB = 30           # trim leading/trailing audio more than 30 dB below the clip's peak
TARGET_RMS_DB = -20   # every clip is scaled to this average loudness

# Frame settings: 25 ms windows every 10 ms, the standard choice for speech.
# No clue what this is
N_FFT = 400
HOP = 160

# Pitch search range for adult speech. Wider ranges cause octave errors (seen in Phase 2).
F0_MIN, F0_MAX = 65, 600

# If most of the clips are mono channel like we saw in Phase 2, then why are we loading as mono?
def load_clean(path):
    """Load a clip as mono 16 kHz audio, trim silence and normalise loudness.

    Returns the cleaned audio and its loudness before normalisation, which we keep as a
    separate feature so later phases can test whether absolute loudness helps or leaks.
    """
    y, _ = librosa.load(PROJECT_ROOT / path, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    rms = np.sqrt(np.mean(y**2))
    # This formula needs explanantion and needs to be inserted in the report.
    raw_loudness_db = 20 * np.log10(rms + 1e-10)
    y = y * (10 ** (TARGET_RMS_DB / 20) / (rms + 1e-10))
    return y, raw_loudness_db


def summarise(name, frames):
    """Collapse a (n_rows, n_frames) feature matrix to mean/std/min/max per row."""
    frames = np.atleast_2d(frames)
    out = {}
    for i, row in enumerate(frames):
        key = name if len(frames) == 1 else f"{name}_{i:02d}"
        out.update({f"{key}_mean": row.mean(), f"{key}_std": row.std(),
                    f"{key}_min": row.min(), f"{key}_max": row.max()})
    return out


def pitch_features(y):
    """Pitch level, variation and voicing from pYIN. Variation is in semitones relative to
    the clip's own median, so it describes intonation rather than how deep the voice is."""
    f0, _, _ = librosa.pyin(y, sr=SR, fmin=F0_MIN, fmax=F0_MAX, frame_length=1024, hop_length=HOP)
    # What is this voiced variable? Why are we filtering out NaN values? What is the significance of 5?
    voiced = f0[~np.isnan(f0)]
    if len(voiced) < 5:
        return {"f0_median_hz": np.nan, "f0_std_st": np.nan, "f0_range_st": np.nan, "voiced_frac": 0.0}
    # This formula needs explanation and needs to be inserted in the report. st = Semitones
    st = 12 * np.log2(voiced / np.median(voiced))
    return {"f0_median_hz": np.median(voiced), "f0_std_st": st.std(),
            "f0_range_st": np.percentile(st, 90) - np.percentile(st, 10),
            "voiced_frac": len(voiced) / len(f0)}


def extract(path):
    # All these features need to be understood.
    """Full pipeline for one clip: clean, then compute every feature."""
    y, raw_loudness_db = load_clean(path)
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))  # magnitude spectrogram, shared below
    # What is MFCC? Why are we using 40 mels and 20 mfccs? What is the significance of these numbers?
    # Why is the S-S**2 and S=S for a few other features? What is the significance of these numbers?
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(librosa.feature.melspectrogram(S=S**2, sr=SR, n_mels=40)),
                                n_mfcc=20)

    feats = {"speech_s": len(y) / SR, "raw_loudness_db": raw_loudness_db}
    # What does the |= mean?
    feats |= summarise("mfcc", mfcc)
    feats |= summarise("mfcc_delta", librosa.feature.delta(mfcc))
    feats |= summarise("chroma", librosa.feature.chroma_stft(S=S**2, sr=SR, n_fft=N_FFT))
    # Need to understand this as well.
    # 5 bands keeps the top band edge (200 Hz * 2^5 = 6.4 kHz) below the 8 kHz Nyquist limit.
    feats |= summarise("contrast", librosa.feature.spectral_contrast(S=S, sr=SR, n_bands=5))
    feats |= summarise("centroid", librosa.feature.spectral_centroid(S=S, sr=SR))
    feats |= summarise("bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=SR))
    feats |= summarise("rolloff", librosa.feature.spectral_rolloff(S=S, sr=SR))
    feats |= summarise("zcr", librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP))
    feats |= summarise("rms_db", librosa.amplitude_to_db(librosa.feature.rms(S=S, frame_length=N_FFT)))
    feats |= pitch_features(y)
    return feats


def build(meta, out_path, key_cols):
    # Parallel building. How?
    rows = Parallel(n_jobs=-1)(delayed(extract)(p) for p in tqdm(meta["path"], desc=out_path.name))
    features = pd.concat([meta[key_cols].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    features.to_csv(out_path, index=False)
    return features


def main():
    train = build(pd.read_csv(PREPROCESSED / "metadata.csv"), PREPROCESSED / "features.csv",
                  ["path", "dataset", "speaker", "gender", "emotion", "label"])
    test = build(pd.read_csv(PREPROCESSED / "test_metadata.csv"), PREPROCESSED / "test_features.csv",
                 ["Id", "path"])

    feature_cols = [c for c in train.columns if c not in ("path", "dataset", "speaker", "gender", "emotion", "label")]
    assert feature_cols == [c for c in test.columns if c not in ("Id", "path")], "train/test columns differ"
    print(f"\n{len(feature_cols)} features | train {train.shape} | test {test.shape}")
    print("Missing values per column (non-zero only):")
    print(pd.concat([train[feature_cols], test[feature_cols]]).isna().sum().loc[lambda s: s > 0])


if __name__ == "__main__":
    main()
