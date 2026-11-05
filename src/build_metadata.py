"""Phase 1: build one metadata table describing every training and test clip.

Reads only file and folder names (no audio is loaded) and writes:
    data/preprocessed/metadata.csv       one row per training clip (RAVDESS + TESS)
    data/preprocessed/test_metadata.csv  one row per Kaggle test clip

Run from the project root:
    python src/build_metadata.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW = PROJECT_ROOT / "data" / "raw"
OUT = PROJECT_ROOT / "data" / "preprocessed"

# Both Kaggle downloads contain an identical nested copy of the data (verified by
# checksum), so we point at the top-level copy only and never search recursively.
RAVDESS_DIR = RAW / "ravdess"
TESS_DIR = RAW / "tess" / "TESS Toronto emotional speech set data"
TEST_DIR = RAW / "kaggle_test" / "MP4_Kaggle_Testset" / "MP4_Kaggle_Testset"

# The 7 labels used in Sample_Submission.csv. Our predictions must use exactly these strings.
LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]

# RAVDESS emotion codes (3rd field of the filename) -> emotion name.
RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fear", "07": "disgust", "08": "surprised",
}

# TESS emotion suffixes (last part of the filename) -> emotion name.
TESS_EMOTIONS = {
    "angry": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy",
    "neutral": "neutral", "ps": "surprised", "sad": "sad",
}


def ravdess_rows():
    """Parse RAVDESS filenames like 03-01-06-01-02-01-12.wav (see README for the code table)."""
    rows = []
    for path in sorted(RAVDESS_DIR.glob("Actor_*/*.wav")):
        modality, channel, emotion, intensity, statement, repetition, actor = path.stem.split("-")
        assert (modality, channel) == ("03", "01"), f"not audio-only speech: {path.name}"
        rows.append({
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "dataset": "ravdess",
            "speaker": f"ravdess_{actor}",
            "gender": "male" if int(actor) % 2 == 1 else "female",
            "emotion": RAVDESS_EMOTIONS[emotion],
            "intensity": "normal" if intensity == "01" else "strong",
            "text": "kids" if statement == "01" else "dogs",
            "repetition": int(repetition),
        })
    return rows


def tess_rows():
    """Parse TESS files like OAF_Fear/OAF_back_fear.wav.

    The speaker comes from the folder name because one file is misnamed (OA_ instead of OAF_).
    The emotion comes from the filename because folder spellings are inconsistent
    (OAF_Fear vs YAF_fear, OAF_Pleasant_surprise vs YAF_pleasant_surprised).
    """
    rows = []
    for path in sorted(TESS_DIR.glob("*_*/*.wav")):
        speaker = path.parent.name.split("_")[0]  # OAF (older) or YAF (younger)
        _, word, emotion = path.stem.split("_")
        rows.append({
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "dataset": "tess",
            "speaker": f"tess_{speaker}",
            "gender": "female",
            "emotion": TESS_EMOTIONS[emotion.lower()],
            "intensity": None,
            "text": word.lower(),
            "repetition": 1,
        })
    return rows


def main():
    train = pd.DataFrame(ravdess_rows() + tess_rows())

    # `label` is what we train on and submit. RAVDESS "calm" has no submission label, so it is
    # left empty (excluded from training) by default; we can test merging it into neutral later.
    train["label"] = train["emotion"].where(train["emotion"].isin(LABELS))

    test = pd.DataFrame({"path": sorted(TEST_DIR.glob("*.wav"), key=lambda p: int(p.stem))})
    test.insert(0, "Id", [int(p.stem) for p in test["path"]])
    test["path"] = [p.relative_to(PROJECT_ROOT).as_posix() for p in test["path"]]

    # Sanity checks: fail loudly if the data is not what we expect.
    assert len(train.query("dataset == 'ravdess'")) == 1440, "expected 1440 RAVDESS clips"
    assert len(train.query("dataset == 'tess'")) == 2800, "expected 2800 TESS clips"
    assert train["path"].is_unique
    assert len(test) == 201, "expected 201 test clips"

    OUT.mkdir(parents=True, exist_ok=True)
    train.to_csv(OUT / "metadata.csv", index=False)
    test.to_csv(OUT / "test_metadata.csv", index=False)

    print(f"Wrote {len(train)} training rows and {len(test)} test rows to {OUT.relative_to(PROJECT_ROOT)}/\n")
    print("Clips per emotion and dataset:")
    print(pd.crosstab(train["emotion"], train["dataset"], margins=True), "\n")
    print("Clips per speaker:")
    print(train.groupby(["dataset", "gender"])["speaker"].agg(["nunique", "count"]), "\n")
    print("Training label distribution (calm excluded):")
    print(train["label"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
