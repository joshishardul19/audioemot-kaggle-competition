"""Phase 4: one trustworthy way to score any model.

Every model in Phases 5-7 is scored with `evaluate`, so results are comparable.

Validation design:
    RAVDESS  each actor is one group, so validation actors are never seen in training.
    TESS     only 2 speakers, so a speaker hold-out is impossible. Each target word is one
             group instead: validation words are never seen in training. Because the speakers
             are seen, TESS scores are optimistic and are reported separately.
    calm     never scored, because the test set has no calm label. Calm clips can still be
             used for training (see load_training_data).

Usage (from a notebook in notebooks/ or a script in src/):
    from validation import load_training_data, evaluate, pipeline
    X, y, meta = load_training_data()
    scores, pred = evaluate(pipeline(LogisticRegression()), X, y, meta, name="logreg")
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREPROCESSED = PROJECT_ROOT / "data" / "preprocessed"

LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprised"]
META_COLS = ["path", "dataset", "speaker", "gender", "emotion", "label", "text"]
N_FOLDS = 5

# Phase 4 decision: neither feature improved the score measurably, and both partly identify
# the dataset (recording level and sentence length), so we leave them out from Phase 5 on.
DROP_FEATURES = ["raw_loudness_db", "speech_s"]


def load_training_data(calm="drop", drop_features=()):
    """Return features X, labels y and metadata for training.

    calm:          "drop" excludes RAVDESS calm clips; "neutral" trains on them as neutral.
    drop_features: feature columns to leave out (e.g. ["raw_loudness_db"]).
    """
    df = pd.read_csv(PREPROCESSED / "features.csv")
    df = df.merge(pd.read_csv(PREPROCESSED / "metadata.csv")[["path", "text"]], on="path")
    if calm == "neutral":
        df.loc[df["emotion"] == "calm", "label"] = "neutral"
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    X = df.drop(columns=[*META_COLS, *drop_features])
    return X, df["label"], df[META_COLS]


def cv_groups(meta):
    """Group id per clip: the actor for RAVDESS, the target word for TESS."""
    return np.where(meta["dataset"] == "ravdess", meta["speaker"], "tess_" + meta["text"])


def folds(meta):
    """The 5 (train_index, validation_index) splits used everywhere. Deterministic."""
    return list(GroupKFold(n_splits=N_FOLDS).split(meta, groups=cv_groups(meta)))


def pipeline(model):
    """Wrap a model so imputation and scaling are fitted on training folds only."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)


def _fit_predict(model, X, y, train_idx, val_idx):
    model = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
    return val_idx, model.predict(X.iloc[val_idx])


def evaluate(model, X, y, meta, name="model"):
    """Cross-validate a model and return (scores, out-of-fold predictions).

    scores holds macro F1 overall, per dataset and the std across folds. Calm clips are
    never scored. Out-of-fold means every prediction comes from a model that did not see
    that clip (or its speaker / word) during training.
    """
    splits = folds(meta)
    results = Parallel(n_jobs=-1)(delayed(_fit_predict)(model, X, y, tr, va) for tr, va in splits)
    pred = pd.Series(index=y.index, dtype=object)
    for val_idx, fold_pred in results:
        pred.iloc[val_idx] = fold_pred

    scored = meta["emotion"] != "calm"
    fold_f1 = [f1_score(y.iloc[va][scored.iloc[va]], pred.iloc[va][scored.iloc[va]], average="macro")
               for _, va in splits]
    scores = {"model": name,
              "macro_f1": f1_score(y[scored], pred[scored], average="macro"),
              "fold_std": np.std(fold_f1)}
    for dataset in ["ravdess", "tess"]:
        mask = scored & (meta["dataset"] == dataset)
        scores[f"{dataset}_f1"] = f1_score(y[mask], pred[mask], average="macro")
    return scores, pred


def report(y, pred, meta, title=""):
    """Per-class precision/recall/F1 and a confusion matrix (rows = true, normalised)."""
    scored = meta["emotion"] != "calm"
    print(classification_report(y[scored], pred[scored], labels=LABELS, digits=3, zero_division=0))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y[scored], pred[scored], labels=LABELS, normalize="true",
                                            values_format=".2f", cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
