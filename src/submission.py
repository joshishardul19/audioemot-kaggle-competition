"""Load the Kaggle test features and write submission files in the required format.

Usage (from a notebook in notebooks/):
    from submission import load_test_data, write_submission
    X_test, ids = load_test_data(X.columns)
    write_submission(ids, model.predict(X_test), "phase5_svm")
"""

import pandas as pd

from validation import LABELS, PREPROCESSED, PROJECT_ROOT

SUBMISSIONS = PROJECT_ROOT / "submissions"
N_TEST = 201


def load_test_data(columns):
    """Test features with exactly the training columns, in the same order, plus the clip Ids."""
    df = pd.read_csv(PREPROCESSED / "test_features.csv")
    return df[list(columns)], df["Id"]


def write_submission(ids, pred, name):
    """Write submissions/<name>.csv with columns Id,Label and check it before Kaggle does."""
    sub = pd.DataFrame({"Id": ids, "Label": pred}).sort_values("Id")
    assert len(sub) == N_TEST and sub["Id"].is_unique, f"expected {N_TEST} unique Ids"
    assert sub["Label"].isin(LABELS).all(), f"unknown labels: {set(sub['Label']) - set(LABELS)}"
    SUBMISSIONS.mkdir(exist_ok=True)
    path = SUBMISSIONS / f"{name}.csv"
    sub.to_csv(path, index=False)
    return path
