# Roadmap — Audio Speech Emotion Classification

Companion to [README.md](README.md). Each phase has a goal, the tasks, the concepts it teaches, and a "done when" checkpoint. Work top to bottom; don't skip the checkpoints.

---

## What the brief tells us (and why it shapes the plan)

| Fact from the brief | Consequence for us |
|---|---|
| Training data (TESS + RAVDESS) is **not** on Kaggle | We download both datasets ourselves and build our own labelled training table. |
| Test set = 201 unlabeled clips, a **mix of TESS and RAVDESS** | Our model must handle both sources: different speakers, recording setups and sample rates (TESS ≈ 24.4 kHz, RAVDESS 48 kHz). |
| TESS has 7 emotions, RAVDESS has 8 (adds **calm**), and names differ (`ps` / "pleasant surprise" vs "surprised", "happiness" vs "happy") | We need one unified label set, and its spellings must match `Sample_Submission.csv` exactly. |
| Metric is **mean F1**, averaging not stated | Optimise and report **macro F1** (every emotion counts equally). That's the stricter reading. |
| Course goal: **ensemble learning** on extracted features | The core of the project is classical ML: hand-crafted audio features → individual classifiers → ensembles. Deep learning is a stretch goal. |
| Learning objective: test on **your own voice** | The final phase is recording yourself and seeing how the model handles a new mic, a new room and a new speaker. |

**Two cautions to keep in mind throughout:**

1. **Speaker leakage.** TESS has only 2 speakers, and RAVDESS has 24. A random train/validation split puts the same voice on both sides, and validation scores come out inflated. Where possible, validate on **held-out RAVDESS actors**.
2. **Test clips are probably taken from the public datasets.** That means a model that memorises training files could score near-perfectly on the leaderboard without having learned emotion. Treat the leaderboard as a sanity check. Your speaker-held-out validation score is the honest number, and it's the one worth reporting.

---

## Phase 0 — Setup

**Goal:** a working environment and all the data on disk.

- [x] Use the existing `audio-kagcomp` virtual environment (Python 3.15). Every library below publishes 3.15 packages.
- [x] Install the pinned packages in [requirements.txt](requirements.txt): `pip install -r requirements.txt`.
- [x] Download with the Kaggle CLI (both training downloads contain an identical nested duplicate; we ignore it):
  - Kaggle test set: `MP4_Kaggle_Testset/` and `Sample_Submission.csv`
  - RAVDESS **audio speech only** (`Audio_Speech_Actors_01-24`, 1,440 files)
  - TESS (2,880 files)
- [x] Create the project layout:
  ```
  data/raw/{ravdess,tess,kaggle_test}/
  data/preprocessed/     # feature tables, metadata CSVs
  notebooks/             # exploration
  src/                   # reusable code
  submissions/
  ```

**Done when:** every file loads, and you can list the exact label strings used in `Sample_Submission.csv`.

---

## Phase 1 — Build the metadata table

**Goal:** one CSV with one row per training clip: `path, dataset, speaker, gender, emotion, intensity, statement`.

- [x] Parse the RAVDESS filenames using the 7-part code (see README). Keep only modality 03 and vocal channel 01.
- [x] Parse the TESS labels from its folder and file names (e.g. `OAF_back_angry.wav`). The speaker is `OAF` or `YAF`.
- [x] Map both datasets onto the **submission label set**. Decide what to do with RAVDESS **calm**: keep it as its own class if the test labels include it, otherwise merge it into neutral or drop it.
- [x] Count clips per class, per dataset and per speaker.

**Results** (script: [src/build_metadata.py](src/build_metadata.py); output: `data/preprocessed/metadata.csv`, `test_metadata.csv`):
- Submission labels are 7 classes: `angry, disgust, fear, happy, neutral, sad, surprised`. There's no calm. (`Sample_Submission.csv` also contains one typo, `nuetral`; we always write `neutral`.)
- RAVDESS **calm** (192 clips) is kept in the table, but its `label` is empty, so it's excluded from training for now. Phase 4 will test merging it into neutral.
- 4,048 training clips: 592 per class, except neutral with 496 (RAVDESS has only 96 neutral clips).
- TESS is 2,800 clips from 2 female speakers, 69% of the training data. The model will see far more TESS voice than RAVDESS voice.
- Data quirks handled: one TESS file is misnamed (`OA_bite_neutral.wav`), and TESS folder spellings are inconsistent, so the emotion is read from the filename.

**Concepts:** structured labels from filenames, label harmonisation, class imbalance (for example, RAVDESS has half as many neutral clips as other emotions).

**Done when:** you have a clean `metadata.csv` and a table of class counts, and you can explain any imbalance.

---

## Phase 2 — Explore the audio

**Goal:** build intuition for the data before modelling.

- [x] Load a few clips per emotion, from each dataset, and **listen to them**.
- [x] Plot the waveform, the log-mel spectrogram and the pitch contour for contrasting emotions (e.g. angry vs sad).
- [x] Check the distributions of sample rate, clip duration, leading and trailing silence, and loudness, split by dataset.
- [x] Do the same checks on the Kaggle test clips. Do they look like TESS, like RAVDESS, or like a mix?

**Concepts:** sample rate, waveform vs spectrogram, the mel scale, pitch (F0), energy, and the differences between the two datasets.

**Done when:** you can describe in plain words how anger, sadness and neutral differ acoustically, and how TESS differs from RAVDESS.

**Results** (notebook: [notebooks/02_explore.ipynb](notebooks/02_explore.ipynb); per-file stats: `data/preprocessed/audio_stats.csv`):

*How emotions differ.* These patterns were checked across many clips, not single examples:
- **Pitch:** angry, happy, fear and surprised are clearly higher than neutral (male RAVDESS: neutral 120 Hz, angry 276 Hz). Sad stays low, close to neutral. Pitch *variation* doesn't separate the emotions reliably, except that TESS neutral is acted almost monotone (0.6 semitones, vs 2.6 in RAVDESS).
- **Loudness:** angry is the loudest emotion in both datasets. Sad and neutral are the quietest.
- **Speed:** sad (and disgust) speech is longer than neutral once silence is trimmed (RAVDESS 1.92 s vs 1.59 s; TESS 2.33 s vs 2.02 s).
- **Gender outweighs emotion for raw pitch.** Women's neutral (≈200 Hz) is close to men's angry (≈276 Hz), so pitch is only meaningful relative to each speaker's normal voice.

*How the datasets differ.*

| | RAVDESS | TESS |
|---|---|---|
| Sample rate | 48,000 Hz (5 files stereo) | 24,414 Hz (1 file at 96 kHz) |
| Clip length | 3.7 s | 2.1 s |
| Silence in clip | ≈51% | ≈4% |
| Background noise floor | ≈ −105 dB (near digital silence) | ≈ −53 dB (audible) |

*The test set.* 121 clips are TESS (24,414 Hz). 80 clips are **RAVDESS resampled to 16 kHz**: they match RAVDESS's duration, silence and noise floor. That's 60% TESS, 40% RAVDESS.

*The main lesson: confounding.* Pooled data suggests that "neutral has the least silence", but within each dataset it doesn't. In RAVDESS neutral has the *most* silence. The pooled pattern appears only because 81% of neutral clips come from TESS, which has almost no silence. Anything that identifies the dataset (sample rate, silence, noise, clip length) can become a shortcut that leads the model to the wrong label.

*What this decided for Phase 3:* resample to 16 kHz (the lowest test rate; above-8 kHz content can't exist in the test RAVDESS clips), convert to mono, trim leading and trailing silence, normalize loudness, and run the identical pipeline on test clips.

---

## Phase 3 — Preprocessing and features

**Goal:** turn every clip into a fixed-length feature vector.

- [x] Standardise every clip: convert to mono, resample to 16,000 Hz, trim silence, and normalise loudness.
- [x] Extract frame-level features:
  - MFCCs and their deltas
  - chroma
  - spectral centroid, bandwidth, roll-off and contrast
  - zero-crossing rate
  - RMS energy
  - pitch (F0)
- [x] Summarise each feature over time (mean, std, min, max), which gives one vector per clip.
- [x] Cache the result to `data/preprocessed/features.csv` so you never re-extract unnecessarily.
- [x] Apply the **exact same pipeline** to the test clips.

**Concepts:** framing and windowing, why we summarise over time, and why features need consistent preprocessing between training and test.

**Done when:** you have a feature table for training and one for test, with matching columns.

**Results** (script: [src/features.py](src/features.py); output: `features.csv`, `test_features.csv`; about 80 s on 8 cores):
- **Cleaning:** mono, 16 kHz, trim leading and trailing audio more than 30 dB below the peak, then scale every clip to −20 dB RMS. Analysis uses 25 ms frames every 10 ms (the speech standard).
- **258 features per clip:**
  - mean/std/min/max of: 20 MFCCs, 20 MFCC deltas, 12 chroma bins, 6 spectral-contrast bands, centroid, bandwidth, roll-off, zero-crossing rate, RMS
  - 4 pitch features: median Hz, plus std, 10–90% range and voiced fraction, with variation in semitones relative to the clip's own median
  - speech duration
  - `raw_loudness_db`, the loudness before normalization. It's kept separately so Phase 4 can test whether it helps or only leaks dataset identity.
- **Missing values:** 5 RAVDESS clips (strong intensity, actors 09 and 21) have no pitch, because pYIN found no voiced frames. These get imputed inside the model pipeline.
- **Dataset fingerprints survive cleaning.** A logistic regression can tell TESS from RAVDESS with **99%** accuracy from these features (97% from MFCCs alone), and it labels all 201 test clips' source correctly. Pitch features alone are at chance (67%). The remaining signature is the microphone, the room and the speakers themselves, which spectral features capture. This doesn't hurt the leaderboard, since the test clips come from the same two datasets. It does matter for generalization (Phase 7 and the own-voice test in Phase 8).

---

## Phase 4 — Validation strategy (do this before any modelling)

**Goal:** a score you can trust.

- [x] Hold out a group of **RAVDESS actors** (e.g. 4–5 actors, balanced by gender) as validation, using grouped cross-validation by speaker.
- [x] Decide how to handle TESS. With only 2 speakers, a true speaker split isn't possible. Options: hold out **one target word set**, or report TESS and RAVDESS validation scores separately.
- [x] Implement the metric: **macro F1**, plus the per-class F1 and a confusion matrix.
- [x] Add a dummy baseline that always predicts the majority class, as a floor any real model must beat.
- [x] Put missing-value imputation (5 clips with no pitch) and scaling **inside** the model pipeline, so they're fitted on training folds only.
- [x] Report scores **per dataset** (TESS and RAVDESS separately) as well as overall, because the features can identify the dataset.
- [x] Decide on `calm` (drop vs merge into neutral) and `raw_loudness_db` (keep vs drop) by comparing validation scores.

**Concepts:** data leakage, grouped cross-validation, why accuracy misleads on imbalanced classes, and how to read a confusion matrix.

**Done when:** one function takes a model and returns macro F1 and a confusion matrix on speaker-held-out data.

**Results** (module: [src/validation.py](src/validation.py); notebook: [notebooks/04_validation.ipynb](notebooks/04_validation.ipynb)):
- **Folds:** 5-fold `GroupKFold`. RAVDESS is grouped by **actor** (4–5 unseen actors per fold, about 50/50 by gender). TESS is grouped by **word** (about 40 unseen words per fold). No group ever appears on both sides of a split.
  - *Why:* with a random split, the same actor appears in both training and validation. The model can then partly recognize the **voice** instead of the emotion, and the score is inflated (leakage).
  - TESS can't be speaker-held-out (2 speakers), so its score is optimistic and is reported separately.
- **Metric:** macro F1, overall and per dataset. Calm is never scored. `fold_std` (spread across folds) is the noise ruler: a difference smaller than it is not real.

| Model | Macro F1 | RAVDESS | TESS |
|---|---|---|---|
| Dummy (always the most common class) | 0.036 | 0.038 | 0.036 |
| Dummy (random, in proportion) | 0.157 | 0.164 | 0.151 |
| Logistic regression (reference) | 0.823 ± 0.020 | **0.490** | 0.966 |

- **RAVDESS (0.49) is the honest score** for unseen speakers. TESS (0.97) is inflated by familiarity with the speakers.
- **Main confusions:** happy ↔ surprised (both high-pitched and lively) and neutral → sad (both low, flat and quiet). The model confuses emotions that sound alike, which matches Phase 2.
- **Decisions:** every change was about 10× smaller than `fold_std`, so none was measurable. In a tie we choose the simpler and safer option:

| Decision | Change in F1 | Choice | Reason |
|---|---|---|---|
| calm → neutral | +0.001 | **drop calm** | no gain; simpler |
| drop `raw_loudness_db` | +0.002 | **drop** | no gain; it's a recording/dataset fingerprint |
| drop `speech_s` | −0.001 | **drop** | no gain; partly reflects each dataset's sentence length |

- **Reference for Phase 5** (calm dropped, both features dropped): logistic regression **0.824 overall, 0.489 RAVDESS, 0.967 TESS**.

---

## Phase 5 — Individual classifiers

**Goal:** understand how each model family behaves on these features.

- [x] Put feature scaling inside a pipeline, so the scaler is fitted only on the training folds.
- [x] Train and compare:
  - logistic regression
  - k-NN
  - SVM (RBF kernel)
  - decision tree
  - random forest
- [x] Do light hyperparameter tuning, always inside the grouped cross-validation.
- [x] Make a **first Kaggle submission** with the best single model, and check that the file format is accepted.

**Concepts:** bias vs variance, why scaling matters (for SVM and k-NN especially), and overfitting to the validation set.

**Done when:** you have a comparison table of the models on macro F1 and one accepted submission.

**Results** (notebook: [notebooks/05_classifiers.ipynb](notebooks/05_classifiers.ipynb); submission: [submissions/audiemo_phase5_svm_rbf.csv](submissions/audiemo_phase5_svm_rbf.csv)):

| Model (tuned) | Best settings | Macro F1 | `fold_std` | RAVDESS | TESS |
|---|---|---|---|---|---|
| **SVM (RBF)** | `C=10`, `gamma=scale` | **0.858** | 0.016 | **0.538** | 0.995 |
| Logistic regression | `C=0.1` | 0.836 | 0.020 | 0.500 | 0.981 |
| Random forest | `max_features=sqrt`, `min_samples_leaf=1` | 0.832 | 0.022 | 0.441 | 0.997 |
| k-NN | `n_neighbors=15`, `weights=distance` | 0.819 | 0.014 | 0.414 | 0.995 |
| Decision tree | `max_depth=8`, `min_samples_leaf=1` | 0.738 | 0.012 | 0.306 | 0.921 |

- **Best single model: SVM (RBF).** It beats the reference (0.824 overall, 0.489 RAVDESS) and has the highest RAVDESS score. Its lead over logistic regression is 0.022, about the size of `fold_std`, so it's a modest win rather than a decisive one.
- **Scaling:** removing the scaler drops SVM from 0.854 to 0.234 and k-NN from 0.803 to 0.527, because both decide by distance. Random forest stays at 0.832, because trees compare one feature at a time.
- **Bias vs variance:** at depth 2 the decision tree scores about 0.25 on both training and validation data (underfitting). Validation peaks near 0.74 at depth 8, while training keeps climbing to 1.0 by depth 16 (overfitting).
- **Tuning added little:** only k-NN's gain (+0.016) exceeds its `fold_std` (0.014), and only just. SVM gained 0.004 and random forest gained nothing.
- **Hardest emotions:** per-emotion F1 for the SVM runs from 0.818 (sad) to 0.888 (surprised). The largest single confusion is happy → fear (0.06).
- **First Kaggle submission made** with the SVM, retrained on all 4,048 clips. The leaderboard score isn't recorded here yet.

---

## Phase 6 — Ensembles (the core assignment)

**Goal:** beat the best single model by combining models.

- [ ] **Bagging:** random forest and extra trees, i.e. many models trained on resampled data.
- [ ] **Boosting:** gradient boosting, XGBoost or LightGBM, where each new model corrects the previous ones' mistakes.
- [ ] **Voting:** hard and soft voting across your best diverse models.
- [ ] **Stacking:** a meta-model trained on the base models' out-of-fold predictions.
- [ ] Analyse which emotions improved, and whether the ensemble just repeats its strongest member or really combines different models.

**Concepts:** why diverse, uncorrelated errors make ensembles work, and soft vs hard voting.

**Done when:** you have an ensemble that beats the best single model on speaker-held-out macro F1, and a submission made with it.

---

## Phase 7 — Error analysis and iteration

**Goal:** understand *why* the model fails.

- [ ] Find the most confused pairs in the confusion matrix (commonly calm vs neutral and fear vs surprise) and listen to the misclassified clips.
- [ ] Measure **cross-dataset generalisation**: train on RAVDESS and test on TESS, then the reverse. Expect a large drop, and work out why.
- [ ] Find which features matter most (permutation importance).
- [ ] Try targeted fixes, e.g. data augmentation (noise, pitch shift, time stretch), feature selection, or per-speaker normalisation.

**Done when:** you can name the model's three main failure modes and give evidence for each.

---

## Phase 8 — Test on your own voice

**Goal:** see how the model holds up on new audio.

- [ ] Record yourself saying a short sentence in several emotions, several takes each.
- [ ] Run your recordings through the **exact same preprocessing and feature pipeline**.
- [ ] Compare the predictions with what you intended. Mic, room noise, accent and your own acting style are all things the model has never seen.

**Concepts:** domain shift, and why acted-emotion datasets generalise poorly to everyday speech.

**Done when:** you have a short write-up of how the model did on your voice and why.

---

## Phase 9 (stretch) — Deep learning

Only after Phases 0–8 are done.

- [ ] A small CNN on log-mel spectrograms, with SpecAugment.
- [ ] Embeddings from a pretrained speech model (wav2vec2, HuBERT or WavLM) fed into a logistic regression or small MLP.
- [ ] Compare these against your best ensemble on the same speaker-held-out split.

---

## Phase 10 — Write-up

- [ ] Summary: data, validation method, models, results table, confusion matrix, failure modes, own-voice test, and next steps.
- [ ] Report the honest validation score next to the leaderboard score, and explain the gap between them.
