# Audio Speech Emotion Classification — Competition Brief

> Source: https://www.kaggle.com/competitions/audio-speech-emotion-classification
> Captured: 29 Oct 2026. Details below are a faithful restatement of the competition pages (Overview, Data, Rules), not a verbatim copy.

## At a glance

| Field | Value |
|---|---|
| Title | Audio speech emotion classification |
| Subtitle | CDS-B1 Module 4 Mini Project-4 |
| Host | CDS Support |
| Type | Community Prediction Competition (course assignment) |
| Start | 15 Jun 2021 |
| Close | 21 Jun 2021 |
| Current status | Closed; late submissions accepted |
| Prizes | Kudos only — no points or medals |
| Participation | 77 entrants, 64 participants, 11 teams, 172 submissions |
| Tags | Custom Metric |

## Problem statement

Build a model that recognises emotion from speech, using **ensemble learning**.

Workflow described by the host:
1. Follow the instructions in the course notebook (not hosted on Kaggle).
2. Download the test data from the competition.
3. Generate predictions and submit them as a `submission.csv` file, following the host's Kaggle submission guide.
4. Once submitted, the mean F-score appears on the leaderboard.

## Learning objectives

- Extract features from audio data.
- Implement ML classification algorithms both individually and as ensembles to classify emotions.
- Record your own voice sample and test it against the trained model.

## Evaluation

Kaggle scores predictions against the ground truth using **Mean F1-Score**.

- Precision: $p = \dfrac{tp}{tp + fp}$
- Recall: $r = \dfrac{tp}{tp + fn}$
- F1: $F1 = 2 \cdot \dfrac{p \cdot r}{p + r}$

F1 weights precision and recall equally, so balanced performance on both is favoured over excelling at one and failing at the other.

*Note:* the page does not state whether the averaging is macro or weighted.

## Data

### Training sources (not hosted on Kaggle — obtain separately)

**TESS — Toronto Emotional Speech Set**
- 2,880 files.
- 200 target words, each spoken in the carrier phrase "Say the word ___".
- Two actresses, native English speakers, university educated, with musical training and normal-range hearing thresholds.
- 7 emotions: anger, disgust, fear, happiness, pleasant surprise, sadness, neutral.

**RAVDESS — Ryerson Audio-Visual Database of Emotional Speech and Song**
- 1,440 files (24 actors × 60 trials).
- 24 professional actors: 12 female, 12 male.
- Two lexically matched statements, neutral North American accent.
- Emotions: neutral, calm, happy, sad, angry, fearful, surprise, disgust.
- Each emotion at two intensities (normal, strong), except neutral (normal only).

### RAVDESS filename convention

Each filename is a 7-part numeric identifier, e.g. `03-01-06-01-02-01-12.wav`.

| Position | Field | Codes |
|---|---|---|
| 1 | Modality | 01 = full-AV, 02 = video-only, 03 = audio-only |
| 2 | Vocal channel | 01 = speech, 02 = song |
| 3 | Emotion | 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised |
| 4 | Intensity | 01 = normal, 02 = strong (no strong for neutral) |
| 5 | Statement | 01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door" |
| 6 | Repetition | 01 = 1st, 02 = 2nd |
| 7 | Actor | 01–24; odd = male, even = female |

Worked example — `03-01-06-01-02-01-12.wav`: audio-only, speech, fearful, normal intensity, "dogs" statement, 1st repetition, actor 12 (female).

### Files provided on Kaggle

| Item | Details |
|---|---|
| Total | 202 files, ~21.57 MB (wav, csv) |
| `MP4_Kaggle_Testset/` | 201 `.wav` test clips named by number (e.g. `7.wav`, `115.wav`); despite the folder name, audio only. Mix of RAVDESS and TESS clips (per data version notes). |
| `Sample_Submission.csv` | 2.05 kB; 201 rows + header; shows required format and header |
| Licence | Subject to competition rules |

Access to file previews and downloads requires signing in and accepting the competition rules.

## Submission format

File: `submission.csv`

| Column | Type | Meaning |
|---|---|---|
| `Id` | ID | Test clip number |
| `Label` | String | Predicted emotion |

- One row per test clip → 201 rows plus header.
- Exact label spellings must match those in `Sample_Submission.csv` (viewable after accepting the rules).

## Rules

The Rules page contains only Kaggle's default placeholder text (no cheating, apply yourself, have fun). No team-size, daily-submission or external-data limits are stated.

## Citation

CDS Support. *Audio speech emotion classification.* https://www.kaggle.com/competitions/audio-speech-emotion-classification, 2021. Kaggle.