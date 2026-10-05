# Sleep Stage Study

Sleep Stage Study is an experimental machine-learning pipeline for sleep-stage
classification using the [Sleep-EDF Expanded](https://physionet.org/content/sleep-edfx/1.0.0/)
polysomnography dataset. The repository compares bandpower-based classical
models with compact temporal neural networks and generates interpretation and
error-analysis artifacts.

## What the project does

1. Pairs each Sleep Cassette PSG recording with its hypnogram.
2. Converts annotations into 30-second labeled signal epochs.
3. Standardizes the signals and extracts Welch bandpower features for delta,
   theta, alpha, and beta bands.
4. Evaluates logistic regression, random forest, and XGBoost with grouped
   five-fold cross-validation.
5. Trains two lightweight raw-signal models: `TinyCNN` and `TinyTCN`.
6. Produces SHAP explanations, a confusion matrix, and a consolidated model
   comparison.

The current consolidated feature set contains 5,258 epochs from 153 recordings,
with 3,698 wake (`W`) epochs and 1,560 REM epochs. Therefore, the checked-in
summary below describes the current **W-vs-REM binary experiment**, not a full
five-class sleep-staging benchmark.

## Current results

| Model | Family | Macro-F1 | Cohen's kappa | Inference (ms/epoch) | Parameters |
|---|---|---:|---:|---:|---:|
| XGBoost | Classical | 0.907 ± 0.020 | 0.814 ± 0.039 | — | — |
| Random Forest | Classical | 0.878 ± 0.022 | 0.757 ± 0.044 | — | — |
| TinyTCN | Deep | 0.841 ± 0.040 | 0.683 ± 0.080 | 0.988 | 42,597 |
| TinyCNN | Deep | 0.840 ± 0.042 | 0.682 ± 0.082 | 0.165 | 3,653 |
| Logistic Regression | Classical | 0.478 ± 0.018 | 0.083 ± 0.020 | — | — |

These values are copied from `results/model_summary.csv`. Earlier multiclass
and channel-ablation experiments are preserved separately in `results/` and
should not be compared directly with the binary table.

## Repository layout

```text
src/sleep_stage_study.py         Consolidated seven-phase pipeline
sleep_stage_study.ipynb          Main exploratory notebook
experiments/                     Earlier notebooks and exported scripts
results/                         Small tables and evaluation figures
shap_figs/                       SHAP summary and class-level plots
features/                        Generated feature arrays (ignored by Git)
derived_sc/                      Per-recording epoch archives (ignored by Git)
sleep-edf-database-expanded-1.0.0/  Local raw dataset (ignored by Git)
```

## Setup

Create an environment and install the dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Download Sleep-EDF Expanded and extract it so the project contains:

```text
sleep-edf-database-expanded-1.0.0/
└── sleep-cassette/
    ├── *-PSG.edf
    └── *-Hypnogram.edf
```

Run the consolidated script from the repository root:

```bash
python src/sleep_stage_study.py
```

The `__main__` block at the bottom of the script controls which phases run.
Raw-data preprocessing and deep-model training are intentionally disabled by
default because they are expensive; the result-export phase is enabled.

## Methodology notes

- Raw EDF files and generated NumPy archives are intentionally excluded from
  Git because the local dataset is about 13 GB and individual artifacts exceed
  GitHub's file-size limit.
- The pipeline currently uses recording-derived group identifiers. Before
  reporting strict subject-independent generalization, verify the Sleep-EDF
  subject-key extraction and regenerate all cross-validation results.
- The consolidated preprocessing code defines five stage labels, while the
  checked-in feature arrays contain only `W` and `REM`. Regenerate the derived
  data before presenting the repository as a five-class study.

