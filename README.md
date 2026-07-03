# Fraud Detection in Loan Applications: A Multi-Algorithm Approach with Cross-Dataset Validation

Master's Thesis in Computer Science

## Overview

This project builds and evaluates a fraud detection pipeline for loan applications. Fraud is formulated as a **three-class classification problem** — NO_FRAUD_DECISION / FRAUD_SUSPECT / CONFIRM_FRAUD — mapping directly to an operational **ALLOW / REVIEW / DENY** decision system with probability-based threshold calibration.

Features are drawn from two distinct operational data sources per application: transactional signals (loan amounts, credit profile, application context) and behavioral signals (device fingerprint, IP geolocation, rule engine scores). The primary dataset contains **91,984 real-world loan application records**. All models are additionally benchmarked on **PaySim**, a publicly available synthetic mobile money transaction dataset, as a cross-domain generalisation test.

## Repository Structure

```
thesis/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
│
├── data_cleaning.py               # JSON extraction, PII masking, feature engineering, imputation
├── training_pipeline.py           # ML pipeline: encode, train, threshold-tune, evaluate
├── run_experiment.py              # Primary dataset experiment (multi-seed, all algorithms)
├── run_paysim_experiment.py       # PaySim cross-domain benchmark
│
└── results/
    ├── class_distribution.png         # Label distribution with counts and percentages
    ├── performance_comparison.png     # ROC-AUC / PR-AUC bar chart per algorithm
    ├── decision_distribution.png      # ALLOW / REVIEW / DENY fraction per algorithm
    ├── score_distributions.png        # Fraud score distributions by true class
    ├── threshold_curves.png           # Precision / Recall / F1 vs threshold
    ├── pr_curves.png                  # Precision-Recall curves
    ├── roc_curves.png                 # ROC curves
    ├── confusion_matrix_{algo}.png    # Dual-panel: counts + row-normalised rates
    ├── feature_importance_{algo}.png  # Top feature importances with value labels
    ├── seed_stability.png             # Mean ± std across seeds (full run only)
    ├── summary.csv                    # Per-algorithm metrics (seed 42)
    ├── multi_seed_summary.csv         # Mean ± std across seeds 42, 123, 7
    └── paysim/                        # Equivalent outputs for PaySim benchmark
        ├── score_distributions.png    # Fraud score separation (explains threshold degeneration)
        └── ...
```

## Dataset

### Primary Dataset

Raw records must be placed at `../Fraud data/` (one level above `thesis/`). Each row contains three JSON blobs — `application_data`, `device_information_data`, and `fraud_decision_data` — extracted and cleaned by `data_cleaning.py`.

Run cleaning once before experiments:

```bash
python data_cleaning.py
```

Output: `fraud_cleaned.csv` — 91,984 rows × 42 columns.

### PaySim

Download `PS_20174392719_1491204439457_log.csv` from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it at `../Fraud data/PS_20174392719_1491204439457_log.csv`. The experiment script loads and samples it automatically.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.10+.

## Reproducing Results

### Primary Dataset (multi-seed, full grid search)

```bash
python run_experiment.py
```

Runs grid search on seed 42 (5-fold CV, PR-AUC scorer), then re-evaluates best parameters on seeds 42, 123, and 7. Reports mean ± std across seeds. **Expected runtime: several hours** (Gradient Boosting grid search is the bottleneck).

### Primary Dataset (quick, seed 42 only)

```bash
python run_experiment.py --no-grid-search
```

Uses fixed hyperparameters, single seed. Completes in minutes. Use this to regenerate figures without re-running grid search.

### PaySim Benchmark

```bash
python run_paysim_experiment.py                # full grid search
python run_paysim_experiment.py --no-grid-search   # quick figures only
```

All outputs are written to `results/paysim/`.

## Algorithms

| Algorithm | Type |
|---|---|
| `dummy` | Majority-class baseline |
| `logistic_regression` | Linear model |
| `random_forest` | Bagging ensemble |
| `gradient_boosting` | Boosting ensemble |

All sklearn models use `class_weight='balanced'`. Hyperparameter search uses `GridSearchCV` with 5-fold stratified CV scored on PR-AUC.

## Three-Tier Decision System

| Score condition | Decision | Operational action |
|---|---|---|
| Below T\_low | **ALLOW** | Auto-approve |
| T\_low ≤ score < T\_high | **REVIEW** | Route to fraud analyst |
| Score ≥ T\_high | **DENY** | Auto-reject |

Thresholds are tuned on the validation set: T\_high maximises precision ≥ 70% on CONFIRM\_FRAUD; T\_low is the highest threshold that retains recall ≥ 95% on any-fraud cases. On PaySim, tree ensemble thresholds degenerate (T\_low > T\_high), collapsing the system to binary ALLOW / DENY — a finding documented in the thesis as a diagnostic of fraud pattern simplicity.

## Key Results

| Algorithm | Primary ROC-AUC | Primary PR-AUC | PaySim ROC-AUC | PaySim PR-AUC |
|---|---|---|---|---|
| Dummy | 0.5000±0.0000 | 0.3333±0.0000 | 0.5000 | 0.0909 |
| Logistic Regression | 0.7441±0.0092 | 0.5417±0.0170 | 0.9886 | 0.9327 |
| Random Forest | 0.9962±0.0004 | 0.9835±0.0012 | **0.9994** | **0.9985** |
| **Gradient Boosting** | **0.9966±0.0002** | **0.9855±0.0009** | 0.9987 | 0.9928 |

Primary metrics are mean ± std across seeds 42, 123, 7. PaySim metrics are single-seed.

## Hardware

Experiments were run on Apple M-series CPU (macOS).
