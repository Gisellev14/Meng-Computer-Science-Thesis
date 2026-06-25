"""
PaySim benchmark experiment.

Loads the PaySim synthetic transaction log, engineers features from the raw
transaction data, draws a class-balanced sample, and runs the same
ALGORITHM_CATALOGUE through run_training_pipeline with binary labels.

Label encoding (binary path):
  0  LEGITIMATE  — isFraud == 0
  1  FRAUD       — isFraud == 1

The existing binary path in run_training_pipeline handles threshold tuning
(T_low / T_op / T_high) and the three-tier decision system automatically.

Feature engineering
-------------------
  type                  — transaction type (categorical, OHE inside pipeline)
  amount                — transaction amount
  oldbalanceOrg         — origin balance before transaction
  newbalanceOrig        — origin balance after transaction
  oldbalanceDest        — destination balance before transaction
  newbalanceDest        — destination balance after transaction
  balance_orig_diff     — oldbalanceOrg - newbalanceOrig  (should equal amount)
  balance_dest_diff     — newbalanceDest - oldbalanceDest (should equal amount)
  error_orig            — amount - balance_orig_diff  (0 for normal transactions)
  error_dest            — amount - balance_dest_diff  (0 for normal transactions)
  hour_of_day           — step % 24  (cyclic hour within simulation)

Sampling
--------
  All fraud rows (8,213) + 10× random non-fraud sample (~82,130) = ~90K total.
  This keeps training scale similar to the main fraud_cleaned.csv experiment.
  class_weight='balanced' on sklearn models and weighted CrossEntropyLoss on
  the GRU handle residual imbalance within the sampled set.

Usage
-----
  python run_paysim_experiment.py                        # all algorithms
  python run_paysim_experiment.py --algo random_forest   # single algorithm
  python run_paysim_experiment.py --no-grid-search       # fast smoke-test
  python run_paysim_experiment.py --fraud-ratio 5        # tighter class ratio
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as sklearn_auc

from training_pipeline import ALGORITHM_CATALOGUE, run_training_pipeline

DATA_PATH    = Path(__file__).parent / "data" / "PS_20174392719_1491204439457_log.csv"
RESULTS_DIR  = Path(__file__).parent / "results" / "paysim"

DECISION_NAMES = ["ALLOW", "REVIEW", "DENY"]


# ------------------------------------------------------------------ #
#  Result-saving helpers                                              #
# ------------------------------------------------------------------ #


def save_results(results: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    algos  = list(results.keys())
    colors = plt.cm.tab10.colors

    # ── 1. Summary CSV ──────────────────────────────────────────────
    rows = []
    for algo, r in results.items():
        rows.append({
            "algorithm": algo,
            "roc_auc":   round(r["roc_auc"], 4),
            "pr_auc":    round(r["pr_auc"],  4),
            "t_low":     r.get("review_threshold"),
            "t_op":      r.get("best_threshold"),
            "t_high":    r.get("deny_threshold"),
        })
    pd.DataFrame(rows).to_csv(results_dir / "summary.csv", index=False)

    # ── 2. Performance bar chart ─────────────────────────────────────
    x = np.arange(len(algos)); width = 0.35
    roc_vals = [results[a]["roc_auc"] for a in algos]
    pr_vals  = [results[a]["pr_auc"]  for a in algos]
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width/2, roc_vals, width, label="ROC-AUC", color="steelblue")
    b2 = ax.bar(x + width/2, pr_vals,  width, label="PR-AUC",  color="darkorange")
    for bar, val in list(zip(b1, roc_vals)) + list(zip(b2, pr_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(algos, rotation=15, ha="right")
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title("Algorithm Performance — PaySim Benchmark")
    ax.legend(); ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(results_dir / "performance_comparison.png", dpi=150)
    plt.close(fig)

    # ── 3. Decision distribution stacked bar chart ───────────────────
    dec_colors = {"ALLOW": "#4878d0", "REVIEW": "#ee854a", "DENY": "#d65f5f"}
    dec_fracs = {}
    for dec in DECISION_NAMES:
        dec_fracs[dec] = []
        for a in algos:
            counts = results[a]["decision_counts"]
            total  = sum(counts.values())
            dec_fracs[dec].append(counts.get(dec, 0) / total if total else 0)
    fig, ax = plt.subplots(figsize=(9, 5))
    bottoms = np.zeros(len(algos))
    for dec in DECISION_NAMES:
        vals = np.array(dec_fracs[dec])
        bars = ax.bar(x, vals, 0.5, bottom=bottoms, label=dec, color=dec_colors[dec])
        for bar, val, bot in zip(bars, vals, bottoms):
            if val > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                        f"{val:.1%}", ha="center", va="center", fontsize=8,
                        color="white", fontweight="bold")
            elif val > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2, bot + val / 2,
                        f"{val:.1%}", ha="center", va="center", fontsize=7,
                        color="white")
        bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels(algos, rotation=15, ha="right")
    ax.set_ylim(0, 1.08); ax.set_ylabel("Fraction of test set")
    ax.set_title("Three-Tier Decision Distribution — PaySim Benchmark")
    ax.legend(loc="upper right"); ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(results_dir / "decision_distribution.png", dpi=150)
    plt.close(fig)

    # ── 4. PR curves (binary) ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (algo, r) in enumerate(results.items()):
        prec = r.get("precision_curve")
        rec  = r.get("recall_curve")
        if prec is not None and rec is not None:
            ax.plot(rec, prec, color=colors[i % 10],
                    label=f"{algo} (AP={r['pr_auc']:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.set_title("Precision-Recall Curves — PaySim Benchmark")
    ax.legend(fontsize=8); ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(results_dir / "pr_curves.png", dpi=150)
    plt.close(fig)

    # ── 5. ROC curves (binary) ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for i, (algo, r) in enumerate(results.items()):
        y_test   = r.get("y_test")
        y_scores = r.get("y_scores")
        if y_test is not None and y_scores is not None:
            fpr, tpr, _ = roc_curve(y_test, y_scores)
            ax.plot(fpr, tpr, color=colors[i % 10],
                    label=f"{algo} (AUC={r['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — PaySim Benchmark"); ax.legend(fontsize=8)
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(results_dir / "roc_curves.png", dpi=150)
    plt.close(fig)

    # ── 6. Per-algorithm confusion matrices & feature importance ─────
    for algo, r in results.items():
        cm = r["confusion_matrix"]
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        class_names = ["LEGITIMATE", "FRAUD"]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, data, fmt, subtitle in zip(
            axes, [cm, cm_norm],
            ["{:,}", "{:.1%}"],
            ["Counts", "Row-normalised (Recall per class)"],
        ):
            im = ax.imshow(data, interpolation="nearest", cmap="Blues",
                           vmin=0, vmax=(None if data is cm else 1.0))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ticks = range(len(class_names))
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            ax.set_xticklabels(class_names, rotation=15, ha="right", fontsize=9)
            ax.set_yticklabels(class_names, fontsize=9)
            thresh = data.max() / 2.0
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                            color="white" if data[i, j] > thresh else "black", fontsize=10)
            ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
            ax.set_title(subtitle)
        fig.suptitle(f"Confusion Matrix — {algo} (PaySim)", fontsize=11)
        fig.tight_layout()
        fig.savefig(results_dir / f"confusion_matrix_{algo}.png", dpi=150)
        plt.close(fig)
        if r.get("feature_importance") is not None:
            fi = r["feature_importance"]
            top = fi.head(15).iloc[::-1]
            fig2, ax2 = plt.subplots(figsize=(9, max(4, 15 * 0.38)))
            bars = ax2.barh(top["feature"], top["importance"], color="steelblue")
            for bar in bars:
                w = bar.get_width()
                ax2.text(w + top["importance"].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                         f"{w:.4f}", va="center", ha="left", fontsize=7)
            ax2.set_xlabel("Feature Importance")
            ax2.set_title(f"Top 15 Feature Importances — {algo} (PaySim)")
            ax2.set_xlim(0, top["importance"].max() * 1.18)
            fig2.tight_layout()
            fig2.savefig(results_dir / f"feature_importance_{algo}.png", dpi=150)
            plt.close(fig2)

    # ── 7. Score distributions (fraud vs legitimate) ──────────────────
    non_dummy = [a for a in algos if a != "dummy"]
    if non_dummy:
        fig, axes = plt.subplots(1, len(non_dummy), figsize=(6 * len(non_dummy), 4),
                                  squeeze=False)
        for col, algo in enumerate(non_dummy):
            r = results[algo]
            y_test   = r.get("y_test")
            y_scores = r.get("y_scores")
            if y_test is None or y_scores is None:
                continue
            ax = axes[0][col]
            for label, name, color in [(0, "Legitimate", "#4878d0"), (1, "Fraud", "#d65f5f")]:
                mask = y_test == label
                ax.hist(y_scores[mask], bins=60, alpha=0.6, label=f"{name} (n={mask.sum():,})",
                        color=color, density=True)
            t_high = r.get("deny_threshold")
            t_low  = r.get("review_threshold")
            if t_high is not None:
                ax.axvline(t_high, color="black", linestyle="--", linewidth=1.2,
                           label=f"T_high={t_high:.2f}")
            if t_low is not None and t_low != t_high:
                ax.axvline(t_low, color="gray", linestyle=":", linewidth=1.2,
                           label=f"T_low={t_low:.2f}")
            ax.set_xlabel("p(Fraud)")
            ax.set_ylabel("Density")
            ax.set_title(algo)
            ax.legend(fontsize=8)
        fig.suptitle("Fraud Score Distributions — PaySim Benchmark", fontsize=12)
        fig.tight_layout()
        fig.savefig(results_dir / "score_distributions.png", dpi=150)
        plt.close(fig)

    print(f"\nPaySim results saved to {results_dir}/")
    saved = sorted(p.name for p in results_dir.glob("*.png")) + \
            sorted(p.name for p in results_dir.glob("*.csv"))
    for f in saved:
        print(f"  {f}")

NUMERIC_COLS = [
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "balance_orig_diff",
    "balance_dest_diff",
    "error_orig",
    "error_dest",
    "hour_of_day",
]

FEATURE_COLS = ["type"] + NUMERIC_COLS   # 'type' stays as string → OHE inside pipeline


def load_and_prep(path: Path, fraud_ratio: int = 10, random_state: int = 42) -> pd.DataFrame:
    """Load PaySim, engineer features, return a class-balanced sample.

    Parameters
    ----------
    fraud_ratio : int
        Non-fraud rows per fraud row. Default 10 → ~9.1% fraud in sample.
    random_state : int
        Controls the non-fraud down-sample.
    """
    print(f"Loading {path} ...")
    df = pd.read_csv(path)
    print(f"  Full dataset: {len(df):,} rows, {df['isFraud'].sum():,} fraud "
          f"({df['isFraud'].mean()*100:.2f}%)")

    # --- Feature engineering ---
    df["balance_orig_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["balance_dest_diff"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["error_orig"]        = df["amount"] - df["balance_orig_diff"]
    df["error_dest"]        = df["amount"] - df["balance_dest_diff"]
    df["hour_of_day"]       = df["step"] % 24

    # --- Binary label (0=LEGITIMATE, 1=FRAUD) ---
    df["label"] = df["isFraud"].astype(int)

    # --- Class-balanced sample: all fraud + fraud_ratio × non-fraud ---
    fraud_df = df[df["isFraud"] == 1]
    legit_df = df[df["isFraud"] == 0]

    n_fraud       = len(fraud_df)
    n_legit_sample = n_fraud * fraud_ratio

    legit_sample = legit_df.sample(n=n_legit_sample, random_state=random_state)
    df_sample = (
        pd.concat([fraud_df, legit_sample], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    fraud_rate = df_sample["label"].mean()
    print(f"  Sampled dataset: {len(df_sample):,} rows "
          f"(fraud={n_fraud:,}, legit={n_legit_sample:,}, "
          f"fraud rate={fraud_rate:.4f})")

    return df_sample


def main():
    parser = argparse.ArgumentParser(description="PaySim benchmark experiment")
    parser.add_argument(
        "--algo",
        choices=list(ALGORITHM_CATALOGUE.keys()),
        default=None,
        help="Run a single algorithm (default: all)",
    )
    parser.add_argument(
        "--no-grid-search",
        action="store_true",
        help="Skip GridSearchCV for a quick smoke-test",
    )
    parser.add_argument(
        "--fraud-ratio",
        type=int,
        default=10,
        help="Non-fraud rows per fraud row in sample (default: 10)",
    )
    args = parser.parse_args()

    df = load_and_prep(DATA_PATH, fraud_ratio=args.fraud_ratio)

    print(f"\n  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"  Label distribution:")
    labels = {0: "LEGITIMATE", 1: "FRAUD"}
    for v, name in labels.items():
        cnt = (df["label"] == v).sum()
        print(f"    {v} ({name:<12}) {cnt:>8,}  ({cnt / len(df) * 100:.2f}%)")

    algos    = [args.algo] if args.algo else list(ALGORITHM_CATALOGUE.keys())
    run_grid = not args.no_grid_search

    results = {}
    for algo in algos:
        print(f"\n{'='*70}")
        print(f"Running: {algo}")
        print(f"{'='*70}")
        results[algo] = run_training_pipeline(
            df=df,
            feature_cols=FEATURE_COLS,
            target_col="label",
            model_name=f"PaySim_{algo}",
            algorithm=algo,
            run_grid_search=run_grid,
        )

    # --- Summary table ---
    print(f"\n{'='*70}")
    print("PAYSIM RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"{'Algorithm':<25}  {'ROC-AUC':>8}  {'PR-AUC':>8}  "
          f"{'T_low':>7}  {'T_op':>7}  {'T_high':>7}")
    print("-" * 68)
    for algo, r in results.items():
        t_low  = r.get("review_threshold") or 0.0
        t_op   = r.get("best_threshold")   or 0.0
        t_high = r.get("deny_threshold")   or 0.0
        print(
            f"{algo:<25}  {r['roc_auc']:>8.4f}  {r['pr_auc']:>8.4f}  "
            f"{t_low:>7.2f}  {t_op:>7.2f}  {t_high:>7.2f}"
        )

    save_results(results, RESULTS_DIR)

    return results


if __name__ == "__main__":
    main()
