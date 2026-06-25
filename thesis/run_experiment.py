"""
Wiring script: load fraud_cleaned.csv and run the training pipeline.

Label encoding
--------------
  0  NO_FRAUD_DECISION  — legitimate application
  1  FRAUD_SUSPECT      — suspicious behaviour, fraud not confirmed
  2  CONFIRM_FRAUD      — confirmed fraud

The three-tier decision system maps naturally:
  class 0 → ALLOW  |  class 1 → REVIEW  |  class 2 → DENY

Usage
-----
  python run_experiment.py                        # all algorithms
  python run_experiment.py --algo random_forest   # single algorithm
  python run_experiment.py --no-grid-search       # skip GridSearchCV (fast smoke-test)
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc as sklearn_auc
from sklearn.preprocessing import label_binarize

from training_pipeline import ALGORITHM_CATALOGUE, run_training_pipeline

DATA_PATH = Path(__file__).parent.parent / "Fraud data" / "fraud_cleaned.csv"

LABEL_MAP = {
    "NO_FRAUD_DECISION": 0,
    "FRAUD_SUSPECT":     1,
    "CONFIRM_FRAUD":     2,
}

FEATURE_COLS = [
    # Application context
    "application_status",
    "app_hour",
    "app_day_of_week",
    # Loan features
    "has_loan_data",
    "loan_term",
    "loan_total_amount",
    "loan_approved_amount",
    "loan_requested_amount",
    # Client features
    "has_cupo_data",
    "client_type",
    "client_cupo_total",
    "client_cupo_remaining",
    "client_delinquency_balance",
    "client_identity_verified",
    "client_num_previous_loans",
    "client_max_days_past_due",
    # Credit features
    "credit_pd_score_missing",
    "credit_pd_score",
    # Application context (continued)
    "channel",
    "product",
    "journey_name",
    "pre_approval",
    # Device features
    "device_result",
    "device_type",
    "device_os_family",
    "device_os_version",
    "device_is_new",
    "device_screen_width",
    "device_screen_height",
    "device_browser_type",
    "device_browser_language",
    "device_browser_timezone",
    "device_cookies_enabled",
    "device_isp",
    "device_ip_region",
    "device_ip_country_code",
    "device_ip_city_mismatch",
    "device_age_days",
    "device_reg_to_app_days",
    "device_rule_score",
    "device_rules_matched",
]

NUMERIC_COLS = [
    "loan_term", "loan_total_amount", "loan_approved_amount", "loan_requested_amount",
    "client_cupo_total", "client_cupo_remaining", "client_delinquency_balance",
    "client_num_previous_loans", "client_max_days_past_due", "credit_pd_score",
    "app_hour", "app_day_of_week",
    "device_os_version", "device_screen_width", "device_screen_height",
    "device_browser_timezone", "device_age_days", "device_reg_to_app_days",
    "device_rule_score", "device_rules_matched",
]


RESULTS_DIR    = Path(__file__).parent / "results"
SEEDS          = [42, 123, 7]   # seed 42 runs full grid search; 123 and 7 reuse best params
CLASS_NAMES    = ["NO_FRAUD_DECISION", "FRAUD_SUSPECT", "CONFIRM_FRAUD"]
DECISION_NAMES = ["ALLOW", "REVIEW", "DENY"]


# ------------------------------------------------------------------ #
#  Result-saving helpers                                              #
# ------------------------------------------------------------------ #

def _save_confusion_matrix(cm, class_names, algo, out_dir):
    # Show both raw counts and row-normalised rates side by side
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, fmt, title_suffix in zip(
        axes,
        [cm, cm_norm],
        ["{:,}", "{:.1%}"],
        ["Counts", "Row-normalised (Recall per class)"],
    ):
        im = ax.imshow(data, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=(None if data is cm else 1.0))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ticks = range(len(class_names))
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(class_names, fontsize=8)
        thresh = data.max() / 2.0
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        color="white" if data[i, j] > thresh else "black", fontsize=9)
        ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
        ax.set_title(title_suffix)
    fig.suptitle(f"Confusion Matrix — {algo} (Primary Dataset)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"confusion_matrix_{algo}.png", dpi=150)
    plt.close(fig)


def _save_feature_importance(fi, algo, out_dir, top_n=20):
    top = fi.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.38)))
    bars = ax.barh(top["feature"], top["importance"], color="steelblue")
    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + top["importance"].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left", fontsize=7)
    ax.set_xlabel("Feature Importance (mean decrease in impurity)")
    ax.set_title(f"Top {top_n} Feature Importances — {algo} (Primary Dataset)")
    ax.set_xlim(0, top["importance"].max() * 1.15)
    fig.tight_layout()
    fig.savefig(out_dir / f"feature_importance_{algo}.png", dpi=150)
    plt.close(fig)


def _save_class_distribution(df, label_map, out_dir):
    """Bar chart of label class distribution."""
    inv_map = {v: k for k, v in label_map.items()}
    counts = df["label_encoded"].value_counts().sort_index()
    labels = [inv_map[i] for i in counts.index]
    colors = ["#4878d0", "#ee854a", "#d65f5f"]
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts.values, color=colors[:len(labels)], edgecolor="white")
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.005,
                f"{cnt:,}\n({cnt/total:.1%})", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of applications")
    ax.set_title(f"Class Distribution — Primary Dataset (n={total:,})")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=150)
    plt.close(fig)


def _save_score_distributions(results, out_dir, class_names):
    """Predicted probability histograms split by true class (shows model separability)."""
    tree_algos = [a for a in results if a not in ("dummy",)]
    if not tree_algos:
        return
    colors = ["#4878d0", "#ee854a", "#d65f5f"]
    n_scores = 2  # p(CONFIRM_FRAUD) and 1-p(NO_FRAUD_DECISION)
    fig, axes = plt.subplots(len(tree_algos), n_scores,
                              figsize=(12, 4 * len(tree_algos)), squeeze=False)
    for row, algo in enumerate(tree_algos):
        r = results[algo]
        y_test   = r["y_test"]
        y_scores = r["y_scores"]
        if not (hasattr(y_scores, "ndim") and y_scores.ndim == 2):
            continue
        score_pairs = [
            (y_scores[:, 2], "p(CONFIRM_FRAUD)", "T_high", r.get("deny_threshold")),
            (1 - y_scores[:, 0], "1 - p(NO_FRAUD_DECISION)", "T_low", r.get("review_threshold")),
        ]
        for col, (scores, score_label, thresh_name, thresh_val) in enumerate(score_pairs):
            ax = axes[row][col]
            for cls_idx, cls_name in enumerate(class_names):
                mask = y_test == cls_idx
                ax.hist(scores[mask], bins=50, alpha=0.55, label=cls_name,
                        color=colors[cls_idx], density=True)
            if thresh_val is not None:
                ax.axvline(thresh_val, color="black", linestyle="--", linewidth=1.2,
                           label=f"{thresh_name}={thresh_val:.2f}")
            ax.set_xlabel(score_label)
            ax.set_ylabel("Density")
            ax.set_title(f"{algo} — {score_label}")
            ax.legend(fontsize=7)
    fig.suptitle("Score Distributions by True Class — Primary Dataset", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.png", dpi=150)
    plt.close(fig)


def _save_seed_stability(all_seed_results, algos, out_dir):
    """Mean ± std bar chart for ROC-AUC and PR-AUC across seeds."""
    if len(all_seed_results) < 2:
        return
    seeds = list(all_seed_results.keys())
    roc_means = [np.mean([all_seed_results[s][a]["roc_auc"] for s in seeds]) for a in algos]
    roc_stds  = [np.std( [all_seed_results[s][a]["roc_auc"] for s in seeds]) for a in algos]
    pr_means  = [np.mean([all_seed_results[s][a]["pr_auc"]  for s in seeds]) for a in algos]
    pr_stds   = [np.std( [all_seed_results[s][a]["pr_auc"]  for s in seeds]) for a in algos]

    x = np.arange(len(algos)); width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - width/2, roc_means, width, yerr=roc_stds, capsize=4,
                label="ROC-AUC", color="steelblue", alpha=0.85)
    b2 = ax.bar(x + width/2, pr_means,  width, yerr=pr_stds,  capsize=4,
                label="PR-AUC",  color="darkorange", alpha=0.85)
    for bars, means, stds in [(b1, roc_means, roc_stds), (b2, pr_means, pr_stds)]:
        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stds) * 1.5,
                    f"{mean:.4f}\n±{std:.4f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(algos, rotation=15, ha="right")
    ax.set_ylim(0, 1.12); ax.set_ylabel("Score")
    ax.set_title(f"Algorithm Stability Across {len(seeds)} Seeds "
                 f"({', '.join(str(s) for s in seeds)}) — Primary Dataset")
    ax.legend(); ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_dir / "seed_stability.png", dpi=150)
    plt.close(fig)


def _save_threshold_curves(results, algos, out_dir):
    """Precision, recall, F1 vs threshold with T_low/T_op/T_high markers."""
    from sklearn.metrics import precision_recall_curve
    fig, axes = plt.subplots(1, len(algos), figsize=(5 * len(algos), 4), squeeze=False)
    for col, algo in enumerate(algos):
        ax = axes[0][col]
        r = results[algo]
        y_test   = r["y_test"]
        y_scores = r["y_scores"]
        if not (hasattr(y_scores, "ndim") and y_scores.ndim == 2):
            ax.set_visible(False); continue
        # Use CONFIRM_FRAUD score for threshold curve (primary decision axis)
        scores = y_scores[:, 2]
        y_bin  = (y_test == 2).astype(int)
        prec, rec, thresh = precision_recall_curve(y_bin, scores)
        # Compute F1 (avoid division by zero)
        with np.errstate(invalid="ignore"):
            f1 = np.where((prec[:-1] + rec[:-1]) > 0,
                          2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1]), 0)
        ax.plot(thresh, prec[:-1], label="Precision", color="#d65f5f")
        ax.plot(thresh, rec[:-1],  label="Recall",    color="#4878d0")
        ax.plot(thresh, f1,        label="F1",        color="#6acc65")
        # Threshold markers
        for tval, tname, tcol in [
            (r.get("review_threshold"), "T_low",  "#888888"),
            (r.get("best_threshold"),   "T_op",   "#333333"),
            (r.get("deny_threshold"),   "T_high", "#000000"),
        ]:
            if tval is not None:
                ax.axvline(tval, color=tcol, linestyle="--", linewidth=1,
                           label=f"{tname}={tval:.2f}")
        ax.set_xlabel("Threshold on p(CONFIRM_FRAUD)")
        ax.set_ylabel("Score")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_title(f"{algo}")
        ax.legend(fontsize=7); ax.grid(linestyle="--", alpha=0.4)
    fig.suptitle("Precision / Recall / F1 vs Threshold — Primary Dataset", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "threshold_curves.png", dpi=150)
    plt.close(fig)


def save_results(results: dict, results_dir: Path,
                 all_seed_results: dict = None) -> None:
    """Save figures and CSVs.

    Parameters
    ----------
    results       : single-seed results dict (seed 42, used for figures)
    results_dir   : output directory
    all_seed_results : {seed: {algo: result_dict}} for multi-seed summary CSV
    """
    results_dir.mkdir(exist_ok=True)

    algos = list(results.keys())

    # ── 1. Per-seed summary CSV (seed 42 only if no multi-seed data) ─
    rows = []
    for algo, r in results.items():
        rows.append({
            "algorithm": algo,
            "roc_auc":   round(r["roc_auc"], 4),
            "pr_auc":    round(r["pr_auc"],  4),
            "t_low":     r.get("review_threshold"),
            "t_high":    r.get("deny_threshold"),
        })
    pd.DataFrame(rows).to_csv(results_dir / "summary.csv", index=False)

    # ── 2. Multi-seed aggregated summary CSV ─────────────────────────
    if all_seed_results and len(all_seed_results) > 1:
        agg_rows = []
        for algo in algos:
            roc_vals = [all_seed_results[s][algo]["roc_auc"] for s in all_seed_results]
            pr_vals  = [all_seed_results[s][algo]["pr_auc"]  for s in all_seed_results]
            tl_vals  = [v for s in all_seed_results
                        if (v := all_seed_results[s][algo].get("review_threshold")) is not None]
            th_vals  = [v for s in all_seed_results
                        if (v := all_seed_results[s][algo].get("deny_threshold"))   is not None]
            agg_rows.append({
                "algorithm":      algo,
                "roc_auc_mean":   round(float(np.mean(roc_vals)), 4),
                "roc_auc_std":    round(float(np.std(roc_vals)),  4),
                "pr_auc_mean":    round(float(np.mean(pr_vals)),  4),
                "pr_auc_std":     round(float(np.std(pr_vals)),   4),
                "t_low_mean":     round(float(np.mean(tl_vals)),  4) if tl_vals else None,
                "t_low_std":      round(float(np.std(tl_vals)),   4) if tl_vals else None,
                "t_high_mean":    round(float(np.mean(th_vals)),  4) if th_vals else None,
                "t_high_std":     round(float(np.std(th_vals)),   4) if th_vals else None,
                "n_seeds":        len(roc_vals),
            })
        pd.DataFrame(agg_rows).to_csv(results_dir / "multi_seed_summary.csv", index=False)

    # ── 2. Performance comparison bar chart ─────────────────────────
    x = np.arange(len(algos))
    width = 0.35
    roc_vals = [results[a]["roc_auc"] for a in algos]
    pr_vals  = [results[a]["pr_auc"]  for a in algos]

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, roc_vals, width, label="ROC-AUC", color="steelblue")
    b2 = ax.bar(x + width / 2, pr_vals,  width, label="PR-AUC",  color="darkorange")
    for bar, val in list(zip(b1, roc_vals)) + list(zip(b2, pr_vals)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Algorithm Performance — Primary Dataset (seed 42)")
    ax.legend()
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(results_dir / "performance_comparison.png", dpi=150)
    plt.close(fig)

    # ── 3. Decision distribution stacked bar chart ──────────────────
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
    ax.set_xticks(x)
    ax.set_xticklabels(algos, rotation=15, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Fraction of test set")
    ax.set_title("Three-Tier Decision Distribution — Primary Dataset (seed 42)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(results_dir / "decision_distribution.png", dpi=150)
    plt.close(fig)

    # ── 4. Per-algorithm confusion matrices & feature importance ─────
    for algo, r in results.items():
        _save_confusion_matrix(
            r["confusion_matrix"], CLASS_NAMES, algo, results_dir
        )
        if r.get("feature_importance") is not None:
            _save_feature_importance(r["feature_importance"], algo, results_dir)

    # ── 4b. New: score distributions, seed stability, threshold curves ──
    _save_score_distributions(results, results_dir, CLASS_NAMES)
    _save_threshold_curves(results, [a for a in algos if a != "dummy"], results_dir)
    if all_seed_results and len(all_seed_results) > 1:
        _save_seed_stability(all_seed_results, algos, results_dir)

    # ── 5. PR curves (OvR per class) ─────────────────────────────────
    n_classes = list(results.values())[0]["n_classes"]
    classes   = list(range(n_classes))
    y_bin_all = {}
    for algo, r in results.items():
        y_test  = r["y_test"]
        y_scores = r["y_scores"]  # (n_test, n_classes) for multi-class
        if hasattr(y_scores, "ndim") and y_scores.ndim == 2:
            y_bin_all[algo] = (label_binarize(y_test, classes=classes), y_scores)

    if y_bin_all:
        fig, axes = plt.subplots(1, n_classes,
                                 figsize=(5 * n_classes, 4), sharey=True)
        colors = plt.cm.tab10.colors
        for ci, class_name in enumerate(CLASS_NAMES):
            ax = axes[ci] if n_classes > 1 else axes
            for i, (algo, (y_bin, y_scores)) in enumerate(y_bin_all.items()):
                from sklearn.metrics import precision_recall_curve, average_precision_score
                prec, rec, _ = precision_recall_curve(y_bin[:, ci], y_scores[:, ci])
                ap = average_precision_score(y_bin[:, ci], y_scores[:, ci])
                ax.plot(rec, prec, color=colors[i % 10],
                        label=f"{algo} (AP={ap:.3f})")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title(f"PR Curve — {class_name}")
            ax.legend(fontsize=7)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
        fig.suptitle("Precision-Recall Curves (OvR) — Primary Dataset")
        fig.tight_layout()
        fig.savefig(results_dir / "pr_curves.png", dpi=150)
        plt.close(fig)

    # ── 6. ROC curves (OvR per class) ────────────────────────────────
    if y_bin_all:
        fig, axes = plt.subplots(1, n_classes,
                                 figsize=(5 * n_classes, 4), sharey=True)
        for ci, class_name in enumerate(CLASS_NAMES):
            ax = axes[ci] if n_classes > 1 else axes
            for i, (algo, (y_bin, y_scores)) in enumerate(y_bin_all.items()):
                fpr, tpr, _ = roc_curve(y_bin[:, ci], y_scores[:, ci])
                roc_auc = sklearn_auc(fpr, tpr)
                ax.plot(fpr, tpr, color=colors[i % 10],
                        label=f"{algo} (AUC={roc_auc:.3f})")
            ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC Curve — {class_name}")
            ax.legend(fontsize=7)
        fig.suptitle("ROC Curves (OvR) — Primary Dataset")
        fig.tight_layout()
        fig.savefig(results_dir / "roc_curves.png", dpi=150)
        plt.close(fig)

    print(f"\nResults saved to {results_dir}/")
    saved = sorted(p.name for p in results_dir.glob("*.png")) + \
            sorted(p.name for p in results_dir.glob("*.csv"))
    for f in saved:
        print(f"  {f}")


def load_and_prep(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Encode label
    df["label_encoded"] = df["label"].map(LABEL_MAP)
    if df["label_encoded"].isna().any():
        unknown = df.loc[df["label_encoded"].isna(), "label"].unique()
        raise ValueError(f"Unknown label values: {unknown}")

    # Cast numeric columns; coerce bad strings to NaN
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute remaining nulls: numeric → column median, categorical → mode
    for col in FEATURE_COLS:
        if df[col].isna().sum() == 0:
            continue
        if col in NUMERIC_COLS:
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


def main():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    print(f"Loading data from {DATA_PATH}...")
    df = load_and_prep(DATA_PATH)

    RESULTS_DIR.mkdir(exist_ok=True)
    _save_class_distribution(df, LABEL_MAP, RESULTS_DIR)

    print(f"  Rows:     {len(df):,}")
    print(f"  Features: {len(FEATURE_COLS)}")
    print(f"  Label distribution:")
    label_names = {v: k for k, v in LABEL_MAP.items()}
    for enc, name in sorted(label_names.items()):
        cnt = (df["label_encoded"] == enc).sum()
        print(f"    {enc} ({name:<22}) {cnt:>8,}  ({cnt/len(df)*100:.2f}%)")

    algos    = [args.algo] if args.algo else list(ALGORITHM_CATALOGUE.keys())
    run_grid = not args.no_grid_search
    seeds    = SEEDS if run_grid else [SEEDS[0]]  # single seed for smoke-tests

    all_seed_results = {}   # {seed: {algo: result}}
    best_params      = {}   # {algo: hyperparams from seed-42 grid search}

    for i, seed in enumerate(seeds):
        print(f"\n{'#'*70}")
        print(f"# SEED {seed}  ({i+1}/{len(seeds)})")
        print(f"{'#'*70}")

        seed_results = {}
        for algo in algos:
            print(f"\n{'='*70}")
            print(f"Running: {algo}  [seed={seed}]")
            print(f"{'='*70}")

            # Seed 42: run full grid search and record best params
            # Later seeds: reuse best params, skip grid search
            fixed = best_params.get(algo) if i > 0 else None

            seed_results[algo] = run_training_pipeline(
                df=df,
                feature_cols=FEATURE_COLS,
                target_col="label_encoded",
                model_name=f"FraudDetection_{algo}",
                algorithm=algo,
                run_grid_search=(i == 0 and run_grid),
                fixed_params=fixed,
                random_state=seed,
            )

            if i == 0:
                best_params[algo] = seed_results[algo]["model"].get_params()

        all_seed_results[seed] = seed_results

    # Use seed-42 results for figures
    results = all_seed_results[SEEDS[0]]

    # --- Summary table (mean ± std across seeds) ---
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY  ({len(seeds)} seeds: {seeds})")
    print(f"{'='*70}")
    print(f"{'Algorithm':<25}  {'ROC-AUC':>16}  {'PR-AUC':>16}  {'T_low':>7}  {'T_high':>7}")
    print("-" * 78)
    for algo in algos:
        roc_vals = [all_seed_results[s][algo]["roc_auc"] for s in seeds]
        pr_vals  = [all_seed_results[s][algo]["pr_auc"]  for s in seeds]
        t_low    = results[algo].get("review_threshold")
        t_high   = results[algo].get("deny_threshold")
        roc_str  = f"{np.mean(roc_vals):.4f}±{np.std(roc_vals):.4f}"
        pr_str   = f"{np.mean(pr_vals):.4f}±{np.std(pr_vals):.4f}"
        print(
            f"{algo:<25}  {roc_str:>16}  {pr_str:>16}"
            f"  {str(t_low) if t_low is not None else 'N/A':>7}"
            f"  {str(t_high) if t_high is not None else 'N/A':>7}"
        )

    save_results(results, RESULTS_DIR, all_seed_results=all_seed_results)

    return all_seed_results


if __name__ == "__main__":
    main()
