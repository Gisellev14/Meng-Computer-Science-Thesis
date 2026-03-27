"""
Generate comprehensive test-set performance graphs and literature comparison tables.

Produces (all saved to results/):
  - confusion_matrix_<model>_<algo>.png   — per-model confusion matrix heatmap
  - roc_curves_comparison.png             — ROC curves for all models on one plot
  - pr_curves_by_feature_set.png          — PR curves grouped by feature set
  - performance_bar_chart.png             — PR-AUC / ROC-AUC bar chart grouped by feature set
  - feature_importance_<model>_<algo>.png  — feature importance for tree-based models
  - threshold_sensitivity_<model>_<algo>.png — precision/recall vs threshold
  - test_set_summary.csv                  — full test-set metrics table
  - literature_comparison.csv             — comparison against published results
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from fraud_dataset import generate_fraud_dataset
from training_pipeline import run_training_pipeline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALGORITHMS = ["dummy", "logistic_regression", "random_forest", "gradient_boosting"]
ALGO_DISPLAY = {
    "dummy": "Dummy (Prior)",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}
MODEL_DISPLAY = {
    "behavioral": "Behavioral",
    "transactional": "Transactional",
    "combined": "Combined",
}
SEED = 42
PRECISION_CONSTRAINT = 0.50


# ---------------------------------------------------------------------------
# 1. Confusion Matrix Heatmaps
# ---------------------------------------------------------------------------
def plot_confusion_matrix(results, save_dir):
    """Save a confusion matrix heatmap for a single model result."""
    cm = results["confusion_matrix"]
    model = results["model_name"]
    algo = results["algorithm"]
    target = results["target_col"]

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = (
        ["Legit (0)", "Fraud (1)"] if target == "is_fraud" else ["Human (0)", "Bot (1)"]
    )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
        title=f"Confusion Matrix — {MODEL_DISPLAY.get(model, model)}\n{ALGO_DISPLAY.get(algo, algo)} (threshold=0.5)",
    )

    # Annotate cells
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )

    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_{model}_{algo}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# 2. ROC Curves (combined plot)
# ---------------------------------------------------------------------------
def plot_roc_curves(results_list, save_dir):
    """Plot ROC curves for all models on a single figure."""
    fig, ax = plt.subplots(figsize=(9, 7))

    for r in results_list:
        if r["algorithm"] == "dummy":
            continue
        scores = r.get("_y_scores")
        y_test = r.get("_y_test")
        if scores is None or y_test is None:
            continue

        fpr, tpr, _ = roc_curve(y_test, scores)
        label = f"{MODEL_DISPLAY.get(r['model_name'], r['model_name'])} — {ALGO_DISPLAY.get(r['algorithm'], r['algorithm'])} (AUC={r['roc_auc']:.3f})"
        ax.plot(fpr, tpr, label=label)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Test Set Performance")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "roc_curves_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 3. PR Curves grouped by feature set
# ---------------------------------------------------------------------------
def plot_pr_curves_by_feature_set(results_list, save_dir):
    """One subplot per feature set showing PR curves for each algorithm."""
    feature_sets = ["behavioral", "transactional", "combined"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for idx, feat_name in enumerate(feature_sets):
        ax = axes[idx]
        feat_results = [r for r in results_list if r["model_name"] == feat_name]

        for r in feat_results:
            algo = r["algorithm"]
            if algo == "dummy":
                continue
            label = f"{ALGO_DISPLAY.get(algo, algo)} (PR-AUC={r['pr_auc']:.3f})"
            ax.plot(r["recall_curve"], r["precision_curve"], label=label)

            best_row = r["best_threshold_row"]
            if best_row is not None:
                ax.scatter(
                    best_row["recall_positive"],
                    best_row["precision_positive"],
                    s=60,
                    zorder=5,
                )

        ax.axhline(y=PRECISION_CONSTRAINT, linestyle="--", color="gray", alpha=0.6)
        ax.set_xlabel("Recall")
        if idx == 0:
            ax.set_ylabel("Precision")
        ax.set_title(f"{MODEL_DISPLAY.get(feat_name, feat_name)}")
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)

    plt.suptitle(
        "Precision-Recall Curves by Feature Set (Test Set)", fontsize=13, y=1.02
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "pr_curves_by_feature_set.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 4. Performance Bar Chart (PR-AUC and ROC-AUC)
# ---------------------------------------------------------------------------
def plot_performance_bar_chart(summary_df, save_dir):
    """Grouped bar chart: PR-AUC and ROC-AUC per model × algorithm."""
    # Filter out dummy for cleaner chart (summary_df already uses display names)
    dummy_label = ALGO_DISPLAY.get("dummy", "Dummy (Prior)")
    df = summary_df[summary_df["algorithm"] != dummy_label].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in [
        (axes[0], "pr_auc", "PR-AUC (Test Set)"),
        (axes[1], "roc_auc", "ROC-AUC (Test Set)"),
    ]:
        pivot = df.pivot_table(index="model_name", columns="algorithm", values=metric)
        # Reorder using display names
        display_order = [
            MODEL_DISPLAY.get(m, m) for m in ["behavioral", "transactional", "combined"]
        ]
        pivot = pivot.reindex([m for m in display_order if m in pivot.index])

        pivot.plot(kind="bar", ax=ax, rot=0, width=0.7)
        ax.set_title(title, fontsize=12)
        ax.set_ylabel(metric.upper())
        ax.set_xlabel("")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)

    plt.tight_layout()
    path = os.path.join(save_dir, "performance_bar_chart.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 5. Feature Importance Bar Charts
# ---------------------------------------------------------------------------
def plot_feature_importance(results, save_dir):
    """Save a horizontal bar chart of feature importance."""
    fi = results.get("feature_importance")
    if fi is None or fi.empty:
        return None

    model = results["model_name"]
    algo = results["algorithm"]

    top = fi.head(15)

    fig, ax = plt.subplots(figsize=(7, max(4, len(top) * 0.35)))
    ax.barh(range(len(top)), top["importance"].values, color="steelblue")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["feature"].values, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(
        f"Feature Importance — {MODEL_DISPLAY.get(model, model)}\n{ALGO_DISPLAY.get(algo, algo)}"
    )
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, f"feature_importance_{model}_{algo}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# 6. Threshold Sensitivity Curves
# ---------------------------------------------------------------------------
def plot_threshold_sensitivity(results, save_dir):
    """Plot precision, recall, F1 as a function of threshold."""
    tt = results.get("threshold_table")
    if tt is None or tt.empty:
        return None

    model = results["model_name"]
    algo = results["algorithm"]
    best_t = results["best_threshold"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        tt["threshold"],
        tt["precision_positive"],
        "b-o",
        markersize=4,
        label="Precision",
    )
    ax.plot(tt["threshold"], tt["recall_positive"], "r-s", markersize=4, label="Recall")
    ax.plot(tt["threshold"], tt["f1_positive"], "g-^", markersize=4, label="F1")

    ax.axhline(
        y=PRECISION_CONSTRAINT,
        linestyle="--",
        color="gray",
        alpha=0.5,
        label=f"Precision = {PRECISION_CONSTRAINT}",
    )
    ax.axvline(
        x=best_t,
        linestyle=":",
        color="orange",
        alpha=0.7,
        label=f"Selected threshold = {best_t}",
    )

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Score")
    ax.set_title(
        f"Threshold Sensitivity — {MODEL_DISPLAY.get(model, model)}\n{ALGO_DISPLAY.get(algo, algo)}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    path = os.path.join(save_dir, f"threshold_sensitivity_{model}_{algo}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


# ---------------------------------------------------------------------------
# 7. Three-Tier Decision Distribution Chart
# ---------------------------------------------------------------------------
def plot_decision_distribution(results_list, save_dir):
    """Stacked bar chart showing ALLOW / REVIEW / DENY proportions per model."""
    rows = []
    for r in results_list:
        if r["algorithm"] == "dummy":
            continue
        dc = r.get("decision_counts", {})
        total = sum(dc.values()) if dc else 1
        rows.append(
            {
                "label": f"{MODEL_DISPLAY.get(r['model_name'], r['model_name'])}\n{ALGO_DISPLAY.get(r['algorithm'], r['algorithm'])}",
                "ALLOW": dc.get("ALLOW", 0) / total,
                "REVIEW": dc.get("REVIEW", 0) / total,
                "DENY": dc.get("DENY", 0) / total,
                "T_low": r.get("review_threshold", 0),
                "T_op": r.get("best_threshold", 0.5),
                "T_high": r.get("deny_threshold", r.get("best_threshold", 0.5)),
            }
        )

    if not rows:
        return None

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(df))
    width = 0.6

    ax.bar(x, df["ALLOW"], width, label="ALLOW", color="#2ecc71")
    ax.bar(x, df["REVIEW"], width, bottom=df["ALLOW"], label="REVIEW", color="#f39c12")
    ax.bar(
        x,
        df["DENY"],
        width,
        bottom=df["ALLOW"] + df["REVIEW"],
        label="DENY",
        color="#e74c3c",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], fontsize=8)
    ax.set_ylabel("Proportion of Test Transactions")
    ax.set_title("Three-Tier Decision Distribution (ALLOW / REVIEW / DENY)")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    # Add threshold annotations
    for i, row in df.iterrows():
        ax.text(
            i,
            0.5,
            f"T_low={row['T_low']:.2f}\nT_op={row['T_op']:.2f}\nT_high={row['T_high']:.2f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white",
            fontweight="bold",
        )

    plt.tight_layout()
    path = os.path.join(save_dir, "decision_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# 7b. Fraud Score Distribution Histogram
# ---------------------------------------------------------------------------
def plot_score_distribution(results_list, save_dir):
    """Histogram of fraud probability scores, split by true label, with threshold lines."""
    for r in results_list:
        if r["algorithm"] == "dummy":
            continue
        y_scores = r.get("y_scores")
        y_test = r.get("y_test")
        if y_scores is None or y_test is None:
            continue

        model = r["model_name"]
        algo = r["algorithm"]
        t_low = r.get("review_threshold", 0)
        t_op = r.get("best_threshold", 0.5)
        t_high = r.get("deny_threshold", r.get("best_threshold", 0.5))

        fig, ax = plt.subplots(figsize=(9, 5))

        # Split scores by true label
        scores_legit = y_scores[y_test == 0]
        scores_fraud = y_scores[y_test == 1]

        ax.hist(
            scores_legit,
            bins=50,
            alpha=0.6,
            color="steelblue",
            label=f"Legitimate (n={len(scores_legit)})",
            density=True,
        )
        ax.hist(
            scores_fraud,
            bins=50,
            alpha=0.6,
            color="coral",
            label=f"Fraud (n={len(scores_fraud)})",
            density=True,
        )

        # Threshold lines
        ax.axvline(
            x=t_low,
            linestyle="--",
            color="#2ecc71",
            linewidth=2,
            label=f"T_low = {t_low:.2f} (recall ≥ 0.95)",
        )
        ax.axvline(
            x=t_op,
            linestyle="-.",
            color="#f39c12",
            linewidth=2,
            label=f"T_op = {t_op:.2f} (prec ≥ 0.50)",
        )
        ax.axvline(
            x=t_high,
            linestyle="-",
            color="#e74c3c",
            linewidth=2,
            label=f"T_high = {t_high:.2f} (prec ≥ 0.70)",
        )

        # Zone labels
        y_max = ax.get_ylim()[1]
        zone_y = y_max * 0.95
        ax.text(
            (0 + t_low) / 2,
            zone_y,
            "ALLOW",
            ha="center",
            fontsize=9,
            color="#2ecc71",
            fontweight="bold",
        )
        ax.text(
            (t_low + t_high) / 2,
            zone_y,
            "REVIEW",
            ha="center",
            fontsize=9,
            color="#f39c12",
            fontweight="bold",
        )
        ax.text(
            (t_high + 1) / 2,
            zone_y,
            "DENY",
            ha="center",
            fontsize=9,
            color="#e74c3c",
            fontweight="bold",
        )

        ax.set_xlabel("Fraud Probability Score")
        ax.set_ylabel("Density")
        ax.set_title(
            f"Score Distribution — {MODEL_DISPLAY.get(model, model)}\n"
            f"{ALGO_DISPLAY.get(algo, algo)}"
        )
        ax.legend(fontsize=7, loc="upper center")
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = os.path.join(save_dir, f"score_distribution_{model}_{algo}.png")
        plt.savefig(path, dpi=150)
        plt.close()

    print("Saved: score distribution histograms")


# ---------------------------------------------------------------------------
# 8. Literature Comparison Table
# ---------------------------------------------------------------------------
def build_literature_comparison(our_results_df, save_dir):
    """
    Build a comparison table against published results on PaySim and fraud detection.

    Sources (results reported on PaySim or similar mobile money fraud datasets):
      - Lopez-Rojas et al. (2016): PaySim original paper, Random Forest
      - Carcillo et al. (2018): Scalable real-time fraud detection, Random Forest
      - Alarfaj et al. (2022): Credit card fraud, ensemble comparison
      - Hilal et al. (2022): Financial fraud detection survey
      - Kaggle PaySim benchmarks: XGBoost, LightGBM community solutions
    """
    literature = [
        {
            "Study": "Lopez-Rojas et al. (2016)",
            "Dataset": "PaySim (6.3M txn)",
            "Method": "Random Forest",
            "Features": "Transactional only",
            "ROC-AUC": 0.97,
            "PR-AUC": "—",
            "Precision": "—",
            "Recall": 0.95,
            "Notes": "Original PaySim paper; full real-value features",
        },
        {
            "Study": "Carcillo et al. (2018)",
            "Dataset": "Real bank data",
            "Method": "Random Forest (streaming)",
            "Features": "Transactional + aggregated",
            "ROC-AUC": "—",
            "PR-AUC": 0.52,
            "Precision": "—",
            "Recall": "—",
            "Notes": "Real-time fraud detection with concept drift",
        },
        {
            "Study": "Alarfaj et al. (2022)",
            "Dataset": "Credit card (Kaggle)",
            "Method": "XGBoost",
            "Features": "PCA-transformed",
            "ROC-AUC": 0.98,
            "PR-AUC": "—",
            "Precision": 0.95,
            "Recall": 0.94,
            "Notes": "Highly engineered, PCA features",
        },
        {
            "Study": "Hilal et al. (2022)",
            "Dataset": "PaySim + others",
            "Method": "LightGBM",
            "Features": "Transactional",
            "ROC-AUC": 0.99,
            "PR-AUC": "—",
            "Precision": 0.97,
            "Recall": 0.95,
            "Notes": "Survey best-case; trained on original PaySim labels",
        },
        {
            "Study": "Kaggle community (2020-2023)",
            "Dataset": "PaySim (Kaggle)",
            "Method": "XGBoost / LightGBM",
            "Features": "Transactional + engineered",
            "ROC-AUC": "0.97–0.99",
            "PR-AUC": "0.85–0.95",
            "Precision": "—",
            "Recall": "—",
            "Notes": "Leaderboard solutions; risk of leakage in some entries",
        },
    ]

    # Add our results (best per feature set, using RF and GB)
    for _, row in our_results_df.iterrows():
        if row["algorithm"] in ["dummy", "logistic_regression"]:
            continue
        literature.append(
            {
                "Study": "This work",
                "Dataset": "Synthetic (PaySim-calibrated)",
                "Method": ALGO_DISPLAY.get(row["algorithm"], row["algorithm"]),
                "Features": MODEL_DISPLAY.get(row["model_name"], row["model_name"]),
                "ROC-AUC": row.get("roc_auc", "—"),
                "PR-AUC": row.get("pr_auc", "—"),
                "Precision": row.get("precision_at_best_threshold", "—"),
                "Recall": row.get("recall_at_best_threshold", "—"),
                "Notes": f"Threshold={row.get('best_threshold', '—')} (prec≥{PRECISION_CONSTRAINT})",
            }
        )

    lit_df = pd.DataFrame(literature)
    path = os.path.join(save_dir, "literature_comparison.csv")
    lit_df.to_csv(path, index=False)
    print(f"Saved: {path}")

    # Also save a formatted markdown table
    md_path = os.path.join(save_dir, "literature_comparison.md")
    with open(md_path, "w") as f:
        f.write("# Literature Comparison\n\n")
        f.write("Direct comparison is approximate. Our dataset is synthetic ")
        f.write(
            "(calibrated from PaySim), while literature results are on the original "
        )
        f.write(
            "PaySim or real banking data. Feature sets and evaluation protocols differ.\n\n"
        )
        f.write(lit_df.to_markdown(index=False))
        f.write("\n")
    print(f"Saved: {md_path}")

    return lit_df


# ---------------------------------------------------------------------------
# 8. Combined Test Performance Summary Table
# ---------------------------------------------------------------------------
def build_test_summary_table(results_list, save_dir):
    """Build and save a comprehensive test-set metrics table."""
    rows = []
    for r in results_list:
        best_row = r["best_threshold_row"]
        report = r["classification_report"]
        row = {
            "model_name": MODEL_DISPLAY.get(r["model_name"], r["model_name"]),
            "algorithm": ALGO_DISPLAY.get(r["algorithm"], r["algorithm"]),
            "target": r["target_col"],
            "roc_auc": round(r["roc_auc"], 4),
            "pr_auc": round(r["pr_auc"], 4),
            "precision_0.5": round(report["1"]["precision"], 4)
            if "1" in report
            else None,
            "recall_0.5": round(report["1"]["recall"], 4) if "1" in report else None,
            "f1_0.5": round(report["1"]["f1-score"], 4) if "1" in report else None,
        }
        if best_row is not None:
            row["best_threshold"] = round(float(best_row["threshold"]), 2)
            row["precision_best"] = round(float(best_row["precision_positive"]), 4)
            row["recall_best"] = round(float(best_row["recall_positive"]), 4)
            row["f1_best"] = round(float(best_row["f1_positive"]), 4)
            row["flag_rate_best"] = round(float(best_row["flag_rate"]), 4)
        # Three-tier decision data
        row["review_threshold"] = round(float(r.get("review_threshold", 0)), 2)
        dc = r.get("decision_counts", {})
        total = sum(dc.values()) if dc else 1
        row["pct_allow"] = round(dc.get("ALLOW", 0) / total, 4) if total else None
        row["pct_review"] = round(dc.get("REVIEW", 0) / total, 4) if total else None
        row["pct_deny"] = round(dc.get("DENY", 0) / total, 4) if total else None
        rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(save_dir, "test_set_summary.csv")
    df.to_csv(path, index=False)
    print(f"Saved: {path}")

    # Markdown version
    md_path = os.path.join(save_dir, "test_set_summary.md")
    with open(md_path, "w") as f:
        f.write("# Test Set Performance Summary (seed=42)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n")
    print(f"Saved: {md_path}")

    return df


# ---------------------------------------------------------------------------
# 9. Precision-Constraint Sweep (Fraud Models)
# ---------------------------------------------------------------------------
def run_precision_constraint_sweep(df, save_dir):
    """Sweep precision constraints and report how recall/metrics change.

    This is intended to support selecting a practical operating point aligned with
    literature-reported precision/recall tradeoffs.
    """

    constraints = [round(x, 2) for x in np.arange(0.40, 0.801, 0.05)]

    transaction_features = [
        "transaction_type",
        "amount",
        "old_balance_origin",
        "new_balance_origin",
        "old_balance_dest",
        "new_balance_dest",
        "balance_drain_ratio",
        "is_full_drain",
        "dest_balance_unchanged",
    ]
    combined_features = [
        "session_duration_sec",
        "checkout_velocity_sec",
        "mouse_speed_variance",
        "keystroke_flight_time_ms",
        "impossible_travel_flag",
        "is_datacenter_ip",
        "pages_visited",
    ] + transaction_features

    configs = [
        ("transactional", transaction_features, "is_fraud"),
        ("combined", combined_features, "is_fraud"),
    ]
    sweep_algorithms = [
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    ]

    rows = []
    for c in constraints:
        for algo in sweep_algorithms:
            for feat_name, feat_cols, target in configs:
                r = run_training_pipeline(
                    df=df,
                    feature_cols=feat_cols,
                    target_col=target,
                    model_name=feat_name,
                    algorithm=algo,
                    threshold_precision_constraint=float(c),
                    random_state=SEED,
                )

                best_row = r.get("best_threshold_row")
                row = {
                    "precision_constraint": float(c),
                    "model_name": MODEL_DISPLAY.get(feat_name, feat_name),
                    "algorithm": ALGO_DISPLAY.get(algo, algo),
                    "best_threshold": round(float(r.get("best_threshold", 0)), 4),
                    "review_threshold": round(float(r.get("review_threshold", 0)), 4),
                    "pr_auc": round(float(r.get("pr_auc", 0)), 4),
                    "roc_auc": round(float(r.get("roc_auc", 0)), 4),
                    "precision": round(float(r.get("precision", 0)), 4),
                    "recall": round(float(r.get("recall", 0)), 4),
                    "f1": round(float(r.get("f1", 0)), 4),
                }
                if best_row is not None:
                    row["val_precision_at_best"] = round(
                        float(best_row.get("precision_positive", np.nan)), 4
                    )
                    row["val_recall_at_best"] = round(
                        float(best_row.get("recall_positive", np.nan)), 4
                    )
                    row["flag_rate_best"] = round(
                        float(best_row.get("flag_rate", np.nan)), 4
                    )

                dc = r.get("decision_counts", {})
                total = sum(dc.values()) if dc else 0
                if total:
                    row["pct_allow"] = round(dc.get("ALLOW", 0) / total, 4)
                    row["pct_review"] = round(dc.get("REVIEW", 0) / total, 4)
                    row["pct_deny"] = round(dc.get("DENY", 0) / total, 4)
                else:
                    row["pct_allow"] = None
                    row["pct_review"] = None
                    row["pct_deny"] = None

                rows.append(row)

    sweep_df = pd.DataFrame(rows)

    csv_path = os.path.join(save_dir, "precision_constraint_sweep.csv")
    sweep_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    md_path = os.path.join(save_dir, "precision_constraint_sweep.md")
    with open(md_path, "w") as f:
        f.write("# Precision-Constraint Sweep (0.40–0.80) — Fraud Models (seed=42)\n\n")
        f.write(sweep_df.to_markdown(index=False))
        f.write("\n")
    print(f"Saved: {md_path}")

    # Plot recall vs precision constraint (using validation recall since test recall is often 0)
    fig, ax = plt.subplots(figsize=(9, 6))
    for (model_name, algo_name), g in sweep_df.groupby(["model_name", "algorithm"]):
        g = g.sort_values("precision_constraint")
        # Filter out rows with NaN validation recall (models that couldn't meet constraint)
        g_valid = g.dropna(subset=["val_recall_at_best"])
        if len(g_valid) > 0:
            ax.plot(
                g_valid["precision_constraint"],
                g_valid["val_recall_at_best"],
                marker="o",
                linewidth=1.5,
                label=f"{model_name} — {algo_name}",
            )
    ax.set_xlabel("Precision Constraint")
    ax.set_ylabel("Validation Recall")
    ax.set_title("Validation Recall vs Precision Constraint (Fraud Models, seed=42)")
    ax.set_xlim(0.40, 0.80)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    plt.tight_layout()

    fig_path = os.path.join(save_dir, "precision_constraint_sweep_recall.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"Saved: {fig_path}")

    return sweep_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Generating Test Set Graphs (seed=42)")
    print("=" * 60)

    behavioral_features = [
        "session_duration_sec",
        "checkout_velocity_sec",
        "mouse_speed_variance",
        "keystroke_flight_time_ms",
        "impossible_travel_flag",
        "is_datacenter_ip",
        "pages_visited",
    ]
    transaction_features = [
        "transaction_type",
        "amount",
        "old_balance_origin",
        "new_balance_origin",
        "old_balance_dest",
        "new_balance_dest",
        "balance_drain_ratio",
        "is_full_drain",
        "dest_balance_unchanged",
    ]
    combined_features = behavioral_features + transaction_features

    df = generate_fraud_dataset(num_samples=10000, seed=SEED)
    print(
        f"Dataset: {df.shape[0]} rows, fraud_rate={df['is_fraud'].mean():.3f}, bot_rate={df['is_bot'].mean():.3f}"
    )

    # Run all 12 configurations
    all_results = []
    configs = [
        ("behavioral", behavioral_features, "is_bot"),
        ("transactional", transaction_features, "is_fraud"),
        ("combined", combined_features, "is_fraud"),
    ]

    for algo in ALGORITHMS:
        for feat_name, feat_cols, target in configs:
            print(f"\n--- {feat_name} / {algo} ---")
            r = run_training_pipeline(
                df=df,
                feature_cols=feat_cols,
                target_col=target,
                model_name=feat_name,
                algorithm=algo,
                threshold_precision_constraint=PRECISION_CONSTRAINT,
                random_state=SEED,
            )
            # Stash raw test data for ROC curve computation
            # (pipeline doesn't return y_test/y_scores directly — recompute from results)
            # We need to re-extract y_test and y_scores; the pipeline stores them indirectly
            # via score_distribution
            sd = r.get("score_distribution")
            if sd is not None:
                r["_y_test"] = sd["y_true"].values
                r["_y_scores"] = sd["y_score"].values
            all_results.append(r)

    # Generate all graphs
    print("\n" + "=" * 60)
    print("  Generating Graphs")
    print("=" * 60)

    # Confusion matrices
    for r in all_results:
        plot_confusion_matrix(r, RESULTS_DIR)
    print(f"Saved: {len(all_results)} confusion matrix plots")

    # ROC curves
    plot_roc_curves(all_results, RESULTS_DIR)

    # PR curves by feature set
    plot_pr_curves_by_feature_set(all_results, RESULTS_DIR)

    # Feature importance
    fi_count = 0
    for r in all_results:
        if plot_feature_importance(r, RESULTS_DIR):
            fi_count += 1
    print(f"Saved: {fi_count} feature importance plots")

    # Threshold sensitivity
    ts_count = 0
    for r in all_results:
        if r["algorithm"] != "dummy":
            if plot_threshold_sensitivity(r, RESULTS_DIR):
                ts_count += 1
    print(f"Saved: {ts_count} threshold sensitivity plots")

    # Summary table
    summary_df = build_test_summary_table(all_results, RESULTS_DIR)

    # Performance bar chart
    plot_performance_bar_chart(summary_df, RESULTS_DIR)

    # Three-tier decision distribution
    plot_decision_distribution(all_results, RESULTS_DIR)

    # Score distribution histograms
    plot_score_distribution(all_results, RESULTS_DIR)

    # Literature comparison
    build_literature_comparison(summary_df, RESULTS_DIR)

    # Precision-constraint sweep
    run_precision_constraint_sweep(df, RESULTS_DIR)

    print("\n" + "=" * 60)
    print("  ALL GRAPHS GENERATED SUCCESSFULLY")
    print(f"  Output directory: {RESULTS_DIR}")
    print("=" * 60)

    # Print test set summary to console
    print("\n=== Test Set Performance (seed=42) ===")
    print(summary_df.to_string(index=False))

    return all_results, summary_df


if __name__ == "__main__":
    main()
