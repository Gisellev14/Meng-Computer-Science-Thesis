import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fraud_dataset import generate_fraud_dataset
from training_pipeline import run_training_pipeline

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def extract_summary(results: dict) -> dict:
    report = results["classification_report"]
    best_row = results["best_threshold_row"]

    summary = {
        "model_name": results["model_name"],
        "algorithm": results.get("algorithm", "unknown"),
        "target": results["target_col"],
        "roc_auc": round(results["roc_auc"], 4),
        "pr_auc": round(results["pr_auc"], 4),
        "precision_at_0.5": round(report["1"]["precision"], 4)
        if "1" in report
        else None,
        "recall_at_0.5": round(report["1"]["recall"], 4) if "1" in report else None,
        "f1_at_0.5": round(report["1"]["f1-score"], 4) if "1" in report else None,
    }

    if best_row is not None:
        summary.update(
            {
                "best_threshold": round(float(best_row["threshold"]), 4),
                "precision_at_best_threshold": round(
                    float(best_row["precision_positive"]), 4
                ),
                "recall_at_best_threshold": round(
                    float(best_row["recall_positive"]), 4
                ),
                "f1_at_best_threshold": round(float(best_row["f1_positive"]), 4),
                "flag_rate_at_best_threshold": round(float(best_row["flag_rate"]), 4),
            }
        )

        if "recall_stealth" in best_row.index and pd.notna(best_row["recall_stealth"]):
            summary["recall_stealth_at_best_threshold"] = round(
                float(best_row["recall_stealth"]), 4
            )
        else:
            summary["recall_stealth_at_best_threshold"] = None
    else:
        summary.update(
            {
                "best_threshold": None,
                "precision_at_best_threshold": None,
                "recall_at_best_threshold": None,
                "f1_at_best_threshold": None,
                "flag_rate_at_best_threshold": None,
                "recall_stealth_at_best_threshold": None,
            }
        )

    # Three-threshold decision system
    summary["review_threshold"] = round(float(results.get("review_threshold", 0)), 4)
    summary["deny_threshold"] = round(float(results.get("deny_threshold", 0)), 4)
    dc = results.get("decision_counts", {})
    total = sum(dc.values()) if dc else 1
    summary["pct_allow"] = round(dc.get("ALLOW", 0) / total, 4) if total else None
    summary["pct_review"] = round(dc.get("REVIEW", 0) / total, 4) if total else None
    summary["pct_deny"] = round(dc.get("DENY", 0) / total, 4) if total else None

    return summary


def plot_combined_pr_curves(results_list, precision_constraint=0.50):
    plt.figure(figsize=(9, 7))

    for results in results_list:
        label = f"{results['model_name']} ({results.get('algorithm', '')})"
        precision_curve = results["precision_curve"]
        recall_curve = results["recall_curve"]
        best_row = results["best_threshold_row"]

        plt.plot(recall_curve, precision_curve, label=label)

        if best_row is not None:
            plt.scatter(
                best_row["recall_positive"], best_row["precision_positive"], s=70
            )
            plt.annotate(
                f"T={best_row['threshold']:.2f}",
                (best_row["recall_positive"], best_row["precision_positive"]),
                textcoords="offset points",
                xytext=(8, -12),
                fontsize=9,
            )

    plt.axhline(
        y=precision_constraint,
        linestyle="--",
        color="gray",
        label=f"Precision = {precision_constraint}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison Across Models")
    plt.legend(fontsize=8, loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "pr_curves_comparison.png"), dpi=150)
    plt.close()
    print(f"\nPR curve plot saved to {RESULTS_DIR}/pr_curves_comparison.png")


# Algorithms to compare for each feature set
ALGORITHMS = ["dummy", "logistic_regression", "random_forest", "gradient_boosting"]


def run_single_seed(
    df,
    behavioral_features,
    transaction_features,
    combined_features,
    precision_constraint,
    seed,
):
    """Run all models for a single seed. Returns list of result dicts."""
    all_results = []

    for algo in ALGORITHMS:
        # Behavioral model (target = is_bot)
        results = run_training_pipeline(
            df=df,
            feature_cols=behavioral_features,
            target_col="is_bot",
            model_name="behavioral",
            algorithm=algo,
            threshold_precision_constraint=precision_constraint,
            random_state=seed,
        )
        all_results.append(results)

        # Transaction-only fraud model
        results = run_training_pipeline(
            df=df,
            feature_cols=transaction_features,
            target_col="is_fraud",
            model_name="transactional",
            algorithm=algo,
            threshold_precision_constraint=precision_constraint,
            random_state=seed,
        )
        all_results.append(results)

        # Combined fraud model
        results = run_training_pipeline(
            df=df,
            feature_cols=combined_features,
            target_col="is_fraud",
            model_name="combined",
            algorithm=algo,
            threshold_precision_constraint=precision_constraint,
            random_state=seed,
        )
        all_results.append(results)

    return all_results


def main():
    # 1. Define feature sets
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
    precision_constraint = 0.50

    # 2. Multi-seed runs for statistical robustness
    seeds = [42, 123, 456, 789, 2024]
    all_summaries = []

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"  SEED = {seed}")
        print(f"{'=' * 60}")

        df = generate_fraud_dataset(num_samples=10000, seed=seed)

        seed_results = run_single_seed(
            df,
            behavioral_features,
            transaction_features,
            combined_features,
            precision_constraint,
            seed,
        )

        for r in seed_results:
            s = extract_summary(r)
            s["seed"] = seed
            all_summaries.append(s)

    # 3. Aggregate across seeds
    summary_df = pd.DataFrame(all_summaries)

    # Mean ± std per (model_name, algorithm)
    group_cols = ["model_name", "algorithm", "target"]
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "best_threshold",
        "precision_at_best_threshold",
        "recall_at_best_threshold",
        "f1_at_best_threshold",
    ]

    agg_df = summary_df.groupby(group_cols)[metric_cols].agg(["mean", "std"]).round(4)
    agg_df.columns = [f"{m}_{s}" for m, s in agg_df.columns]
    agg_df = agg_df.reset_index()

    print("\n" + "=" * 80)
    print("  AGGREGATED RESULTS (mean ± std across seeds)")
    print("=" * 80)
    print(agg_df.to_string(index=False))

    # 4. Save all outputs
    summary_df.to_csv(os.path.join(RESULTS_DIR, "all_seeds_summary.csv"), index=False)
    agg_df.to_csv(os.path.join(RESULTS_DIR, "aggregated_results.csv"), index=False)

    # 5. Re-run first seed to get PR curve data (summaries don't store curves)
    print("\n\nGenerating PR curves for seed=42 ...")
    df_curve = generate_fraud_dataset(num_samples=10000, seed=42)
    # Only plot RF and GB for clarity (dummy and logreg PR curves less useful)
    curve_results = []
    for algo in ["random_forest", "gradient_boosting"]:
        for feat_name, feat_cols, target in [
            ("behavioral", behavioral_features, "is_bot"),
            ("transactional", transaction_features, "is_fraud"),
            ("combined", combined_features, "is_fraud"),
        ]:
            r = run_training_pipeline(
                df=df_curve,
                feature_cols=feat_cols,
                target_col=target,
                model_name=feat_name,
                algorithm=algo,
                threshold_precision_constraint=precision_constraint,
                random_state=42,
            )
            curve_results.append(r)

    plot_combined_pr_curves(curve_results, precision_constraint=precision_constraint)

    # 6. Print summary tables
    print("\n=== Table 1: Model Performance (seed=42) ===")
    seed42 = summary_df[summary_df["seed"] == 42]
    print(
        seed42[["model_name", "algorithm", "target", "roc_auc", "pr_auc"]].to_string(
            index=False
        )
    )

    print(
        f"\n=== Table 2: Operating Threshold (precision >= {precision_constraint}) ==="
    )
    print(
        seed42[
            [
                "model_name",
                "algorithm",
                "best_threshold",
                "precision_at_best_threshold",
                "recall_at_best_threshold",
                "f1_at_best_threshold",
                "flag_rate_at_best_threshold",
            ]
        ].to_string(index=False)
    )

    print("\n=== Table 3: Stealth Bot Detection ===")
    print(
        seed42[
            ["model_name", "algorithm", "recall_stealth_at_best_threshold"]
        ].to_string(index=False)
    )

    print("\n=== Table 4: Aggregated Results (mean ± std) ===")
    print(agg_df.to_string(index=False))

    print("\n=== Table 5: Three-Threshold Decision Distribution (seed=42) ===")
    print(
        seed42[
            [
                "model_name",
                "algorithm",
                "review_threshold",
                "best_threshold",
                "deny_threshold",
                "pct_allow",
                "pct_review",
                "pct_deny",
            ]
        ].to_string(index=False)
    )

    return {"summary_df": summary_df, "agg_df": agg_df}


if __name__ == "__main__":
    main()
