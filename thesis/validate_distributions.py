"""
Distribution validation: compare generated synthetic data against PaySim.
Produces histograms, QQ plots, and KS test results for thesis documentation.
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from fraud_dataset import generate_fraud_dataset
from fraud_dataset_unimodal import generate_fraud_dataset_unimodal
from paysim_dataset import load_paysim

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def compare_distributions(paysim_df, synth_df, approach_name=""):
    """Compare key distributions between PaySim and synthetic data."""

    # Filter PaySim to relevant types
    valid_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
    ps = paysim_df[paysim_df["type"].isin(valid_types)].copy()

    results = []

    # --- 1. Transaction type distribution ---
    ps_type_dist = ps["type"].value_counts(normalize=True).sort_index()
    syn_type_dist = (
        synth_df["transaction_type"].value_counts(normalize=True).sort_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ps_type_dist.plot(
        kind="bar", ax=axes[0], title="PaySim Transaction Types", color="steelblue"
    )
    syn_type_dist.plot(
        kind="bar",
        ax=axes[1],
        title=f"Synthetic Transaction Types ({approach_name})",
        color="coral",
    )
    for ax in axes:
        ax.set_ylabel("Proportion")
        ax.set_ylim(0, 1)
    plt.tight_layout()
    filename = (
        f"dist_transaction_types_{approach_name}.png"
        if approach_name
        else "dist_transaction_types.png"
    )
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()

    # --- 2. Amount distribution (log scale) ---
    ps_log_amount = np.log1p(ps["amount"].clip(lower=0))
    syn_log_amount = np.log1p(synth_df["amount"].clip(lower=0))

    ks_stat, ks_pval = stats.ks_2samp(
        ps_log_amount.sample(min(10000, len(ps_log_amount)), random_state=42),
        syn_log_amount,
    )
    results.append(
        {"feature": "log1p(amount)", "ks_statistic": ks_stat, "ks_pvalue": ks_pval}
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(
        ps_log_amount,
        bins=80,
        density=True,
        alpha=0.6,
        label="PaySim",
        color="steelblue",
    )
    axes[0].hist(
        syn_log_amount,
        bins=80,
        density=True,
        alpha=0.6,
        label="Synthetic",
        color="coral",
    )
    axes[0].set_title(
        f"Amount Distribution (log1p) - {approach_name}\nKS={ks_stat:.4f}, p={ks_pval:.4f}"
    )
    axes[0].set_xlabel("log1p(amount)")
    axes[0].legend()

    # QQ plot
    ps_sample = np.sort(
        ps_log_amount.sample(min(1000, len(ps_log_amount)), random_state=42)
    )
    syn_sample = np.sort(
        syn_log_amount.sample(min(1000, len(syn_log_amount)), random_state=42)
    )
    min_len = min(len(ps_sample), len(syn_sample))
    axes[1].scatter(ps_sample[:min_len], syn_sample[:min_len], s=5, alpha=0.5)
    lims = [
        min(ps_sample.min(), syn_sample.min()),
        max(ps_sample.max(), syn_sample.max()),
    ]
    axes[1].plot(lims, lims, "r--", linewidth=1)
    axes[1].set_xlabel("PaySim quantiles")
    axes[1].set_ylabel("Synthetic quantiles")
    axes[1].set_title("QQ Plot: log1p(amount)")
    plt.tight_layout()
    filename = (
        f"dist_amount_{approach_name}.png" if approach_name else "dist_amount.png"
    )
    plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
    plt.close()

    # --- 3. Balance distributions ---
    for col_ps, col_syn, label in [
        ("oldbalanceOrg", "old_balance_origin", "Old Balance Origin"),
        ("oldbalanceDest", "old_balance_dest", "Old Balance Dest"),
    ]:
        ps_log = np.log1p(ps[col_ps].clip(lower=0))
        syn_log = np.log1p(synth_df[col_syn].clip(lower=0))

        ks_stat, ks_pval = stats.ks_2samp(
            ps_log.sample(min(10000, len(ps_log)), random_state=42),
            syn_log,
        )
        results.append(
            {
                "feature": f"log1p({col_syn})",
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
            }
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            ps_log, bins=80, density=True, alpha=0.6, label="PaySim", color="steelblue"
        )
        ax.hist(
            syn_log, bins=80, density=True, alpha=0.6, label="Synthetic", color="coral"
        )
        ax.set_title(
            f"{label} (log1p) - {approach_name}\nKS={ks_stat:.4f}, p={ks_pval:.4f}"
        )
        ax.legend()
        plt.tight_layout()
        filename = (
            f"dist_{col_syn}_{approach_name}.png"
            if approach_name
            else f"dist_{col_syn}.png"
        )
        plt.savefig(os.path.join(RESULTS_DIR, filename), dpi=150)
        plt.close()

    # --- 4. Fraud rate comparison ---
    ps_fraud_rate = ps["isFraud"].mean()
    syn_fraud_rate = synth_df["is_fraud"].mean()

    ps_fraud_by_type = ps.groupby("type")["isFraud"].mean()
    syn_fraud_by_type = synth_df.groupby("transaction_type")["is_fraud"].mean()

    print("\n=== Fraud Rate Comparison ===")
    print(f"PaySim overall fraud rate:    {ps_fraud_rate:.6f}")
    print(f"Synthetic overall fraud rate: {syn_fraud_rate:.6f}")
    print(f"\nPaySim fraud by type:\n{ps_fraud_by_type}")
    print(f"\nSynthetic fraud by type:\n{syn_fraud_by_type}")

    # --- 5. Feature correlation matrix (synthetic) ---
    numeric_cols = synth_df.select_dtypes(include=[np.number]).columns
    # Exclude non-feature columns
    exclude = {"is_bot", "is_fraud"}
    feat_cols = [c for c in numeric_cols if c not in exclude]
    corr = synth_df[feat_cols + ["is_fraud"]].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=8)
    ax.set_yticklabels(corr.columns, fontsize=8)
    plt.colorbar(im, ax=ax)
    ax.set_title("Feature Correlation Matrix (Synthetic Dataset)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "feature_correlation_matrix.png"), dpi=150)
    plt.close()

    # --- 6. Summary table ---
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "ks_test_results.csv"), index=False)
    print("\n=== KS Test Results ===")
    print(results_df.to_string(index=False))
    print(f"\nAll distribution plots saved to {RESULTS_DIR}/")

    return results_df


if __name__ == "__main__":
    paysim_path = "thesis/data/PS_20174392719_1491204439457_log.csv"
    paysim_df = load_paysim(paysim_path)

    print("=== Generating Unimodal Dataset ===")
    unimodal_df = generate_fraud_dataset_unimodal(num_samples=10000, seed=42)
    unimodal_results = compare_distributions(paysim_df, unimodal_df, "unimodal")

    print("\n=== Generating Multimodal Dataset ===")
    multimodal_df = generate_fraud_dataset(num_samples=10000, seed=42)
    multimodal_results = compare_distributions(paysim_df, multimodal_df, "multimodal")

    # Create comparison summary
    print("\n=== KS Test Comparison ===")
    comparison = pd.merge(
        unimodal_results.add_suffix("_unimodal"),
        multimodal_results.add_suffix("_multimodal"),
        left_on="feature_unimodal",
        right_on="feature_multimodal",
        suffixes=("_unimodal", "_multimodal"),
    )
    comparison["improvement"] = (
        comparison["ks_statistic_multimodal"] - comparison["ks_statistic_unimodal"]
    )
    print(
        comparison[
            [
                "feature_unimodal",
                "ks_statistic_unimodal",
                "ks_statistic_multimodal",
                "improvement",
            ]
        ].to_string(index=False)
    )

    print(f"\nAll distribution plots saved to {RESULTS_DIR}/")
    print("Files generated:")
    print("  Unimodal: dist_*_unimodal.png")
    print("  Multimodal: dist_*_multimodal.png")
    print("  Comparison: ks_test_results.csv")
