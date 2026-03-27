"""Generate dataset sample tables for thesis inclusion."""

import os
import pandas as pd
from fraud_dataset import generate_fraud_dataset

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

df = generate_fraud_dataset(num_samples=10000, seed=42)

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

N = 30

# --- 1. Sample tables (head 30) ---
beh_sample = df[["profile", "is_bot"] + behavioral_features].head(N)
tx_sample = df[["profile", "is_bot", "is_fraud"] + transaction_features].head(N)
comb_sample = df[["profile", "is_bot", "is_fraud"] + combined_features].head(N)

beh_sample.to_csv(os.path.join(RESULTS_DIR, "sample_behavioral.csv"), index=False)
tx_sample.to_csv(os.path.join(RESULTS_DIR, "sample_transactional.csv"), index=False)
comb_sample.to_csv(os.path.join(RESULTS_DIR, "sample_combined.csv"), index=False)

# --- 2. Descriptive statistics ---
beh_stats = df[behavioral_features].describe().round(4)
tx_numeric = [f for f in transaction_features if f != "transaction_type"]
tx_stats = df[tx_numeric].describe().round(4)

beh_stats.to_csv(os.path.join(RESULTS_DIR, "stats_behavioral.csv"))
tx_stats.to_csv(os.path.join(RESULTS_DIR, "stats_transactional.csv"))

# --- 3. Class distribution ---
class_dist = pd.DataFrame(
    {
        "Label": ["is_bot=0", "is_bot=1", "is_fraud=0", "is_fraud=1"],
        "Count": [
            (df["is_bot"] == 0).sum(),
            (df["is_bot"] == 1).sum(),
            (df["is_fraud"] == 0).sum(),
            (df["is_fraud"] == 1).sum(),
        ],
        "Percentage": [
            f"{(df['is_bot'] == 0).mean():.1%}",
            f"{(df['is_bot'] == 1).mean():.1%}",
            f"{(df['is_fraud'] == 0).mean():.1%}",
            f"{(df['is_fraud'] == 1).mean():.1%}",
        ],
    }
)
class_dist.to_csv(os.path.join(RESULTS_DIR, "class_distribution.csv"), index=False)

# --- 4. Profile distribution ---
profile_dist = df["profile"].value_counts().reset_index()
profile_dist.columns = ["Profile", "Count"]
profile_dist["Percentage"] = (profile_dist["Count"] / len(df) * 100).round(1).astype(
    str
) + "%"
profile_dist.to_csv(os.path.join(RESULTS_DIR, "profile_distribution.csv"), index=False)

# --- 5. Generate markdown tables ---
md_path = os.path.join(RESULTS_DIR, "dataset_tables.md")
with open(md_path, "w") as f:
    f.write("# Dataset Sample Tables for Thesis\n\n")

    # Behavioral sample (show 15 rows for readability)
    f.write("## Behavioral Feature Set — Sample (first 15 rows, seed=42)\n\n")
    beh_short = beh_sample.head(15).copy()
    beh_short["session_duration_sec"] = beh_short["session_duration_sec"].round(1)
    beh_short["checkout_velocity_sec"] = beh_short["checkout_velocity_sec"].round(1)
    beh_short["mouse_speed_variance"] = beh_short["mouse_speed_variance"].round(2)
    beh_short["keystroke_flight_time_ms"] = beh_short["keystroke_flight_time_ms"].round(
        1
    )
    f.write(beh_short.to_markdown(index=False))
    f.write("\n\n")

    # Transactional sample
    f.write("## Transactional Feature Set — Sample (first 15 rows, seed=42)\n\n")
    tx_short = tx_sample.head(15).copy()
    tx_short["amount"] = tx_short["amount"].round(2)
    tx_short["old_balance_origin"] = tx_short["old_balance_origin"].round(2)
    tx_short["new_balance_origin"] = tx_short["new_balance_origin"].round(2)
    tx_short["old_balance_dest"] = tx_short["old_balance_dest"].round(2)
    tx_short["new_balance_dest"] = tx_short["new_balance_dest"].round(2)
    tx_short["balance_drain_ratio"] = tx_short["balance_drain_ratio"].round(4)
    f.write(tx_short.to_markdown(index=False))
    f.write("\n\n")

    # Descriptive stats - Behavioral
    f.write("## Behavioral Features — Descriptive Statistics\n\n")
    f.write(beh_stats.to_markdown())
    f.write("\n\n")

    # Descriptive stats - Transactional
    f.write("## Transactional Features — Descriptive Statistics\n\n")
    f.write(tx_stats.to_markdown())
    f.write("\n\n")

    # Class distribution
    f.write("## Class Distribution (seed=42, N=10,000)\n\n")
    f.write(class_dist.to_markdown(index=False))
    f.write("\n\n")

    # Profile distribution
    f.write("## Profile Distribution (seed=42, N=10,000)\n\n")
    f.write(profile_dist.to_markdown(index=False))
    f.write("\n")

print(f"Saved CSVs and markdown to {RESULTS_DIR}/")
print("  - sample_behavioral.csv")
print("  - sample_transactional.csv")
print("  - sample_combined.csv")
print("  - stats_behavioral.csv")
print("  - stats_transactional.csv")
print("  - class_distribution.csv")
print("  - profile_distribution.csv")
print("  - dataset_tables.md")
