from fraud_dataset import generate_fraud_dataset
from training_pipeline import run_training_pipeline

df = generate_fraud_dataset(num_samples=10000, seed=42)

combined_features = [
    "session_duration_sec",
    "checkout_velocity_sec",
    "mouse_speed_variance",
    "keystroke_flight_time_ms",
    "impossible_travel_flag",
    "is_datacenter_ip",
    "pages_visited",
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

results_combined = run_training_pipeline(
    df=df,
    feature_cols=combined_features,
    target_col="is_fraud",
    model_name="combined",
    algorithm="random_forest",
)
