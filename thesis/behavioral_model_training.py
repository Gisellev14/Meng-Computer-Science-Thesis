from fraud_dataset import generate_fraud_dataset
from training_pipeline import run_training_pipeline

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

results_behavioral = run_training_pipeline(
    df=df,
    feature_cols=behavioral_features,
    target_col="is_bot",
    model_name="behavioral",
    algorithm="random_forest",
)
