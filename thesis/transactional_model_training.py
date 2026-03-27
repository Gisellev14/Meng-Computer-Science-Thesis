from fraud_dataset import generate_fraud_dataset
from training_pipeline import run_training_pipeline

df = generate_fraud_dataset(num_samples=10000, seed=42)

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

results_transaction = run_training_pipeline(
    df=df,
    feature_cols=transaction_features,
    target_col="is_fraud",
    model_name="transactional",
    algorithm="random_forest",
)
