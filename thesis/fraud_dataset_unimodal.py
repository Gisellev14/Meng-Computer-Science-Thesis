import os
import pandas as pd
import numpy as np

from behavioral_dataset import generate_behavioral_data
from paysim_dataset import load_paysim, extract_paysim_parameters

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_PAYSIM_PATH = os.path.join(
    _SCRIPT_DIR, "data", "PS_20174392719_1491204439457_log.csv"
)


def apply_risky_patterns(rng, tx_features):
    """
    Apply risky transaction patterns INDEPENDENTLY of both fraud label and is_bot.
    Patterns are driven only by observable transaction characteristics (type, amount)
    to avoid leaking bot/fraud information into features.

      - Account draining is most common in TRANSFER and CASH_OUT (Europol IOCTA, 2023)
    """
    tx_type = tx_features["transaction_type"]
    amount = tx_features["amount"]
    old_bal = tx_features["old_balance_origin"]

    # Base drain probability depends only on transaction type and amount ratio
    # High-risk types (TRANSFER, CASH_OUT) have higher drain rates in PaySim
    if tx_type in ["TRANSFER", "CASH_OUT"]:
        drain_prob = 0.12
        dest_unchanged_prob = 0.06
    else:
        drain_prob = 0.03
        dest_unchanged_prob = 0.01

    # Large transactions relative to balance are more likely to drain
    if old_bal > 0 and amount / old_bal > 0.8:
        drain_prob += 0.15
        dest_unchanged_prob += 0.05

    # Apply patterns probabilistically
    if rng.random() < drain_prob:
        tx_features["amount"] = tx_features["old_balance_origin"]
        tx_features["new_balance_origin"] = 0.0

    if rng.random() < dest_unchanged_prob:
        tx_features["new_balance_dest"] = tx_features["old_balance_dest"]

    return tx_features


def _generate_amount_unimodal(rng, paysim_params, is_bot=False):
    """Generate amounts using single log-normal distribution (unimodal approach)."""
    # Use overall PaySim parameters for single distribution
    mean = paysim_params["amount_log_mean"]
    std = paysim_params["amount_log_std"]

    # Add bot shift if needed
    if is_bot:
        mean += 0.5

    # Generate amount from single log-normal
    log_amount = rng.normal(loc=mean, scale=std)
    amount = np.expm1(log_amount)

    return amount


def _generate_balance_unimodal(rng, paysim_params, balance_type="origin"):
    """Generate balances using simple log-normal (unimodal approach)."""
    if balance_type == "origin":
        mean = paysim_params["old_balance_origin_log_mean"]
        std = paysim_params["old_balance_origin_log_std"]
    else:  # destination
        mean = paysim_params["old_balance_dest_log_mean"]
        std = paysim_params["old_balance_dest_log_std"]

    # Simple log-normal without zero-inflation
    balance = np.expm1(rng.normal(loc=mean, scale=std))

    return balance


def compute_derived_features(tx_features):
    """
    Compute derived features that capture fraud signals.
    """
    old_bal = tx_features["old_balance_origin"]
    amount = tx_features["amount"]

    # Balance drain ratio (how much of origin balance is used)
    if old_bal > 0:
        drain_ratio = amount / old_bal
    else:
        drain_ratio = 0.0

    # Full drain indicator
    is_full_drain = 1.0 if old_bal > 0 and amount >= old_bal * 0.99 else 0.0

    tx_features["balance_drain_ratio"] = drain_ratio
    tx_features["is_full_drain"] = is_full_drain

    return tx_features


def generate_transaction_features_unimodal(
    n_samples,
    paysim_params,
    behavioral_df,
    rng,
):
    """
    Generate transaction features using unimodal approach.
    """
    profiles = behavioral_df["profile"].values
    is_bot_flags = behavioral_df["is_bot"].values

    # Transaction type probabilities (adjusted for bots)
    base_probs = np.array(
        [0.43, 0.10, 0.45, 0.02]
    )  # PAYMENT, TRANSFER, CASH_OUT, DEBIT

    tx_features = []

    for i in range(n_samples):
        profile = profiles[i]
        is_bot = is_bot_flags[i]

        # Adjust transaction type probabilities for bots
        if is_bot:
            # Bots favor TRANSFER and CASH_OUT (fraud-prone types)
            bot_multiplier = {
                "PAYMENT": 0.5,
                "TRANSFER": 2.2,
                "CASH_OUT": 2.0,
                "DEBIT": 0.5,
            }
            adjusted_probs = np.array(
                [
                    base_probs[j] * bot_multiplier[p]
                    for j, p in enumerate(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"])
                ]
            )
            adjusted_probs = adjusted_probs / adjusted_probs.sum()
        else:
            adjusted_probs = base_probs

        # Sample transaction type
        tx_type = rng.choice(
            ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"], p=adjusted_probs
        )

        # Generate amount (unimodal)
        amount = _generate_amount_unimodal(rng, paysim_params, is_bot)

        # Generate balances (unimodal)
        old_balance_orig = _generate_balance_unimodal(rng, paysim_params, "origin")
        old_balance_dest = _generate_balance_unimodal(rng, paysim_params, "dest")

        # Apply amount-to-balance capping to prevent overspend
        max_amount = old_balance_orig * rng.uniform(0.05, 0.85)
        amount = min(amount, max_amount)

        # Calculate new balances
        new_balance_orig = old_balance_orig - amount
        new_balance_dest = old_balance_dest + amount

        # Create transaction features
        tx = {
            "profile": profile,
            "is_bot": is_bot,
            "transaction_type": tx_type,
            "amount": amount,
            "old_balance_origin": old_balance_orig,
            "new_balance_origin": new_balance_orig,
            "old_balance_dest": old_balance_dest,
            "new_balance_dest": new_balance_dest,
        }

        # Apply risky patterns (independent of bot/fraud labels)
        tx = apply_risky_patterns(rng, tx)

        # Compute derived features
        tx = compute_derived_features(tx)

        tx_features.append(tx)

    return pd.DataFrame(tx_features)


def assign_fraud_labels(tx_df):
    """
    Assign fraud labels based on additive probability model.
    """
    fraud_probs = []

    for _, tx in tx_df.iterrows():
        prob = 0.0

        # Base fraud rate (2% overall)
        prob += 0.02

        # Bot profile increases fraud probability
        if tx["is_bot"]:
            prob += 0.15

        # Transaction type risk
        if tx["transaction_type"] in ["TRANSFER", "CASH_OUT"]:
            prob += 0.10

        # Amount threshold (large amounts)
        if tx["amount"] > 200000:
            prob += 0.15

        # Network indicators
        if tx.get("is_datacenter_ip", 0) == 1:
            prob += 0.10
        if tx.get("impossible_travel_flag", 0) == 1:
            prob += 0.05

        # Drain patterns
        if tx["is_full_drain"] == 1:
            prob += 0.20
        if tx["balance_drain_ratio"] > 0.9:
            prob += 0.10

        # Cap probability
        prob = min(prob, 0.95)

        fraud_probs.append(prob)

    # Assign labels based on probabilities
    rng = np.random.default_rng(42)  # Fixed seed for reproducible labels
    fraud_labels = rng.random(len(fraud_probs)) < np.array(fraud_probs)

    return fraud_labels.astype(int)


def generate_fraud_dataset_unimodal(
    num_samples=10000,
    paysim_path=_DEFAULT_PAYSIM_PATH,
    seed=42,
):
    """
    Generate synthetic fraud dataset using unimodal approach.
    """
    rng = np.random.default_rng(seed)

    # Load PaySim and extract parameters
    paysim_df = load_paysim(paysim_path)
    paysim_params = extract_paysim_parameters(paysim_df)

    # Generate behavioral features
    behavioral_df = generate_behavioral_data(num_samples, seed)

    # Generate transaction features (unimodal)
    tx_df = generate_transaction_features_unimodal(
        num_samples, paysim_params, behavioral_df, rng
    )

    # Assign fraud labels
    fraud_labels = assign_fraud_labels(tx_df)
    tx_df["is_fraud"] = fraud_labels

    # Add label noise
    noise_rate = 0.03
    n_flip = int(noise_rate * len(tx_df))
    flip_indices = rng.choice(len(tx_df), n_flip, replace=False)
    tx_df.loc[flip_indices, "is_fraud"] = 1 - tx_df.loc[flip_indices, "is_fraud"]

    return tx_df
