import os
import pandas as pd
import numpy as np

from behavioral_dataset import generate_behavioral_data, _clip_positive
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


def _generate_amount_multimodal(rng, paysim_params, is_bot=False):
    """Generate amounts using multimodal mixture model to match PaySim distribution."""
    peak_centers = np.array(paysim_params["amount_peak_centers"])
    peak_weights = np.array(paysim_params["amount_peak_weights"])
    peak_std = paysim_params["amount_peak_std"]

    # Sample which peak to use
    peak_idx = rng.choice(len(peak_centers), p=peak_weights)
    selected_center = peak_centers[peak_idx]

    # Add bot shift if needed
    if is_bot:
        selected_center += 0.5

    # Generate amount from selected peak
    log_amount = rng.normal(loc=selected_center, scale=peak_std)
    amount = np.expm1(log_amount)

    return amount


def _generate_balance_zero_inflated(rng, paysim_params, balance_type="origin"):
    """Generate balances with zero-inflation to match PaySim's spike at zero."""
    if balance_type == "origin":
        zero_rate = paysim_params["origin_balance_zero_rate"]
        mean = paysim_params["old_balance_origin_log_mean"]
        std = paysim_params["old_balance_origin_log_std"]
    else:  # destination
        zero_rate = paysim_params["dest_balance_zero_rate"]
        mean = paysim_params["old_balance_dest_log_mean"]
        std = paysim_params["old_balance_dest_log_std"]

    # Sample zero or non-zero balance
    if rng.random() < zero_rate:
        return 0.0
    else:
        return np.expm1(rng.normal(loc=mean, scale=std))


def compute_derived_features(tx_features):
    """
    Compute derived features that capture fraud signals.
    """
    old_bal = tx_features["old_balance_origin"]
    amount = tx_features["amount"]

    # Ratio of amount to origin balance (how much is being drained)
    if old_bal > 0:
        tx_features["balance_drain_ratio"] = amount / old_bal
    else:
        tx_features["balance_drain_ratio"] = 0.0

    # Binary: is this a full drain?
    tx_features["is_full_drain"] = int(
        tx_features["new_balance_origin"] == 0 and old_bal > 0
    )

    # Binary: did destination balance stay unchanged?
    tx_features["dest_balance_unchanged"] = int(
        tx_features["old_balance_dest"] == tx_features["new_balance_dest"]
    )

    return tx_features


def generate_transaction_features(rng, is_bot, profile, paysim_params):
    tx_probs = paysim_params["transaction_type_probs"]

    transaction_types = list(tx_probs.keys())
    base_probs = np.array(list(tx_probs.values()), dtype=float)

    # Bot transaction type adjustment: bots prefer TRANSFER/CASH_OUT
    if is_bot == 1:
        adjusted_probs = base_probs.copy()

        type_index = {t: i for i, t in enumerate(transaction_types)}
        if "TRANSFER" in type_index:
            adjusted_probs[type_index["TRANSFER"]] *= 2.2
        if "CASH_OUT" in type_index:
            adjusted_probs[type_index["CASH_OUT"]] *= 2.0
        if "PAYMENT" in type_index:
            adjusted_probs[type_index["PAYMENT"]] *= 0.3

        adjusted_probs = adjusted_probs / adjusted_probs.sum()
    else:
        adjusted_probs = base_probs

    tx_type = rng.choice(transaction_types, p=adjusted_probs)

    # Amount calibrated from PaySim multimodal distribution, shifted for bots
    amount = _generate_amount_multimodal(rng, paysim_params, is_bot)
    amount = _clip_positive(amount, low=1.0)

    # Generate balances with zero-inflation to match PaySim's distribution
    old_balance_origin = _generate_balance_zero_inflated(rng, paysim_params, "origin")
    old_balance_dest = _generate_balance_zero_inflated(
        rng, paysim_params, "destination"
    )

    # In real banking, most transactions cannot exceed available balance.
    # Cap amount to a random fraction of balance for realistic spend behavior.
    # This prevents accidental full-drains; deliberate drains are handled
    # later by apply_risky_patterns (which simulates fraud-like patterns).
    if old_balance_origin > 0:
        max_spend_ratio = rng.uniform(0.05, 0.85)
        amount = min(amount, old_balance_origin * max_spend_ratio)
        amount = _clip_positive(amount, low=1.0)

    new_balance_origin = max(old_balance_origin - amount, 0.0)
    new_balance_dest = old_balance_dest + amount

    return {
        "transaction_type": tx_type,
        "amount": amount,
        "old_balance_origin": old_balance_origin,
        "new_balance_origin": new_balance_origin,
        "old_balance_dest": old_balance_dest,
        "new_balance_dest": new_balance_dest,
    }


def assign_fraud_label(
    rng,
    profile,
    tx_type,
    amount,
    is_datacenter_ip,
    impossible_travel_flag,
    is_full_drain,
    dest_unchanged,
    paysim_params,
):
    # Base fraud rate from PaySim dataset
    fraud_prob = paysim_params["fraud_base_rate"]

    # Bot-related risk (use clean profile, not noisy is_bot)
    is_bot_clean = 1 if profile.startswith("bot_") else 0
    if is_bot_clean == 1:
        fraud_prob += 0.20

    if profile == "bot_standard":
        fraud_prob += 0.18
    elif profile == "bot_stealth":
        fraud_prob += 0.08

    # High-risk transaction types
    if tx_type in ["TRANSFER", "CASH_OUT"]:
        fraud_prob += 0.18

    # Amount thresholds as fraud signals
    if amount > 5000:
        fraud_prob += 0.06
    if amount > 15000:
        fraud_prob += 0.08

    # Network indicators
    if is_datacenter_ip == 1:
        fraud_prob += 0.10
    if impossible_travel_flag == 1:
        fraud_prob += 0.04

    # Transaction pattern risk factors (generated independently via apply_risky_patterns)
    if is_full_drain == 1:
        fraud_prob += 0.22
    if dest_unchanged == 1:
        fraud_prob += 0.10

    fraud_prob = min(fraud_prob, 0.95)

    return int(rng.random() < fraud_prob)


def generate_fraud_dataset(
    num_samples=10000,
    seed=42,
    paysim_path=_DEFAULT_PAYSIM_PATH,
):
    rng = np.random.default_rng(seed)

    # 1. Generate behavior dataset
    df = generate_behavioral_data(
        num_samples=num_samples, seed=seed, return_profile=True
    )

    # 2. Load PaySim calibration
    paysim_df = load_paysim(paysim_path)
    paysim_params = extract_paysim_parameters(paysim_df)

    # 3. Generate transaction features + fraud label
    transaction_rows = []
    fraud_labels = []

    for _, row in df.iterrows():
        tx_features = generate_transaction_features(
            rng=rng,
            is_bot=row["is_bot"],
            profile=row["profile"],
            paysim_params=paysim_params,
        )

        # Apply risky patterns BEFORE fraud label (no leakage — uses only tx features)
        tx_features = apply_risky_patterns(rng=rng, tx_features=tx_features)

        # Compute derived features BEFORE fraud label
        tx_features = compute_derived_features(tx_features)

        # Now assign fraud label using the derived features as risk factors
        fraud_label = assign_fraud_label(
            rng=rng,
            profile=row["profile"],
            tx_type=tx_features["transaction_type"],
            amount=tx_features["amount"],
            is_datacenter_ip=row["is_datacenter_ip"],
            impossible_travel_flag=row["impossible_travel_flag"],
            is_full_drain=tx_features["is_full_drain"],
            dest_unchanged=tx_features["dest_balance_unchanged"],
            paysim_params=paysim_params,
        )

        transaction_rows.append(tx_features)
        fraud_labels.append(fraud_label)

    tx_df = pd.DataFrame(transaction_rows)
    df_final = pd.concat([df.reset_index(drop=True), tx_df], axis=1)
    df_final["is_fraud"] = fraud_labels

    return df_final


if __name__ == "__main__":
    df_fraud = generate_fraud_dataset()
    print(df_fraud.head())
    print("\n=== is_bot distribution ===")
    print(df_fraud["is_bot"].value_counts(normalize=True))
    print("\n=== is_fraud distribution ===")
    print(df_fraud["is_fraud"].value_counts(normalize=True))
    print("\n=== transaction type distribution ===")
    print(df_fraud["transaction_type"].value_counts(normalize=True))
    print("\n=== bot fraud percentage ===")
    print(df_fraud.groupby("is_bot")["is_fraud"].mean())
    print("\n=== fraud per profile ===")
    print(df_fraud.groupby("profile")["is_fraud"].mean())
    print("\n=== fraud per transaction type ===")
    print(df_fraud.groupby("transaction_type")["is_fraud"].mean())
