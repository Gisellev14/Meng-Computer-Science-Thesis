import pandas as pd
import numpy as np
from scipy.signal import find_peaks


def load_paysim(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def extract_paysim_parameters(df: pd.DataFrame) -> dict:
    """
    Extracts calibration parameters from PaySim
    to simulate realistic fintech transaction behavior.
    """

    # Only transaction types relevant for fintech fraud modeling
    valid_types = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]
    df = df[df["type"].isin(valid_types)].copy()

    tx_type_probs = (
        df["type"]
        .value_counts(normalize=True)
        .reindex(valid_types, fill_value=0)
        .to_dict()
    )

    # Log-normal fit for amount
    log_amount = np.log1p(df["amount"])
    amount_log_mean = float(log_amount.mean())
    amount_log_std = float(log_amount.std())

    # Log-normal fit for balances
    log_old_balance_org = np.log1p(df["oldbalanceOrg"])
    log_old_balance_dest = np.log1p(df["oldbalanceDest"])

    # --- Empirical drain statistics for calibration ---
    # Non-fraud transactions: what fraction naturally drains origin to 0?
    non_fraud = df[df["isFraud"] == 0]
    has_balance = non_fraud["oldbalanceOrg"] > 0
    legit_drain_rate = (
        float((non_fraud.loc[has_balance, "newbalanceOrig"] == 0).mean())
        if has_balance.sum() > 0
        else 0.0
    )

    # Fraud transactions: what fraction drains origin to 0?
    fraud = df[df["isFraud"] == 1]
    has_balance_fraud = fraud["oldbalanceOrg"] > 0
    fraud_drain_rate = (
        float((fraud.loc[has_balance_fraud, "newbalanceOrig"] == 0).mean())
        if has_balance_fraud.sum() > 0
        else 0.0
    )

    # Median amount-to-balance ratio for legit vs fraud
    legit_with_bal = non_fraud[non_fraud["oldbalanceOrg"] > 0]
    fraud_with_bal = fraud[fraud["oldbalanceOrg"] > 0]
    legit_amt_bal_ratio = (
        float((legit_with_bal["amount"] / legit_with_bal["oldbalanceOrg"]).median())
        if len(legit_with_bal) > 0
        else 0.5
    )
    fraud_amt_bal_ratio = (
        float((fraud_with_bal["amount"] / fraud_with_bal["oldbalanceOrg"]).median())
        if len(fraud_with_bal) > 0
        else 1.0
    )

    # Destination balance unchanged rate
    legit_dest_unchanged = float(
        (non_fraud["oldbalanceDest"] == non_fraud["newbalanceDest"]).mean()
    )
    fraud_dest_unchanged = (
        float((fraud["oldbalanceDest"] == fraud["newbalanceDest"]).mean())
        if len(fraud) > 0
        else 0.0
    )

    # Enhanced distribution parameters for better synthetic matching
    # Zero-inflation rates for balances
    origin_zero_rate = float((df["oldbalanceOrg"] == 0).mean())
    dest_zero_rate = float((df["oldbalanceDest"] == 0).mean())

    # Amount distribution peaks (for multimodal generation)
    amount_hist, amount_bins = np.histogram(log_amount, bins=50, density=True)
    # Find peaks in the amount distribution
    peaks, properties = find_peaks(amount_hist, height=0.01, distance=3)
    peak_centers = amount_bins[peaks] + np.diff(amount_bins)[0] / 2
    peak_heights = amount_hist[peaks]

    # Normalize peak heights to create mixture weights
    if len(peak_heights) > 0:
        peak_weights = peak_heights / peak_heights.sum()
    else:
        # Fallback to single mode if no peaks found
        peak_centers = np.array([amount_log_mean])
        peak_weights = np.array([1.0])

    params = {
        "transaction_type_probs": tx_type_probs,
        "amount_log_mean": amount_log_mean,
        "amount_log_std": amount_log_std,
        "old_balance_origin_log_mean": float(log_old_balance_org.mean()),
        "old_balance_origin_log_std": float(log_old_balance_org.std()),
        "old_balance_dest_log_mean": float(log_old_balance_dest.mean()),
        "old_balance_dest_log_std": float(log_old_balance_dest.std()),
        "fraud_base_rate": float(df["isFraud"].mean()),
        # Enhanced distribution parameters
        "origin_balance_zero_rate": origin_zero_rate,
        "dest_balance_zero_rate": dest_zero_rate,
        "amount_peak_centers": peak_centers.tolist(),
        "amount_peak_weights": peak_weights.tolist(),
        "amount_peak_std": float(amount_log_std * 0.3),  # Narrower std for each peak
        # Calibration targets for synthetic data validation
        "legit_drain_rate": legit_drain_rate,
        "fraud_drain_rate": fraud_drain_rate,
        "legit_amt_bal_ratio_median": legit_amt_bal_ratio,
        "fraud_amt_bal_ratio_median": fraud_amt_bal_ratio,
        "legit_dest_unchanged_rate": legit_dest_unchanged,
        "fraud_dest_unchanged_rate": fraud_dest_unchanged,
    }

    return params


if __name__ == "__main__":
    paysim = load_paysim("data/PS_20174392719_1491204439457_log.csv")
    params = extract_paysim_parameters(paysim)

    print("=== PaySim Parameters ===")
    for k, v in params.items():
        print(f"{k}: {v}")
