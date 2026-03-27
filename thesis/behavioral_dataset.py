import pandas as pd
import numpy as np
import uuid


def _clip_positive(x, low=0.1):
    return float(max(low, x))


def generate_behavioral_data(
    num_samples=10000,  # amount of samples
    bot_ratio=0.15,  # total bot ratio
    stealth_bot_ratio=0.30,  # % of stealth bots
    fast_human_ratio=0.15,  # % of fast humans (power users)
    label_noise=0.03,  # label noise (FP/FN)
    measurement_noise=0.05,  # measurement errors
    return_profile=True,  # return profile (just for analysis)
    seed=42,  # for reproducibility
):
    """
    Generates a synthetic dataset for bot vs. human behavioral detection.
    """

    data = []
    rng = np.random.default_rng(seed)

    # Calculate number of bots and humans based on the ratio
    num_bots = int(num_samples * bot_ratio)
    num_humans = num_samples - num_bots

    num_stealth_bots = int(num_bots * stealth_bot_ratio)
    num_standard_bots = num_bots - num_stealth_bots

    num_fast_humans = int(num_humans * fast_human_ratio)
    num_standard_humans = num_humans - num_fast_humans

    def add_measurement_noise(row: dict):
        for k in [
            "session_duration_sec",
            "checkout_velocity_sec",
            "mouse_speed_variance",
            "keystroke_flight_time_ms",
        ]:
            val = float(row[k])
            noise = rng.normal(0, measurement_noise)
            row[k] = _clip_positive(
                val * (1.0 + noise), low=0.0 if k == "keystroke_flight_time_ms" else 0.1
            )
        return row

    # 1. Generate Human Data (Label: 0)
    # HUMANS: standard
    for _ in range(num_standard_humans):
        row = {
            "transaction_id": str(uuid.uuid4()),
            "is_bot": 0,
            "profile": "human_normal",
            "session_duration_sec": _clip_positive(
                rng.lognormal(mean=np.log(120), sigma=0.35)
            ),
            "checkout_velocity_sec": _clip_positive(
                rng.lognormal(mean=np.log(45), sigma=0.35)
            ),
            "mouse_speed_variance": float(rng.normal(loc=95, scale=25)),
            "keystroke_flight_time_ms": float(rng.normal(loc=150, scale=50)),
            "impossible_travel_flag": int(rng.random() < 0.02),
            "is_datacenter_ip": 0,
            "pages_visited": int(max(1, rng.normal(loc=6, scale=2))),
        }
        data.append(add_measurement_noise(row))

    # HUMANS: fast/power users (repeat customers who navigate quickly)
    for _ in range(num_fast_humans):
        row = {
            "transaction_id": str(uuid.uuid4()),
            "is_bot": 0,
            "profile": "human_fast",
            "session_duration_sec": _clip_positive(
                rng.lognormal(mean=np.log(55), sigma=0.40)
            ),
            "checkout_velocity_sec": _clip_positive(
                rng.lognormal(mean=np.log(18), sigma=0.45)
            ),
            "mouse_speed_variance": float(rng.normal(loc=80, scale=30)),
            "keystroke_flight_time_ms": float(rng.normal(loc=150, scale=50)),
            "impossible_travel_flag": int(rng.random() < 0.03),  # VPN/roaming
            "is_datacenter_ip": int(rng.random() < 0.02),  # rare but possible
            "pages_visited": int(max(1, rng.normal(loc=3, scale=1.5))),
        }
        data.append(add_measurement_noise(row))

    # 2. Generate Bot Data (Label: 1)
    # BOTS: standard (scripted automation, no human emulation)
    for _ in range(num_standard_bots):
        row = {
            "transaction_id": str(uuid.uuid4()),
            "is_bot": 1,
            "profile": "bot_standard",
            "session_duration_sec": _clip_positive(
                rng.lognormal(mean=np.log(6), sigma=0.55)
            ),
            "checkout_velocity_sec": _clip_positive(
                rng.lognormal(mean=np.log(2.5), sigma=0.60)
            ),
            "mouse_speed_variance": float(
                rng.normal(loc=8, scale=6)
            ),  # near-zero with noise
            "keystroke_flight_time_ms": float(rng.normal(loc=50, scale=5)),
            "impossible_travel_flag": int(rng.random() < 0.60),
            "is_datacenter_ip": int(rng.random() < 0.70),
            "pages_visited": int(rng.choice([1, 2, 3], p=[0.65, 0.25, 0.10])),
        }
        data.append(add_measurement_noise(row))

    # BOTS: stealth (human-emulating automation, e.g., headless browsers with puppeteer)
    for _ in range(num_stealth_bots):
        row = {
            "transaction_id": str(uuid.uuid4()),
            "is_bot": 1,
            "profile": "bot_stealth",
            "session_duration_sec": _clip_positive(
                rng.lognormal(mean=np.log(75), sigma=0.30)
            ),
            "checkout_velocity_sec": _clip_positive(
                rng.lognormal(mean=np.log(22), sigma=0.30)
            ),
            # Stealth bots emulate mouse/keyboard, but show lower real variability
            "mouse_speed_variance": float(rng.normal(loc=55, scale=18)),
            "keystroke_flight_time_ms": float(rng.normal(loc=125, scale=35)),
            # Higher probability of proxies/datacenter
            "impossible_travel_flag": int(rng.random() < 0.25),
            "is_datacenter_ip": int(rng.random() < 0.30),
            "pages_visited": int(max(1, rng.normal(loc=4, scale=1.8))),
        }
        data.append(add_measurement_noise(row))

    # Combine, shuffle, and clean up the dataset
    df = pd.DataFrame(data)

    # Ensure no negative times from normal distributions
    df["session_duration_sec"] = df["session_duration_sec"].clip(lower=0.1)
    df["checkout_velocity_sec"] = df["checkout_velocity_sec"].clip(lower=0.1)
    df["keystroke_flight_time_ms"] = df["keystroke_flight_time_ms"].clip(lower=0.0)
    df["pages_visited"] = df["pages_visited"].clip(lower=1)
    df["mouse_speed_variance"] = df["mouse_speed_variance"].clip(lower=0.0)

    # Label noise
    if label_noise > 0:
        flip_rng = np.random.default_rng(seed + 1)
        flip = flip_rng.random(len(df)) < label_noise
        df.loc[flip, "is_bot"] = 1 - df.loc[flip, "is_bot"]

    # Shuffle the dataset so bots and humans are mixed
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    if not return_profile:
        df = df.drop(columns=["profile"])

    return df


if __name__ == "__main__":
    # Generate 10,000 records with a 15% bot attack rate
    df_synthetic = generate_behavioral_data(
        num_samples=10000,
        seed=42,
        bot_ratio=0.15,
        stealth_bot_ratio=0.30,
        fast_human_ratio=0.15,
        label_noise=0.03,
        measurement_noise=0.05,
        return_profile=True,
    )

    # Display the first few rows and basic statistics
    print(df_synthetic.head())
    print("\nTarget Class Distribution:")
    print(df_synthetic["is_bot"].value_counts(normalize=True))
    print("\nProfile Distribution:")
    print(df_synthetic["profile"].value_counts(normalize=True))
