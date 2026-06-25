"""
Complete data pipeline: flatten JSON columns, remove/mask PII,
and apply all cleaning transformations.

Transformations applied
-----------------------
  - Drop zero-variance columns (device_is_proxy, device_is_botnet,
    client_identity_approved)
  - Drop label-leaking columns (fraud_decision_created_at,
    client_cupo_state, credit_check_status)
  - Engineer timestamp features (app_hour, app_day_of_week,
    device_age_days, device_reg_to_app_days)
  - Normalize device_os → device_os_family + device_os_version
  - Parse device_screen → device_screen_width + device_screen_height
  - Convert boolean strings to 0/1 integers
  - Bucket high-cardinality categoricals:
      device_isp and device_ip_region → top-20 + OTHER
      device_browser_language → ES / EN / PT / OTHER
      device_ip_city dropped (too sparse)
  - Null imputation:
      Loan columns (48% null) → fill 0 + has_loan_data flag
      Cupo / delinquency (28% null) → fill 0 + has_cupo_data flag
      credit_pd_score (32% null) → median + credit_pd_score_missing flag
      Remaining ~1% nulls → median (numeric) or mode (categorical)

Input:  Fraud data/fraud_combined.csv
Output: Fraud data/fraud_cleaned.csv
"""

import csv
import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

csv.field_size_limit(10_000_000)

INPUT  = Path(__file__).parent.parent / "Fraud data" / "fraud_combined.csv"
OUTPUT = Path(__file__).parent.parent / "Fraud data" / "fraud_cleaned.csv"

TOP_N = 20  # how many values to keep before collapsing to OTHER


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get(obj, *keys, default=None):
    for k in keys:
        if not isinstance(obj, dict):
            return default
        obj = obj.get(k)
        if obj is None:
            return default
    return obj


def _bool_int(val):
    if val is None:
        return None
    if isinstance(val, bool):
        return int(val)
    s = str(val).strip().lower()
    if s in ("true", "1", "yes"):
        return 1
    if s in ("false", "0", "no"):
        return 0
    return None


def _to_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _to_int(val):
    try:
        return int(float(val))
    except (TypeError, ValueError):
        return None


def _hash_id(val):
    if not val:
        return None
    return hashlib.sha256(val.encode()).hexdigest()[:16]


def _parse_ts(s):
    """Parse a timestamp string to a UTC-aware datetime."""
    if not s or s.strip() in ("", "None", "null"):
        return None
    s = s.strip()
    s = re.sub(r'Z$', '+00:00', s)
    # "2023-03-09 23:34:23.933 +0100" → strip space, colon-ify tz
    s = re.sub(r'\s+([+-])(\d{2}):?(\d{2})$', r'\1\2:\3', s)
    # Unify date/time separator
    s = re.sub(r'^(\d{4}-\d{2}-\d{2}) ', r'\1T', s)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            return datetime.strptime(s, fmt).astimezone(timezone.utc)
        except ValueError:
            continue
    return None


OS_FAMILIES = [
    ("ANDROID",  "ANDROID"),
    ("IOS",      "IOS"),
    ("IPHONE",   "IOS"),
    ("IPAD",     "IOS"),
    ("WINDOWS",  "WINDOWS"),
    ("MAC",      "MAC"),
    ("LINUX",    "LINUX"),
    ("CHROMEOS", "CHROMEOS"),
    ("UBUNTU",   "LINUX"),
]

def _parse_os(val):
    if not val or str(val).strip() in ("", "None"):
        return None, None
    v = str(val).upper().strip()
    family = "OTHER"
    for keyword, fam in OS_FAMILIES:
        if keyword in v:
            family = fam
            break
    nums = re.findall(r'\d+', v)
    return family, (int(nums[0]) if nums else None)


def _parse_screen(val):
    if not val or str(val).strip() in ("", "None"):
        return None, None
    m = re.search(r'(\d+)[xX](\d+)', str(val))
    return (int(m.group(1)), int(m.group(2))) if m else (None, None)


def _bucket_language(val):
    if not val or str(val).strip() in ("", "None"):
        return None
    v = str(val).upper().strip()
    for prefix, bucket in (("ES", "ES"), ("EN", "EN"), ("PT", "PT")):
        if v.startswith(prefix):
            return bucket
    return "OTHER"


# ---------------------------------------------------------------------------
# Phase 1: JSON extractors
# ---------------------------------------------------------------------------

def extract_application(raw):
    if not raw or raw.strip() in ("", "null"):
        return {}
    try:
        d = json.loads(raw)
    except Exception:
        return {}

    loan        = d.get("loan") or {}
    client      = d.get("client") or {}
    credit      = d.get("creditCheck") or {}
    journey     = d.get("journey") or {}
    cupo        = client.get("addiCupo") or {}
    status_obj  = client.get("clientStatus") or {}
    credit_score = credit.get("score") or {}
    prev_loans  = client.get("loans") or []

    max_dpd = None
    if prev_loans:
        dpds = [_to_int(l.get("currentDaysPastDue")) for l in prev_loans
                if l.get("currentDaysPastDue") is not None]
        max_dpd = max(dpds) if dpds else None

    return {
        "loan_term":              _to_int(loan.get("term")),
        "loan_total_amount":      _to_float(loan.get("totalAmount")),
        "loan_approved_amount":   _to_float(loan.get("approvedAmount")),
        "loan_requested_amount":  _to_float(d.get("requestedAmount")),
        "client_type":            client.get("type"),
        "client_cupo_total":      _to_float(cupo.get("total")),
        "client_cupo_remaining":  _to_float(cupo.get("remainingBalance")),
        "client_delinquency_balance": _to_float(status_obj.get("delinquencyBalance")),
        "client_identity_verified": _bool_int(_get(client, "basicIdentity", "isVerified")),
        "client_num_previous_loans": len(prev_loans),
        "client_max_days_past_due": max_dpd,
        "credit_pd_score":        _to_float(credit_score.get("addiProbabilityDefault")),
        "channel":                d.get("channel"),
        "product":                d.get("product"),
        "journey_name":           journey.get("name"),
        "pre_approval":           _bool_int(d.get("preApproval")),
    }


def extract_device(raw):
    if not raw or raw.strip() in ("", "null"):
        return {}
    try:
        d = json.loads(raw)
    except Exception:
        return {}

    details    = d.get("details") or {}
    dev        = details.get("device") or {}
    browser    = dev.get("browser") or {}
    real_ip    = details.get("realIp") or {}
    real_loc   = real_ip.get("ipLocation") or {}
    stated_ip  = details.get("statedIp") or {}
    stated_loc = stated_ip.get("ipLocation") or {}
    rules      = details.get("ruleResults") or {}

    r_city, s_city = real_loc.get("city"), stated_loc.get("city")
    ip_mismatch = _bool_int(r_city != s_city) if r_city is not None and s_city is not None else None

    return {
        "device_result":            d.get("result"),
        "device_type":              dev.get("type"),
        "device_os_raw":            dev.get("os"),
        "device_is_new":            _bool_int(dev.get("isNew")),
        "device_screen_raw":        dev.get("screen"),
        "device_first_seen_str":    dev.get("firstSeen"),
        "device_browser_type":      browser.get("type"),
        "device_browser_language_raw": browser.get("language"),
        "device_browser_timezone":  _to_int(browser.get("timezone")),
        "device_cookies_enabled":   _bool_int(browser.get("cookiesEnabled")),
        "device_isp_raw":           real_ip.get("isp"),
        "device_ip_region_raw":     real_loc.get("region"),
        "device_ip_country_code":   real_loc.get("countryCode"),
        "device_ip_city_mismatch":  ip_mismatch,
        "device_rule_score":        _to_int(rules.get("score")),
        "device_rules_matched":     _to_int(rules.get("rulesMatched")),
    }


def extract_label(raw):
    if not raw or raw.strip() in ("", "null"):
        return "NO_FRAUD_DECISION"
    try:
        return json.loads(raw).get("reason") or "NO_FRAUD_DECISION"
    except Exception:
        return "NO_FRAUD_DECISION"


# ---------------------------------------------------------------------------
# Phase 2: compute statistics for imputation and bucketing
# ---------------------------------------------------------------------------

def compute_stats(rows):
    isp_counter     = Counter()
    region_counter  = Counter()
    numerics = {k: [] for k in [
        "credit_pd_score", "loan_requested_amount", "client_max_days_past_due",
        "client_cupo_total", "client_cupo_remaining", "client_delinquency_balance",
        "device_browser_timezone", "device_rule_score", "device_rules_matched",
        "device_age_days", "device_reg_to_app_days",
    ]}
    categoricals = {k: Counter() for k in [
        "client_type", "channel", "product", "journey_name",
        "device_result", "device_type", "device_browser_type", "device_ip_country_code",
    ]}

    for r in rows:
        if r.get("device_isp_raw"):
            isp_counter[r["device_isp_raw"]] += 1
        if r.get("device_ip_region_raw"):
            region_counter[r["device_ip_region_raw"]] += 1
        for k in numerics:
            v = r.get(k)
            if v is not None:
                numerics[k].append(v)
        for k in categoricals:
            v = r.get(k)
            if v:
                categoricals[k][v] += 1

    def safe_median(lst):
        return median(sorted(lst)) if lst else 0.0

    def safe_mode(ctr):
        return ctr.most_common(1)[0][0] if ctr else None

    return {
        "top_isps":    {k for k, _ in isp_counter.most_common(TOP_N)},
        "top_regions": {k for k, _ in region_counter.most_common(TOP_N)},
        **{f"median_{k}": safe_median(v) for k, v in numerics.items()},
        **{f"mode_{k}":   safe_mode(v)   for k, v in categoricals.items()},
    }


# ---------------------------------------------------------------------------
# Phase 3: transform a single extracted row
# ---------------------------------------------------------------------------

def transform(raw, stats):
    # --- Timestamp-derived features ---
    app_ts      = _parse_ts(raw.get("app_ts_str"))
    dev_reg_ts  = _parse_ts(raw.get("dev_reg_ts_str"))
    dev_first_ts = _parse_ts(raw.get("device_first_seen_str"))

    app_hour        = app_ts.hour      if app_ts else None
    app_day_of_week = app_ts.weekday() if app_ts else None

    device_age_days = int((app_ts - dev_first_ts).days) \
        if app_ts and dev_first_ts else None
    device_reg_to_app_days = int((app_ts - dev_reg_ts).days) \
        if app_ts and dev_reg_ts else None

    # --- Loan data flag + imputation ---
    has_loan_data       = 1 if raw.get("loan_term") is not None else 0
    loan_term           = raw.get("loan_term")           or 0
    loan_total_amount   = raw.get("loan_total_amount")   or 0.0
    loan_approved_amount = raw.get("loan_approved_amount") or 0.0
    loan_requested_amount = raw.get("loan_requested_amount")
    if loan_requested_amount is None:
        loan_requested_amount = 0.0 if not has_loan_data \
            else stats["median_loan_requested_amount"]

    # --- Cupo / delinquency flag + imputation ---
    has_cupo_data            = 1 if raw.get("client_cupo_total") is not None else 0
    client_cupo_total        = raw.get("client_cupo_total")        or 0.0
    client_cupo_remaining    = raw.get("client_cupo_remaining")    or 0.0
    client_delinquency_balance = raw.get("client_delinquency_balance") or 0.0
    client_max_days_past_due = raw.get("client_max_days_past_due")
    if client_max_days_past_due is None:
        client_max_days_past_due = 0 if not has_cupo_data \
            else _to_int(stats["median_client_max_days_past_due"])

    # --- credit_pd_score: median impute + missing flag ---
    credit_pd_score_missing = 1 if raw.get("credit_pd_score") is None else 0
    credit_pd_score = raw.get("credit_pd_score") \
        if not credit_pd_score_missing else stats["median_credit_pd_score"]

    # --- Remaining ~1% nulls ---
    def _fill_cat(key, stat_key):
        return raw.get(key) or stats[stat_key]

    def _fill_num(key, stat_key):
        v = raw.get(key)
        return v if v is not None else stats[stat_key]

    def _fill_bool(key, default=0):
        v = raw.get(key)
        return v if v is not None else default

    client_type            = _fill_cat("client_type",          "mode_client_type")
    client_identity_verified = _fill_bool("client_identity_verified")
    client_num_previous_loans = raw.get("client_num_previous_loans") or 0
    channel                = _fill_cat("channel",               "mode_channel")
    product                = _fill_cat("product",               "mode_product")
    journey_name           = _fill_cat("journey_name",          "mode_journey_name")
    pre_approval           = _fill_bool("pre_approval")
    device_result          = _fill_cat("device_result",         "mode_device_result")
    device_type            = _fill_cat("device_type",           "mode_device_type")
    device_is_new          = _fill_bool("device_is_new")
    device_browser_type    = _fill_cat("device_browser_type",   "mode_device_browser_type")
    device_browser_timezone = _fill_num("device_browser_timezone", "median_device_browser_timezone")
    device_cookies_enabled = _fill_bool("device_cookies_enabled", default=1)
    device_ip_country_code = _fill_cat("device_ip_country_code", "mode_device_ip_country_code")
    device_ip_city_mismatch = _fill_bool("device_ip_city_mismatch")
    device_rule_score      = _fill_num("device_rule_score",     "median_device_rule_score")
    device_rules_matched   = _fill_num("device_rules_matched",  "median_device_rules_matched")

    # Fill timestamp-derived features with medians if parse failed
    if app_hour is None:
        app_hour = _to_int(stats.get("median_app_hour", 12))
    if app_day_of_week is None:
        app_day_of_week = _to_int(stats.get("median_app_day_of_week", 2))
    if device_age_days is None:
        device_age_days = _to_int(stats["median_device_age_days"])
    if device_reg_to_app_days is None:
        device_reg_to_app_days = _to_int(stats["median_device_reg_to_app_days"])

    # --- OS family + version ---
    device_os_family, device_os_version = _parse_os(raw.get("device_os_raw"))
    if device_os_family is None:
        device_os_family = "OTHER"

    # --- Screen dimensions ---
    device_screen_width, device_screen_height = _parse_screen(raw.get("device_screen_raw"))

    # --- High-cardinality bucketing ---
    isp = raw.get("device_isp_raw")
    device_isp = isp if (isp and isp in stats["top_isps"]) else "OTHER"

    region = raw.get("device_ip_region_raw")
    device_ip_region = region if (region and region in stats["top_regions"]) else "OTHER"

    device_browser_language = _bucket_language(raw.get("device_browser_language_raw")) or "OTHER"

    return {
        "application_id":            raw["application_id"],
        "client_id_hashed":          raw["client_id_hashed"],
        "application_status":        raw["application_status"],
        "app_hour":                  app_hour,
        "app_day_of_week":           app_day_of_week,
        "has_loan_data":             has_loan_data,
        "loan_term":                 loan_term,
        "loan_total_amount":         loan_total_amount,
        "loan_approved_amount":      loan_approved_amount,
        "loan_requested_amount":     loan_requested_amount,
        "has_cupo_data":             has_cupo_data,
        "client_type":               client_type,
        "client_cupo_total":         client_cupo_total,
        "client_cupo_remaining":     client_cupo_remaining,
        "client_delinquency_balance": client_delinquency_balance,
        "client_identity_verified":  client_identity_verified,
        "client_num_previous_loans": client_num_previous_loans,
        "client_max_days_past_due":  client_max_days_past_due,
        "credit_pd_score_missing":   credit_pd_score_missing,
        "credit_pd_score":           credit_pd_score,
        "channel":                   channel,
        "product":                   product,
        "journey_name":              journey_name,
        "pre_approval":              pre_approval,
        "device_result":             device_result,
        "device_type":               device_type,
        "device_os_family":          device_os_family,
        "device_os_version":         device_os_version,
        "device_is_new":             device_is_new,
        "device_screen_width":       device_screen_width,
        "device_screen_height":      device_screen_height,
        "device_browser_type":       device_browser_type,
        "device_browser_language":   device_browser_language,
        "device_browser_timezone":   device_browser_timezone,
        "device_cookies_enabled":    device_cookies_enabled,
        "device_isp":                device_isp,
        "device_ip_region":          device_ip_region,
        "device_ip_country_code":    device_ip_country_code,
        "device_ip_city_mismatch":   device_ip_city_mismatch,
        "device_age_days":           device_age_days,
        "device_reg_to_app_days":    device_reg_to_app_days,
        "device_rule_score":         device_rule_score,
        "device_rules_matched":      device_rules_matched,
        "label":                     raw["label"],
    }


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

FIELDNAMES = [
    # Identifiers (exclude from feature_cols when training)
    "application_id",
    "client_id_hashed",
    # Application context
    "application_status",
    "app_hour",
    "app_day_of_week",
    # Loan features
    "has_loan_data",
    "loan_term",
    "loan_total_amount",
    "loan_approved_amount",
    "loan_requested_amount",
    # Client features
    "has_cupo_data",
    "client_type",
    "client_cupo_total",
    "client_cupo_remaining",
    "client_delinquency_balance",
    "client_identity_verified",
    "client_num_previous_loans",
    "client_max_days_past_due",
    # Credit features
    "credit_pd_score_missing",
    "credit_pd_score",
    # Application context (continued)
    "channel",
    "product",
    "journey_name",
    "pre_approval",
    # Device features
    "device_result",
    "device_type",
    "device_os_family",
    "device_os_version",
    "device_is_new",
    "device_screen_width",
    "device_screen_height",
    "device_browser_type",
    "device_browser_language",
    "device_browser_timezone",
    "device_cookies_enabled",
    "device_isp",
    "device_ip_region",
    "device_ip_country_code",
    "device_ip_city_mismatch",
    "device_age_days",
    "device_reg_to_app_days",
    "device_rule_score",
    "device_rules_matched",
    # Label
    "label",
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # --- Phase 1: extract from source ---
    print("Phase 1: Extracting from source CSV...")
    raw_rows = []
    with open(INPUT, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            app   = extract_application(row["application_data"])
            dev   = extract_device(row["device_information_data"])
            label = extract_label(row["fraud_decision_data"])
            raw_rows.append({
                "application_id":    row["application_id"],
                "client_id_hashed":  _hash_id(row["client_id"]),
                "application_status": row["application_status"],
                "app_ts_str":        row["application_created_at"],
                "dev_reg_ts_str":    row["device_information_created_at"],
                "label":             label,
                **app,
                **dev,
            })
            if (i + 1) % 10_000 == 0:
                print(f"  {i+1:,} rows extracted...", flush=True)

    print(f"  Total: {len(raw_rows):,} rows\n")

    # --- Phase 2: compute stats ---
    print("Phase 2: Computing statistics...")

    # Pre-compute timestamp-derived numerics so they're available for stats
    app_hours, app_dows, age_days, reg_days = [], [], [], []
    for r in raw_rows:
        app_ts     = _parse_ts(r.get("app_ts_str"))
        dev_reg_ts = _parse_ts(r.get("dev_reg_ts_str"))
        dev_fst_ts = _parse_ts(r.get("device_first_seen_str"))
        if app_ts:
            app_hours.append(app_ts.hour)
            app_dows.append(app_ts.weekday())
        if app_ts and dev_fst_ts:
            age_days.append((app_ts - dev_fst_ts).days)
        if app_ts and dev_reg_ts:
            reg_days.append((app_ts - dev_reg_ts).days)

    from statistics import median as _med
    extra_stats = {
        "median_app_hour":            _med(sorted(app_hours)) if app_hours else 12,
        "median_app_day_of_week":     _med(sorted(app_dows))  if app_dows  else 2,
        "median_device_age_days":     _med(sorted(age_days))  if age_days  else 0,
        "median_device_reg_to_app_days": _med(sorted(reg_days)) if reg_days else 0,
    }

    stats = {**compute_stats(raw_rows), **extra_stats}
    print(f"  median credit_pd_score:      {stats['median_credit_pd_score']:.4f}")
    print(f"  median device_age_days:      {stats['median_device_age_days']:.1f}")
    print(f"  median device_reg_to_app:    {stats['median_device_reg_to_app_days']:.1f}")
    print(f"  top ISPs kept:               {len(stats['top_isps'])}")
    print(f"  top regions kept:            {len(stats['top_regions'])}\n")

    # --- Phase 3: transform and write ---
    print("Phase 3: Applying transformations and writing output...")
    label_counts = {}
    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for raw in raw_rows:
            out = transform(raw, stats)
            writer.writerow(out)
            label_counts[out["label"]] = label_counts.get(out["label"], 0) + 1

    total = len(raw_rows)
    print(f"\nDone. {total:,} rows → {OUTPUT}")
    print(f"\nLabel distribution:")
    for lbl, cnt in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"  {lbl:<25} {cnt:>8,}  ({cnt/total*100:.2f}%)")
    print(f"\nFinal schema: {len(FIELDNAMES)} columns "
          f"({len(FIELDNAMES) - 3} features + application_id + client_id_hashed + label)")


if __name__ == "__main__":
    main()
