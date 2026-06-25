import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
    average_precision_score,
)


# Algorithm catalogue: each entry defines a base estimator and its hyperparameter grid.
ALGORITHM_CATALOGUE = {
    "dummy": {
        "estimator": lambda rs: DummyClassifier(strategy="prior", random_state=rs),
        "param_grid": {},
    },
    "logistic_regression": {
        "estimator": lambda rs: LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=rs
        ),
        "param_grid": {
            "C": [0.01, 0.1, 1, 10],
            "l1_ratio": [0],
        },
    },
    "random_forest": {
        "estimator": lambda rs: RandomForestClassifier(
            class_weight="balanced", random_state=rs
        ),
        "param_grid": {
            "n_estimators": [200, 400],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 4],
            "max_features": ["sqrt", 0.5],
            "bootstrap": [True],
        },
    },
    "gradient_boosting": {
        "estimator": lambda rs: GradientBoostingClassifier(random_state=rs),
        "param_grid": {
            "n_estimators": [200, 400],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
            "min_samples_leaf": [1, 4],
        },
    },
}


def _encode_features(X_train, X_test, categorical_cols):
    """Fit OneHotEncoder on training data only, then transform both splits.

    Returns the fitted encoder so the caller can reuse it for additional splits
    (e.g. a held-out test set) without refitting on a different data slice.
    """
    if not categorical_cols:
        encoder = None
        return X_train, X_test, X_train.columns.tolist(), encoder

    encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
    encoder.fit(X_train[categorical_cols])

    def _apply(X):
        ohe = pd.DataFrame(
            encoder.transform(X[categorical_cols]),
            columns=encoder.get_feature_names_out(categorical_cols),
            index=X.index,
        )
        return pd.concat([X.drop(columns=categorical_cols), ohe], axis=1)

    X_train = _apply(X_train)
    X_test  = _apply(X_test)
    return X_train, X_test, X_train.columns.tolist(), encoder


def _build_threshold_table(y_true, y_scores, p_series=None):
    """Build a threshold trade-off table from scores."""
    thresholds = np.arange(0.05, 1.00, 0.05)
    rows = []
    for t in thresholds:
        y_pred_t = (y_scores >= t).astype(int)
        row = {
            "threshold": round(float(t), 2),
            "precision_positive": precision_score(
                y_true, y_pred_t, pos_label=1, zero_division=0
            ),
            "recall_positive": recall_score(
                y_true, y_pred_t, pos_label=1, zero_division=0
            ),
            "f1_positive": f1_score(y_true, y_pred_t, pos_label=1, zero_division=0),
            "flag_rate": round(float(np.mean(y_pred_t)), 4),
        }
        if p_series is not None:
            mask_stealth = p_series == "bot_stealth"
            if mask_stealth.sum() > 0:
                row["recall_stealth"] = recall_score(
                    y_true[mask_stealth],
                    y_pred_t[mask_stealth],
                    pos_label=1,
                    zero_division=0,
                )
            else:
                row["recall_stealth"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _select_best_threshold(df_thresholds, precision_constraint):
    """Select T_op: highest-recall threshold where precision >= precision_constraint.

    This is the recommended operational threshold that balances precision and recall.
    """
    valid = df_thresholds["precision_positive"] >= precision_constraint
    if valid.any():
        best_row = (
            df_thresholds.loc[valid]
            .sort_values(["recall_positive", "threshold"], ascending=[False, False])
            .iloc[0]
        )
        return float(best_row["threshold"]), best_row
    return 0.5, None


def _select_deny_threshold(df_thresholds, deny_precision_constraint=0.70):
    """Select T_high: lowest threshold where precision >= deny_precision_constraint.

    Transactions at or above T_high are automatically denied. A higher precision
    requirement (e.g. 70%) ensures that at least 70% of auto-blocked transactions
    are truly fraudulent, even at the cost of lower recall.
    """
    valid = df_thresholds["precision_positive"] >= deny_precision_constraint
    if valid.any():
        best_row = (
            df_thresholds.loc[valid]
            .sort_values(["threshold"], ascending=[True])
            .iloc[0]
        )
        return float(best_row["threshold"]), best_row
    return None, None


def _select_review_threshold(df_thresholds, recall_constraint=0.95):
    """Select T_low: the highest threshold where recall >= recall_constraint.

    Transactions scoring below T_low are auto-allowed (ALLOW).
    Those between T_low and T_high go to manual REVIEW.
    """
    valid = df_thresholds["recall_positive"] >= recall_constraint
    if valid.any():
        best_row = (
            df_thresholds.loc[valid]
            .sort_values(["threshold"], ascending=[False])
            .iloc[0]
        )
        return float(best_row["threshold"]), best_row
    # Fallback: use lowest available threshold
    row = df_thresholds.iloc[0]
    return float(row["threshold"]), row


def assign_decisions(y_scores, t_low, t_high):
    """Map probability scores to three-tier decisions.

    Parameters
    ----------
    y_scores : array-like
        Fraud probability scores from model.predict_proba.
    t_low : float
        Review threshold — scores below this are ALLOW.
    t_high : float
        Deny threshold — scores at or above this are DENY.

    Returns
    -------
    decisions : np.ndarray of str
        One of 'ALLOW', 'REVIEW', 'DENY' per transaction.
    """
    decisions = np.where(
        y_scores >= t_high,
        "DENY",
        np.where(y_scores >= t_low, "REVIEW", "ALLOW"),
    )
    return decisions


def run_training_pipeline(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    model_name: str,
    algorithm: str = "random_forest",
    profile_col: str = "profile",
    threshold_precision_constraint: float = 0.50,
    deny_precision_constraint: float = 0.70,
    review_recall_constraint: float = 0.95,
    test_size: float = 0.2,
    val_size: float = 0.125,
    random_state: int = 42,
    run_grid_search: bool = True,
    fixed_params: dict = None,
):
    """
    Reusable pipeline for behavioral, transaction-only, and combined fraud models.

    Splits data into train / validation / test:
      - Thresholds are tuned on the validation set (avoids test-set snooping).
      - Final metrics are reported on the held-out test set.

    Three-threshold decision system:
      - T_high (DENY):   lowest threshold where precision >= deny_precision_constraint (0.70).
      - T_op   (operational): highest-recall threshold where precision >= threshold_precision_constraint (0.50).
      - T_low  (REVIEW): highest threshold where recall >= review_recall_constraint (0.95).
      - Score >= T_high → DENY, T_low <= Score < T_high → REVIEW, Score < T_low → ALLOW.
      - T_op is reported as the recommended operational threshold.

    Parameters
    ----------
    algorithm : str
        Key into ALGORITHM_CATALOGUE. One of 'dummy', 'logistic_regression',
        'random_forest', 'gradient_boosting'.
    threshold_precision_constraint : float
        Minimum precision for the operational threshold (T_op). Default 0.50.
    deny_precision_constraint : float
        Minimum precision for the DENY threshold (T_high). Default 0.70.
    review_recall_constraint : float
        Minimum recall for the REVIEW threshold (T_low). Default 0.95.
    val_size : float
        Fraction of the *training* set to hold out for threshold tuning.
        Default 0.125 means 10% of total data (0.125 * 0.8 = 0.10).
    """
    if algorithm not in ALGORITHM_CATALOGUE:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Choose from {list(ALGORITHM_CATALOGUE.keys())}"
        )

    algo_spec = ALGORITHM_CATALOGUE[algorithm]

    # Copy to avoid mutating original
    df_model = df.copy()

    # Keep profiles only for analysis, not for training
    profiles = df_model[profile_col] if profile_col in df_model.columns else None

    # Select features and target
    X = df_model[feature_cols].copy()
    y = df_model[target_col].copy()

    # Identify categorical columns before splitting
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # Split into train+val and test
    if profiles is not None:
        X_trainval, X_test, y_trainval, y_test, p_trainval, p_test = train_test_split(
            X, y, profiles, test_size=test_size, random_state=random_state, stratify=y
        )
        # Further split train+val into train and val
        X_train, X_val, y_train, y_val, p_train, p_val = train_test_split(
            X_trainval,
            y_trainval,
            p_trainval,
            test_size=val_size,
            random_state=random_state,
            stratify=y_trainval,
        )
    else:
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval,
            y_trainval,
            test_size=val_size,
            random_state=random_state,
            stratify=y_trainval,
        )
        p_train, p_val, p_test = None, None, None

    # One-hot encode: fit on train only, reuse the same encoder for val and test
    X_train, X_val, feature_names, ohe_encoder = _encode_features(
        X_train, X_val, categorical_cols
    )
    if categorical_cols and ohe_encoder is not None:
        ohe_test = pd.DataFrame(
            ohe_encoder.transform(X_test[categorical_cols]),
            columns=ohe_encoder.get_feature_names_out(categorical_cols),
            index=X_test.index,
        )
        X_test = pd.concat([X_test.drop(columns=categorical_cols), ohe_test], axis=1)
    feature_names = X_train.columns.tolist()

    # Build model
    base_model = algo_spec["estimator"](random_state)
    param_grid = algo_spec["param_grid"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    pr_auc_scorer = make_scorer(
        average_precision_score, response_method="predict_proba"
    )

    if fixed_params is not None:
        # Use caller-supplied hyperparameters (e.g. best params from a prior grid search)
        base_model.set_params(**fixed_params)
        model = base_model
        model.fit(X_train, y_train)
        print(f"\n=== {model_name} | Using fixed hyperparameters (no grid search) ===")
    elif run_grid_search and param_grid:
        print(f"\n=== Starting Grid Search: {model_name} ({algorithm}) ===")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=pr_auc_scorer,
            n_jobs=-1,
            verbose=2,
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"\n=== Best Hyperparameters for {model_name} ===")
        print(grid_search.best_params_)
    else:
        model = base_model
        model.fit(X_train, y_train)

    n_classes = len(np.unique(y_train))
    classes   = sorted(np.unique(y_train))

    # --- Final evaluation on TEST set ---
    y_pred   = model.predict(X_test)
    y_proba  = model.predict_proba(X_test)

    print(f"\n=== {model_name} | Classification Report ===")
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"\n=== {model_name} | Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    # ------------------------------------------------------------------ #
    #  Binary path: threshold tuning + binary metrics                     #
    # ------------------------------------------------------------------ #
    if n_classes == 2:
        y_val_scores = model.predict_proba(X_val)[:, 1]
        val_threshold_table = _build_threshold_table(y_val, y_val_scores, p_val)

        best_threshold, _ = _select_best_threshold(
            val_threshold_table, threshold_precision_constraint
        )
        deny_threshold, _ = _select_deny_threshold(
            val_threshold_table, deny_precision_constraint
        )
        if deny_threshold is None:
            deny_threshold = best_threshold
        review_threshold, _ = _select_review_threshold(
            val_threshold_table, review_recall_constraint
        )
        if review_threshold > best_threshold:
            review_threshold = best_threshold
        if best_threshold > deny_threshold:
            best_threshold = deny_threshold

        print(
            f"\n=== {model_name} | Thresholds (validation set): "
            f"T_low={review_threshold}, T_op={best_threshold}, T_high={deny_threshold} ==="
        )

        y_scores = y_proba[:, 1]
        roc_auc  = roc_auc_score(y_test, y_scores)
        pr_auc   = average_precision_score(y_test, y_scores)
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_test, y_scores
        )

        test_threshold_table = _build_threshold_table(y_test, y_scores, p_test)
        best_row_mask = test_threshold_table["threshold"] == round(best_threshold, 2)
        best_row = test_threshold_table.loc[best_row_mask].iloc[0] \
            if best_row_mask.any() else None

        if best_row is not None:
            print(f"\n=== {model_name} | Test metrics at threshold {best_threshold} ===")
            print(best_row.to_string())

        decisions = assign_decisions(y_scores, review_threshold, deny_threshold)
        decision_counts = pd.Series(decisions).value_counts()
        n_test = len(y_scores)
        print(
            f"\n=== {model_name} | Three-Tier Decisions "
            f"(T_low={review_threshold}, T_op={best_threshold}, T_high={deny_threshold}) ==="
        )
        for d in ["ALLOW", "REVIEW", "DENY"]:
            cnt = decision_counts.get(d, 0)
            print(f"  {d}: {cnt} ({cnt / n_test:.1%})")

    # ------------------------------------------------------------------ #
    #  Multi-class path: macro OvR metrics + threshold tuning on val set  #
    #  Class encoding: 0=NO_FRAUD_DECISION, 1=FRAUD_SUSPECT, 2=CONFIRM   #
    #  T_high tuned on p(class 2) vs (y==2)                              #
    #  T_low  tuned on 1-p(class 0) vs (y>0)                             #
    #  Decision: p2>=T_high → DENY; (1-p0)>=T_low → REVIEW; else ALLOW  #
    # ------------------------------------------------------------------ #
    else:
        test_threshold_table = best_row = None
        y_scores = y_proba
        pr_thresholds = precision_curve = recall_curve = None

        # Validation-set threshold tuning
        y_val_proba = model.predict_proba(X_val)

        # T_high: auto-deny when sufficiently confident it is CONFIRM_FRAUD
        y_val_deny_scores = y_val_proba[:, 2]
        y_val_deny_labels = (y_val == 2).astype(int)
        val_deny_table = _build_threshold_table(y_val_deny_labels, y_val_deny_scores)
        deny_threshold, _ = _select_deny_threshold(val_deny_table, deny_precision_constraint)
        if deny_threshold is None:
            deny_threshold = 0.5

        # T_low: route to REVIEW when any-fraud signal is strong enough
        y_val_review_scores = 1.0 - y_val_proba[:, 0]
        y_val_review_labels = (y_val > 0).astype(int)
        val_review_table = _build_threshold_table(y_val_review_labels, y_val_review_scores)
        review_threshold, _ = _select_review_threshold(val_review_table, review_recall_constraint)

        # T_low must not exceed T_high
        if review_threshold > deny_threshold:
            review_threshold = deny_threshold

        best_threshold = deny_threshold  # T_op not separately tuned on multi-class path

        print(
            f"\n=== {model_name} | Thresholds (validation set): "
            f"T_low={review_threshold}, T_high={deny_threshold} ==="
        )

        # Multi-class OvR metrics on test set
        roc_auc = roc_auc_score(
            y_test, y_proba, multi_class="ovr", average="macro"
        )
        y_bin = label_binarize(y_test, classes=classes)
        pr_auc = float(np.mean([
            average_precision_score(y_bin[:, i], y_proba[:, i])
            for i in range(n_classes)
        ]))

        # Threshold-based decisions on test set
        p_deny   = y_proba[:, 2]
        p_review = 1.0 - y_proba[:, 0]
        decisions = np.where(
            p_deny >= deny_threshold,
            "DENY",
            np.where(p_review >= review_threshold, "REVIEW", "ALLOW"),
        )
        decision_counts = pd.Series(decisions).value_counts()
        n_test = len(decisions)
        print(
            f"\n=== {model_name} | Three-Tier Decisions "
            f"(T_low={review_threshold}, T_high={deny_threshold}) ==="
        )
        for d in ["ALLOW", "REVIEW", "DENY"]:
            cnt = decision_counts.get(d, 0)
            print(f"  {d}: {cnt} ({cnt / n_test:.1%})")

    print(f"\n=== {model_name} | ROC-AUC: {roc_auc:.4f} ===")
    print(f"\n=== {model_name} | PR-AUC:  {pr_auc:.4f} ===")

    # Feature importance (tree-based or linear models)
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        print(f"\n=== {model_name} | Feature Importance ===")
        print(feature_importance.to_string(index=False))
    elif hasattr(model, "coef_"):
        # coef_ is (1, n_features) for binary, (n_classes, n_features) for multi-class
        coef_importance = np.abs(model.coef_).mean(axis=0)
        feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": coef_importance}
        ).sort_values("importance", ascending=False)
        print(f"\n=== {model_name} | Feature Coefficients (mean abs) ===")
        print(feature_importance.to_string(index=False))
    else:
        feature_importance = None

    # Score distribution by profile if available
    score_distribution = None
    if p_test is not None and n_classes == 2:
        score_distribution = pd.DataFrame(
            {"profile": p_test.values, "y_true": y_test.values, "y_score": y_scores}
        )
        print(f"\n=== {model_name} | Score Distribution by Profile ===")
        print(
            score_distribution.groupby("profile")["y_score"].describe()[
                ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
            ]
        )

    return {
        "model_name": model_name,
        "algorithm": algorithm,
        "model": model,
        "target_col": target_col,
        "X_columns": feature_names,
        "n_classes": n_classes,
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold_table": test_threshold_table,
        "best_threshold": best_threshold,
        "best_threshold_row": best_row,
        "deny_threshold": deny_threshold,
        "review_threshold": review_threshold,
        "decisions": decisions,
        "decision_counts": decision_counts.to_dict(),
        "feature_importance": feature_importance,
        "score_distribution": score_distribution,
        "precision_curve": precision_curve,
        "recall_curve": recall_curve,
        "pr_thresholds": pr_thresholds,
        "p_train": p_train,
        "y_scores": y_scores,
        "y_test": y_test,
    }
