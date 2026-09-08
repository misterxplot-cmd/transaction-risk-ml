"""Training primitives for an educational transaction classification experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "anomaly"
CLASSES = ("high_risk", "low_risk", "moderate_risk")
CATEGORICAL = ["location_region", "purchase_pattern", "age_group"]
NUMERIC = ["hour_of_day", "amount", "ip_prefix", "login_frequency", "session_duration"]
FEATURES = NUMERIC + CATEGORICAL


def feature_columns(variant: str) -> tuple[list[str], list[str]]:
    """Change only inputs, not the estimator, between ablation experiments."""
    if variant not in {"with_score", "without_score", "behavior_only"}:
        raise ValueError(f"Unknown variant: {variant}")
    numeric, categorical = NUMERIC.copy(), CATEGORICAL.copy()
    if variant == "with_score":
        numeric.append("risk_score")
    if variant != "behavior_only":
        categorical.append("transaction_type")
    return numeric, categorical


def validate_data(frame: pd.DataFrame) -> None:
    required = set(FEATURES + [TARGET, "risk_score", "transaction_type"])
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != set(CLASSES):
        raise ValueError(f"Expected the three labels {CLASSES}")
    if frame[TARGET].value_counts().min() < 5:
        raise ValueError("At least five rows of each class are required")
    for name in NUMERIC + ["risk_score"]:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"{name} must be numeric")
        if np.isinf(frame[name].to_numpy(dtype=float)).any():
            raise ValueError(f"{name} contains infinity")


def make_model(variant: str, trees: int = 100, seed: int = 42) -> Pipeline:
    numeric, categorical = feature_columns(variant)
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipe, numeric),
        ("categorical", categorical_pipe, categorical),
    ])
    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=2, random_state=seed, n_jobs=2,
        )),
    ])
