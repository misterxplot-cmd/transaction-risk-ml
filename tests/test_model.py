import numpy as np
import pandas as pd
import pytest

from risk_model import FEATURES, feature_columns, make_model, validate_data


def dataset():
    count = 30
    return pd.DataFrame({
        "hour_of_day": np.arange(count) % 24, "amount": np.arange(count) + 1.0,
        "ip_prefix": 10.0, "login_frequency": 3, "session_duration": 20,
        "location_region": ["Europe"] * count, "purchase_pattern": ["focused"] * count,
        "age_group": ["adult"] * count, "risk_score": np.arange(count) * 3.0,
        "transaction_type": ["transfer"] * count,
        "anomaly": ["low_risk", "moderate_risk", "high_risk"] * 10,
    })


def test_behavior_model_never_uses_target_proxies():
    numeric, categorical = feature_columns("behavior_only")
    assert set(numeric + categorical) == set(FEATURES)
    assert not {"risk_score", "transaction_type", "anomaly"} & set(FEATURES)
    frame = dataset()
    model = make_model("behavior_only", trees=5).fit(frame, frame.anomaly)
    changed = frame.copy()
    changed["risk_score"] = -999
    changed["transaction_type"] = "unrelated"
    np.testing.assert_array_equal(model.predict(frame), model.predict(changed))


def test_unknown_category_and_missing_numeric_work():
    frame = dataset()
    model = make_model("behavior_only", trees=5).fit(frame, frame.anomaly)
    row = frame.iloc[[0]].copy()
    row["location_region"] = "unseen_region"
    row["amount"] = np.nan
    assert model.predict_proba(row).shape == (1, 3)


def test_validation_catches_missing_data_and_labels():
    frame = dataset()
    validate_data(frame)
    with pytest.raises(ValueError, match="Missing columns"):
        validate_data(frame.drop(columns="amount"))
    frame.loc[0, "anomaly"] = "unknown"
    with pytest.raises(ValueError, match="three labels"):
        validate_data(frame)
