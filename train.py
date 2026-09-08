"""Compare fixed feature sets on one reproducible held-out split."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from risk_model import CLASSES, FEATURES, TARGET, make_model, validate_data


def run(data_path: Path, output: Path, trees: int = 100, seed: int = 42) -> dict:
    frame = pd.read_csv(data_path)
    validate_data(frame)
    train_index, test_index = train_test_split(
        frame.index, test_size=0.2, random_state=seed, stratify=frame[TARGET],
    )
    train, test = frame.loc[train_index], frame.loc[test_index]
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "rows": len(frame), "train_rows": len(train), "test_rows": len(test),
        "seed": seed, "trees": trees, "sklearn_version": sklearn.__version__,
        "classes": frame[TARGET].value_counts().to_dict(),
        "risk_score_ranges": frame.groupby(TARGET)["risk_score"].agg(["min", "max"]).to_dict("index"),
        "experiments": {},
        "limitations": [
            "Educational metaverse dataset, not verified production fraud labels.",
            "risk_score is a target proxy; transaction_type includes hindsight-like categories.",
            "One random split, no temporal or cross-domain validation and no probability calibration.",
            "The deployable variant was chosen in advance by feature availability, not test performance.",
        ],
    }
    estimators = {"dummy": DummyClassifier(strategy="most_frequent")}
    estimators.update({name: make_model(name, trees, seed) for name in (
        "with_score", "without_score", "behavior_only",
    )})
    for name, estimator in estimators.items():
        started = time.perf_counter()
        estimator.fit(train[FEATURES] if name == "dummy" else train, train[TARGET])
        fit_seconds = time.perf_counter() - started
        prediction = estimator.predict(test[FEATURES] if name == "dummy" else test)
        metrics = classification_report(test[TARGET], prediction, output_dict=True, zero_division=0)
        report["experiments"][name] = {
            "fit_seconds": fit_seconds, "classification_report": metrics,
            "confusion_matrix": confusion_matrix(test[TARGET], prediction, labels=CLASSES).tolist(),
            "confusion_matrix_labels": list(CLASSES),
        }
        print(f"{name}: accuracy={metrics['accuracy']:.4f}; macro F1={metrics['macro avg']['f1-score']:.4f}")
        if name == "behavior_only":
            metadata = {
                "schema_version": 1, "training_mode": "metaverse_educational",
                "features": FEATURES, "classes": list(estimator.classes_),
                "sklearn_version": sklearn.__version__,
                "dataset_sha256": report["dataset_sha256"],
                "created_at_utc": report["created_at_utc"],
            }
            joblib.dump({"model": estimator, "metadata": metadata}, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.trees < 1:
        parser.error("--trees must be positive")
    run(args.data, args.output, args.trees, args.seed)
