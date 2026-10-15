# Script to train machine learning model.

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import (
    compute_model_metrics,
    compute_slice_metrics,
    inference,
    save_model,
    train_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "census.csv"
    model_dir = project_root / "model"
    slice_output_path = project_root / "slice_output.txt"

    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading census dataset from %s", data_path)
    data = pd.read_csv(data_path)

    logger.info("Splitting dataset into train and test sets (80/20)")
    train, test = train_test_split(data, test_size=0.20, random_state=42)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    logger.info("Processing training data...")
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True
    )

    logger.info("Processing test data...")
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=lb,
    )

    logger.info("Training machine learning model (RandomForestClassifier)...")
    model = train_model(X_train, y_train)

    logger.info("Evaluating model on test dataset...")
    preds = inference(model, X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    logger.info("Test Evaluation Metrics:")
    logger.info("  Precision: %.4f", precision)
    logger.info("  Recall:    %.4f", recall)
    logger.info("  F-beta:    %.4f", fbeta)

    logger.info("Saving trained model and preprocessing artifacts to %s",
                model_dir)
    save_model(model, model_dir / "model.joblib")
    save_model(encoder, model_dir / "encoder.joblib")
    save_model(lb, model_dir / "lb.joblib")

    logger.info("Computing slice metrics for categorical features...")
    slice_lines = []
    slice_lines.append("=" * 70)
    slice_lines.append("OVERALL TEST METRICS")
    slice_lines.append(f"Precision: {precision:.4f}")
    slice_lines.append(f"Recall:    {recall:.4f}")
    slice_lines.append(f"F1-Score:  {fbeta:.4f}")
    slice_lines.append("=" * 70)
    slice_lines.append("")

    for feature in cat_features:
        slice_lines.append(f"SLICE METRICS FOR FEATURE: {feature}")
        slice_lines.append("-" * 70)
        slice_lines.append(
            f"{'Value':<28} {'Count':<8} {'Precision':<12} "
            f"{'Recall':<12} {'F1-Score':<10}"
        )
        slice_lines.append("-" * 70)
        metrics = compute_slice_metrics(
            test, feature, model, encoder, lb, cat_features, label="salary"
        )
        for m in metrics:
            slice_lines.append(
                f"{m['value']:<28} {m['count']:<8} {m['precision']:<12.4f} "
                f"{m['recall']:<12.4f} {m['fbeta']:<10.4f}"
            )
        slice_lines.append("")

    with open(slice_output_path, "w") as f:
        f.write("\n".join(slice_lines) + "\n")

    logger.info("Slice metrics successfully saved to %s", slice_output_path)


if __name__ == "__main__":
    main()
