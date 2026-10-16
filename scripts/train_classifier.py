"""Train the support-ticket classifier, print evaluation metrics and save the model.

Usage:  python scripts/train_classifier.py
Output: models/ticket_classifier.joblib  +  models/ticket_classifier.metrics.json
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.classifier.model import LABELS, train_and_evaluate  # noqa: E402
from app.config import settings  # noqa: E402


def main() -> None:
    """Call train_and_evaluate() and print its metrics in a readable form."""
    metrics = train_and_evaluate(settings.classifier_data_path, settings.classifier_model_path)

    print(f"\nTrain size: {metrics['train_size']}   Test size (newest tickets): {metrics['test_size']}")
    print(f"Macro F1 on the time-based test split: {metrics['macro_f1']}\n")

    print(f"{'class':<12}{'precision':>10}{'recall':>10}{'f1':>10}")
    for label in LABELS:
        m = metrics["per_class"][label]
        print(f"{label:<12}{m['precision']:>10.2f}{m['recall']:>10.2f}{m['f1-score']:>10.2f}")

    print("\nConfusion matrix (rows = true, columns = predicted):", LABELS)
    for label, row in zip(LABELS, metrics["confusion_matrix"]["matrix"]):
        print(f"  {label:<10} {row}")

    print("\nConfidence threshold -> share auto-classified / accuracy on that share:")
    for row in metrics["coverage_at_thresholds"]:
        print(f"  {row['threshold']:.1f} -> {row['auto_classified_share']:.0%} auto, accuracy {row['accuracy_on_auto']}")

    print(f"\nSaved model to {settings.classifier_model_path}")


if __name__ == "__main__":
    main()
