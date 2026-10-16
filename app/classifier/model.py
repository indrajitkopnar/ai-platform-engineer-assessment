"""Train, evaluate and persist the fast classifier (TF-IDF + logistic regression).

Why this model first:
- trains in seconds, runs on CPU in about a millisecond, cheap to retrain nightly;
- probabilities give a confidence score that drives the cascade (fast model → LLM → human);
- interpretable (top-weighted words per class) and a strong baseline for short support texts.
A fine-tuned multilingual transformer is the upgrade path once there are thousands of labels.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from app.classifier.preprocessing import clean_message

LABELS = ["billing", "technical", "general"]


@dataclass
class Ticket:
    received_at: str
    subject: str
    body: str
    category: str

    @property
    def text(self) -> str:
        return clean_message(self.subject, self.body)


def load_tickets(path: Path) -> list[Ticket]:
    with path.open(encoding="utf-8", newline="") as f:
        return [Ticket(**row) for row in csv.DictReader(f)]


def time_split(tickets: list[Ticket], test_fraction: float = 0.25) -> tuple[list[Ticket], list[Ticket]]:
    """Train on older tickets, test on newer ones — mirrors production and exposes drift.
    (A random split would leak near-duplicate tickets from the same week into both sides.)"""
    ordered = sorted(tickets, key=lambda t: t.received_at)
    cut = int(len(ordered) * (1 - test_fraction))
    return ordered[:cut], ordered[cut:]


def build_pipeline() -> Pipeline:
    return Pipeline([
        # word 1–2 grams catch phrases like "charged twice"; sublinear tf damps repeated words
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        # class_weight="balanced" protects small classes; C=5 = mild regularisation for a small dataset
        ("clf", LogisticRegression(C=5.0, max_iter=2000, class_weight="balanced")),
    ])


def coverage_at_thresholds(model: Pipeline, texts: list[str], labels: list[str]) -> list[dict]:
    """For each confidence threshold: what share is auto-classified, and how accurate is that share.
    This table is how the business picks the operating point (automation vs. accuracy)."""
    probs = model.predict_proba(texts)
    classes = list(model.classes_)
    rows = []
    for threshold in (0.4, 0.5, 0.6, 0.7, 0.8):
        auto = [(classes[p.argmax()], y) for p, y in zip(probs, labels) if p.max() >= threshold]
        accuracy = sum(pred == y for pred, y in auto) / len(auto) if auto else None
        rows.append({"threshold": threshold, "auto_classified_share": round(len(auto) / len(labels), 2),
                     "accuracy_on_auto": None if accuracy is None else round(accuracy, 2)})
    return rows


def train_and_evaluate(data_path: Path, model_path: Path) -> dict:
    """Train on the older split, report metrics on the newer split, then refit on all data and save."""
    tickets = load_tickets(data_path)
    train, test = time_split(tickets)

    model = build_pipeline().fit([t.text for t in train], [t.category for t in train])
    test_texts, test_labels = [t.text for t in test], [t.category for t in test]
    predictions = model.predict(test_texts)

    metrics = {
        "train_size": len(train),
        "test_size": len(test),
        "macro_f1": round(f1_score(test_labels, predictions, average="macro"), 3),
        "per_class": classification_report(test_labels, predictions, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": {"labels": LABELS, "matrix": confusion_matrix(test_labels, predictions, labels=LABELS).tolist()},
        "coverage_at_thresholds": coverage_at_thresholds(model, test_texts, test_labels),
    }

    final_model = build_pipeline().fit([t.text for t in tickets], [t.category for t in tickets])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path)
    model_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def load_model(model_path: Path) -> Pipeline:
    return joblib.load(model_path)
