"""Dummy baseline classifier builder for URL phishing detection.

Provides trivial majority class baseline to benchmark minimum expected accuracy.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier


def build_dummy(strategy: str = "most_frequent", random_state: int = 42) -> DummyClassifier:
    """Instantiate a DummyClassifier baseline.

    Args:
        strategy: Baseline strategy ('most_frequent').
        random_state: Deterministic random seed.

    Returns:
        DummyClassifier instance.
    """
    return DummyClassifier(strategy=strategy, random_state=random_state)

