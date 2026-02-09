from __future__ import annotations

import numpy as np

from ml_engine.sequence_model import SequenceMLPClassifier


def test_sequence_model_fit_predict_shapes() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(220, 6))
    y = np.where(x[:, 0] > 0.3, 2, np.where(x[:, 0] < -0.3, 0, 1)).astype(int)

    model = SequenceMLPClassifier(sequence_length=16, hidden_layer_sizes=(32, 16), max_iter=40, random_state=42)
    model.fit(x, y)
    preds = model.predict(x)
    proba = model.predict_proba(x)

    assert preds.shape == (220,)
    assert proba.shape[0] == 220
    assert np.all((preds >= 0) & (preds <= 2))


def test_sequence_model_predict_with_short_input() -> None:
    model = SequenceMLPClassifier(sequence_length=10, hidden_layer_sizes=(16,), max_iter=20)
    x_train = np.random.default_rng(1).normal(size=(80, 4))
    y_train = np.random.default_rng(2).integers(0, 3, size=80)
    model.fit(x_train, y_train)

    x_short = np.random.default_rng(3).normal(size=(3, 4))
    preds = model.predict(x_short)
    assert preds.shape == (3,)
