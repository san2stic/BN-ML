from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier


class SequenceMLPClassifier:
    """
    Sequence-aware classifier using flattened rolling windows.
    It keeps a sklearn-compatible predict/predict_proba API so it can join the ensemble.
    """

    def __init__(
        self,
        sequence_length: int = 32,
        hidden_layer_sizes: tuple[int, ...] = (128, 64),
        alpha: float = 1e-4,
        max_iter: int = 200,
        random_state: int = 42,
    ) -> None:
        self.sequence_length = max(2, int(sequence_length))
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)

        self.model: MLPClassifier | None = None
        self.classes_ = np.array([0, 1, 2], dtype=int)
        self._fallback_class = 1

    def _to_windows(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        if arr.ndim != 2:
            raise ValueError("SequenceMLPClassifier expects 2D array input")
        n, f = arr.shape
        if n < self.sequence_length:
            return np.empty((0, self.sequence_length * f), dtype=float)

        windows = [
            arr[idx - self.sequence_length + 1 : idx + 1].reshape(-1)
            for idx in range(self.sequence_length - 1, n)
        ]
        out = np.asarray(windows, dtype=float)
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "SequenceMLPClassifier":
        arr_y = np.asarray(y, dtype=int)
        windows = self._to_windows(x)
        if windows.shape[0] == 0:
            raise ValueError("Not enough rows to build sequence windows")
        y_seq = arr_y[self.sequence_length - 1 :]
        if y_seq.size == 0:
            raise ValueError("No sequence targets available")

        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self.model.fit(windows, y_seq)
        self.classes_ = np.asarray(getattr(self.model, "classes_", np.array([0, 1, 2], dtype=int)), dtype=int)
        if y_seq.size:
            counts = np.bincount(np.clip(y_seq, 0, 2), minlength=3)
            self._fallback_class = int(np.argmax(counts))
        return self

    def _prefix_pred(self, n: int) -> np.ndarray:
        return np.full(n, int(self._fallback_class), dtype=int)

    def predict(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        n = arr.shape[0]
        if self.model is None:
            return self._prefix_pred(n)

        windows = self._to_windows(arr)
        if windows.shape[0] == 0:
            return self._prefix_pred(n)

        tail = np.asarray(self.model.predict(windows), dtype=int)
        prefix_len = max(0, n - len(tail))
        if prefix_len == 0:
            return tail
        return np.concatenate([self._prefix_pred(prefix_len), tail])

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        n = arr.shape[0]
        classes = np.asarray(self.classes_, dtype=int)
        k = max(len(classes), 1)

        if self.model is None:
            out = np.zeros((n, k), dtype=float)
            fallback_idx = int(np.where(classes == self._fallback_class)[0][0]) if self._fallback_class in classes else 0
            out[:, fallback_idx] = 1.0
            return out

        windows = self._to_windows(arr)
        if windows.shape[0] == 0:
            out = np.zeros((n, k), dtype=float)
            fallback_idx = int(np.where(classes == self._fallback_class)[0][0]) if self._fallback_class in classes else 0
            out[:, fallback_idx] = 1.0
            return out

        tail = np.asarray(self.model.predict_proba(windows), dtype=float)
        prefix_len = max(0, n - tail.shape[0])
        if prefix_len == 0:
            return tail

        prefix = np.zeros((prefix_len, tail.shape[1]), dtype=float)
        fallback_idx = int(np.where(classes == self._fallback_class)[0][0]) if self._fallback_class in classes else 0
        prefix[:, fallback_idx] = 1.0
        return np.vstack([prefix, tail])

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> "SequenceMLPClassifier":
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
