"""
Inference wrapper for the cluster-level ML classifier.

Loads a pickle produced by ``train_cluster_classifier.py`` and exposes a
simple ``predict / predict_with_proba`` interface.  Designed to be called
from inside the geometric pipeline (cargo.py), not as a CLI.

Pickle format (version 1)::

    {
        "version":       1,
        "model":         RandomForestClassifier | LGBMClassifier,
        "scaler":        StandardScaler,
        "feature_names": list[str],   # 23 strings, canonical order
        "label_map":     dict[str, int],  # e.g. {"cargo":1,"vehicle":2,"person":3}
    }

Usage::

    from pathlib import Path
    from cargo_geometric.cluster_classifier import ClusterClassifier

    cc = ClusterClassifier.load(Path("models/cluster_classifier_lgbm.pkl"))
    preds = cc.predict(X)                       # (N,) int
    preds, proba = cc.predict_with_proba(X)    # proba: (N, n_classes)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ClusterClassifier:
    model:         Any                  # RF or LGBM fitted classifier
    scaler:        Any                  # fitted StandardScaler
    feature_names: list[str]           # canonical feature order
    label_map:     dict[str, int]      # {"cargo":1, "vehicle":2, ...}
    inv_label_map: dict[int, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        if not self.inv_label_map:
            self.inv_label_map = {v: k for k, v in self.label_map.items()}

    @classmethod
    def load(cls, path: Path) -> "ClusterClassifier":
        """Load a cluster classifier pickle.

        Raises
        ------
        FileNotFoundError
            If the pickle does not exist — the caller decides the fallback.
        ValueError
            If the pickle has an unsupported version or missing keys.
        """
        import joblib

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"ClusterClassifier.load: pickle not found at {path}"
            )

        payload = joblib.load(path)

        version = payload.get("version")
        if version != 1:
            raise ValueError(
                f"ClusterClassifier.load: unsupported pickle version "
                f"{version!r} in {path}.  Expected version=1."
            )

        required_keys = {"model", "scaler", "feature_names", "label_map"}
        missing = required_keys - set(payload.keys())
        if missing:
            raise ValueError(
                f"ClusterClassifier.load: missing keys {missing} in {path}"
            )

        return cls(
            model=payload["model"],
            scaler=payload["scaler"],
            feature_names=list(payload["feature_names"]),
            label_map=dict(payload["label_map"]),
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels.

        Parameters
        ----------
        X : (N, n_features) float array — cluster feature matrix

        Returns
        -------
        (N,) int32 array of predicted class integers.
        """
        if len(X) == 0:
            return np.empty(0, dtype=np.int32)
        X_scaled = self.scaler.transform(X)
        return np.asarray(self.model.predict(X_scaled), dtype=np.int32)

    def predict_with_proba(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict cluster labels and class probabilities.

        Parameters
        ----------
        X : (N, n_features) float array

        Returns
        -------
        preds  : (N,) int32 — predicted class integers
        proba  : (N, n_classes) float64 — probability per class
        """
        if len(X) == 0:
            return np.empty(0, dtype=np.int32), np.empty((0, len(self.label_map)))
        X_scaled = self.scaler.transform(X)
        proba = np.asarray(self.model.predict_proba(X_scaled), dtype=np.float64)
        preds = np.asarray(self.model.predict(X_scaled), dtype=np.int32)
        return preds, proba
