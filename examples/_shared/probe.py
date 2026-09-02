"""Linear probe: L2 logistic regression on standardized pooled activations, with a
layer sweep. Mirrors Wehner & Fritz section 3.1.2.

Extraction (slow, needs the model) is separate from fitting (fast, pure numpy) so
the sweep can be iterated cheaply.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Probe:
    """A fitted linear probe. Operates on ``(N, D)`` activation arrays for its layer."""

    layer: int
    pooling: str
    mean_: np.ndarray  # standardizer, (D,)
    scale_: np.ndarray  # standardizer, (D,)
    coef_: np.ndarray  # (D,)
    intercept_: float
    val_auc: float
    layer_aucs: dict[int, float] | None = None  # full sweep, for diagnostics

    def logit(self, acts: np.ndarray) -> np.ndarray:
        x = (np.asarray(acts, dtype=np.float64) - self.mean_) / self.scale_
        return x @ self.coef_ + self.intercept_

    def score(self, acts: np.ndarray) -> np.ndarray:
        """P(positive class), i.e. P(toxic) / P(harmful). ``(N,)`` in [0, 1]."""
        return 1.0 / (1.0 + np.exp(-self.logit(acts)))

    def direction(self, *, standardized: bool = True) -> np.ndarray:
        """Unit vector along the decision boundary normal. ``standardized=False``
        maps it back to raw activation space."""
        d = self.coef_ if standardized else self.coef_ / self.scale_
        return d / (np.linalg.norm(d) + 1e-12)

    def save(self, path: str | Path) -> None:
        save_probe(self, path)


def fit_probe(
    acts_by_layer: Mapping[int, np.ndarray],
    labels: np.ndarray,
    *,
    pooling: str = "mean",
    val_frac: float = 0.2,
    seed: int = 0,
    C: float = 1.0,
) -> Probe:
    """Fit one probe per layer, keep the best val-AUC layer, then refit it on all data."""
    y = np.asarray(labels).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("labels must contain both classes")

    def _fit(x: np.ndarray, y: np.ndarray) -> tuple[StandardScaler, LogisticRegression]:
        scaler = StandardScaler().fit(x)
        clf = LogisticRegression(C=C, max_iter=1000)  # default regularization is L2
        clf.fit(scaler.transform(x), y)
        return scaler, clf

    layer_aucs: dict[int, float] = {}
    for layer, x in sorted(acts_by_layer.items()):
        x = np.asarray(x, dtype=np.float64)
        x_tr, x_va, y_tr, y_va = train_test_split(
            x, y, test_size=val_frac, random_state=seed, stratify=y
        )
        scaler, clf = _fit(x_tr, y_tr)
        layer_aucs[layer] = float(
            roc_auc_score(y_va, clf.decision_function(scaler.transform(x_va)))
        )

    best_layer = max(layer_aucs, key=layer_aucs.__getitem__)
    scaler, clf = _fit(np.asarray(acts_by_layer[best_layer], dtype=np.float64), y)
    return Probe(
        layer=best_layer,
        pooling=pooling,
        mean_=scaler.mean_.copy(),
        scale_=scaler.scale_.copy(),
        coef_=clf.coef_[0].copy(),
        intercept_=float(clf.intercept_[0]),
        val_auc=layer_aucs[best_layer],
        layer_aucs=layer_aucs,
    )


def save_probe(probe: Probe, path: str | Path) -> None:
    """Serialize as a single ``.npz`` (arrays) with metadata in the same file."""
    path = Path(path)
    # write to an explicit file handle: np.savez(path, ...) appends '.npz' even when
    # the path already ends in it, silently creating 'name.npz.npz'.
    meta = json.dumps(
        {
            "layer": probe.layer,
            "pooling": probe.pooling,
            "intercept_": probe.intercept_,
            "val_auc": probe.val_auc,
            "layer_aucs": probe.layer_aucs,
        }
    ).encode()
    with open(path, "wb") as fh:
        np.savez(
            fh,
            mean_=probe.mean_,
            scale_=probe.scale_,
            coef_=probe.coef_,
            _meta=np.frombuffer(meta, dtype=np.uint8),
        )


def load_probe(path: str | Path) -> Probe:
    path = Path(path)
    data = np.load(path)
    meta = json.loads(bytes(data["_meta"]).decode())
    layer_aucs = meta.get("layer_aucs")
    if layer_aucs is not None:
        layer_aucs = {int(k): float(v) for k, v in layer_aucs.items()}
    return Probe(
        layer=int(meta["layer"]),
        pooling=str(meta["pooling"]),
        mean_=data["mean_"],
        scale_=data["scale_"],
        coef_=data["coef_"],
        intercept_=float(meta["intercept_"]),
        val_auc=float(meta["val_auc"]),
        layer_aucs=layer_aucs,
    )


def probe_summary(probe: Probe) -> str:
    return json.dumps(
        {k: v for k, v in asdict(probe).items() if not isinstance(v, np.ndarray)}, indent=2
    )


def save_probe_verified(probe: Probe, path: str | Path) -> Path:
    """Save, then reload and assert it round-tripped. Returns the resolved path."""
    out = Path(path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    save_probe(probe, out)
    check = load_probe(out)
    if check.layer != probe.layer or abs(check.val_auc - probe.val_auc) > 1e-6:
        raise RuntimeError(
            f"save/load mismatch: wrote L{probe.layer} auc {probe.val_auc:.4f}, "
            f"reloaded L{check.layer} auc {check.val_auc:.4f} from {out}"
        )
    print(f"saved + verified -> {out}  (L{check.layer}, val_auc {check.val_auc:.4f})")
    return out
