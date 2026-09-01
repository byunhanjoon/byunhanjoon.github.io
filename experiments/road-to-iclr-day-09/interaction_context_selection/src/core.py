from __future__ import annotations

import hashlib
import json
import math
import platform
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
PROCESSED = ROOT / "results" / "processed"
PLOTS = ROOT / "plots"


DATASETS: dict[str, dict[str, Any]] = {
    "adult": {"task": "classification", "openml_id": 1590},
    "bank-marketing": {"task": "classification", "openml_id": 1461},
    "credit-g": {"task": "classification", "openml_id": 31},
    "electricity": {"task": "classification", "openml_id": 151},
    "california_housing": {"task": "regression", "source": "sklearn"},
    "diamonds": {"task": "regression", "openml_id": 42225},
    "churn": {"task": "classification", "openml_id": 40701},
    "house_16H": {"task": "regression", "openml_id": 574},
}


def ensure_dirs() -> None:
    for path in (ROOT / "experiments", RAW, PROCESSED, PLOTS):
        path.mkdir(parents=True, exist_ok=True)


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("::".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little") & 0x7FFFFFFF


@dataclass
class DatasetBundle:
    name: str
    task: str
    source_id: str
    X: pd.DataFrame
    y: np.ndarray
    classes: np.ndarray | None
    candidate_idx: np.ndarray
    selector_idx: np.ndarray
    test_idx: np.ndarray
    z: np.ndarray
    feature_z: np.ndarray
    selector_feature_z: np.ndarray
    test_feature_z: np.ndarray
    target_bins: np.ndarray

    @property
    def X_candidate(self) -> pd.DataFrame:
        return self.X.iloc[self.candidate_idx].reset_index(drop=True)

    @property
    def y_candidate(self) -> np.ndarray:
        return self.y[self.candidate_idx]

    @property
    def X_selector(self) -> pd.DataFrame:
        return self.X.iloc[self.selector_idx].reset_index(drop=True)

    @property
    def y_selector(self) -> np.ndarray:
        return self.y[self.selector_idx]

    @property
    def X_test(self) -> pd.DataFrame:
        return self.X.iloc[self.test_idx].reset_index(drop=True)

    @property
    def y_test(self) -> np.ndarray:
        return self.y[self.test_idx]


def _regression_bins(y: np.ndarray, bins: int = 10) -> np.ndarray:
    ranked = pd.Series(y).rank(method="first")
    return pd.qcut(ranked, q=min(bins, len(y)), labels=False, duplicates="drop").to_numpy()


def _split_indices(y: np.ndarray, task: str, seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_idx = np.arange(len(y))
    strat = y if task == "classification" else _regression_bins(y)
    chosen, _ = train_test_split(
        all_idx,
        train_size=640,
        random_state=seed,
        stratify=strat,
    )
    chosen_strat = strat[chosen]
    candidate, query = train_test_split(
        chosen,
        train_size=256,
        random_state=seed + 101,
        stratify=chosen_strat,
    )
    query_strat = strat[query]
    selector, test = train_test_split(
        query,
        train_size=128,
        random_state=seed + 202,
        stratify=query_strat,
    )
    return np.asarray(candidate), np.asarray(selector), np.asarray(test)


def _make_row_representations(
    X: pd.DataFrame,
    y: np.ndarray,
    task: str,
    candidate_idx: np.ndarray,
    selector_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    Xc = X.iloc[candidate_idx]
    numeric = list(Xc.select_dtypes(include=[np.number, "bool"]).columns)
    categorical = [c for c in Xc.columns if c not in numeric]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    encoder = ColumnTransformer(transformers, sparse_threshold=0.0)
    encoded_c = np.asarray(encoder.fit_transform(Xc), dtype=np.float32)
    n_components = min(16, encoded_c.shape[0] - 1, encoded_c.shape[1])
    pca = PCA(n_components=n_components, random_state=0)
    feature_c_raw = pca.fit_transform(encoded_c).astype(np.float32)
    feature_s_raw = pca.transform(encoder.transform(X.iloc[selector_idx])).astype(np.float32)
    feature_t_raw = pca.transform(encoder.transform(X.iloc[test_idx])).astype(np.float32)
    pca_scaler = StandardScaler().fit(feature_c_raw)
    feature_c = pca_scaler.transform(feature_c_raw).astype(np.float32)
    feature_s = pca_scaler.transform(feature_s_raw).astype(np.float32)
    feature_t = pca_scaler.transform(feature_t_raw).astype(np.float32)
    yc = y[candidate_idx]
    if task == "classification":
        classes = np.unique(y)
        class_to_idx = {v: i for i, v in enumerate(classes)}
        target = np.zeros((len(yc), len(classes)), dtype=np.float32)
        target[np.arange(len(yc)), [class_to_idx[v] for v in yc]] = 1.0
        bins = np.asarray([class_to_idx[v] for v in yc], dtype=int)
    else:
        target = ((yc - yc.mean()) / max(yc.std(), 1e-8)).astype(np.float32)[:, None]
        bins = _regression_bins(yc, 10)
    z = np.concatenate([feature_c, target], axis=1).astype(np.float32)
    return z, feature_c, feature_s, feature_t, bins


def load_dataset(name: str) -> DatasetBundle:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset {name!r}")
    from sklearn.datasets import fetch_california_housing, fetch_openml

    spec = DATASETS[name]
    if spec.get("source") == "sklearn":
        bunch = fetch_california_housing(as_frame=True)
        X = bunch.data
        raw_y = np.asarray(bunch.target)
        source_id = "sklearn:california_housing"
    else:
        bunch = fetch_openml(data_id=spec["openml_id"], as_frame=True, parser="auto")
        X = bunch.data
        raw_y = np.asarray(bunch.target)
        source_id = f"OpenML:{spec['openml_id']} ({bunch.details.get('name', name)})"
    X = X.copy()
    # Normalize pandas missing markers but leave feature semantics to official TabICL preprocessing.
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = X[col].astype("object").where(pd.notna(X[col]), "__MISSING__")
    classes = None
    if spec["task"] == "classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(raw_y)
        classes = encoder.classes_
    else:
        y = raw_y.astype(np.float64)
    candidate, selector, test = _split_indices(y, spec["task"], 0)
    z, fz, sfz, tfz, bins = _make_row_representations(X, y, spec["task"], candidate, selector, test)
    return DatasetBundle(
        name,
        spec["task"],
        source_id,
        X,
        y,
        classes,
        candidate,
        selector,
        test,
        z,
        fz,
        sfz,
        tfz,
        bins,
    )


def sample_context(y: np.ndarray, k: int, rng: np.random.Generator, task: str) -> np.ndarray:
    available = np.arange(len(y))
    if task != "classification":
        return np.sort(rng.choice(available, size=k, replace=False))
    chosen: list[int] = []
    for cls in np.unique(y):
        cls_idx = available[y == cls]
        if len(chosen) < k and len(cls_idx):
            chosen.append(int(rng.choice(cls_idx)))
    remaining = np.setdiff1d(available, np.asarray(chosen, dtype=int), assume_unique=False)
    if len(chosen) < k:
        chosen.extend(map(int, rng.choice(remaining, size=k - len(chosen), replace=False)))
    return np.sort(np.asarray(chosen, dtype=int))


def membership(indices: Iterable[int], n: int = 256) -> np.ndarray:
    x = np.zeros(n, dtype=np.float32)
    x[np.fromiter(indices, dtype=int)] = 1.0
    return x


def indices_string(indices: Iterable[int]) -> str:
    return ";".join(map(str, sorted(map(int, indices))))


def parse_indices(value: str) -> np.ndarray:
    return np.fromiter((int(v) for v in value.split(";") if v), dtype=int)


class TabICLEvaluator:
    """Frozen official TabICLv2 estimator with checkpoint reuse across contexts."""

    def __init__(self, bundle: DatasetBundle, device: str, seed: int = 0, n_estimators: int = 1):
        from tabicl import TabICLClassifier, TabICLRegressor

        self.bundle = bundle
        cls = TabICLClassifier if bundle.task == "classification" else TabICLRegressor
        kwargs: dict[str, Any] = dict(
            n_estimators=n_estimators,
            norm_methods="none" if n_estimators == 1 else None,
            feat_shuffle_method="none" if n_estimators == 1 else "latin",
            device=device,
            use_amp=True,
            batch_size=max(1, n_estimators),
            allow_auto_download=False,
            random_state=seed,
        )
        if bundle.task == "classification":
            kwargs["class_shuffle_method"] = "none" if n_estimators == 1 else "shift"
        self.model = cls(**kwargs)
        original_load = self.model._load_model

        def cached_load(estimator: Any) -> None:
            if not hasattr(estimator, "model_"):
                original_load()

        self.model._load_model = types.MethodType(cached_load, self.model)
        self.checkpoint_path: str | None = None
        self.total_seconds = 0.0
        self.evaluations = 0

    def evaluate(
        self,
        indices: np.ndarray,
        split: str = "selector",
        return_prediction: bool = False,
    ) -> dict[str, Any]:
        b = self.bundle
        Xq = b.X_selector if split == "selector" else b.X_test
        yq = b.y_selector if split == "selector" else b.y_test
        started = time.perf_counter()
        self.model.fit(b.X_candidate.iloc[indices], b.y_candidate[indices])
        if b.task == "classification":
            prediction = np.asarray(self.model.predict_proba(Xq), dtype=np.float64)
            # All generated contexts contain every class, but retain a safe expansion.
            if prediction.shape[1] != len(np.unique(b.y_candidate)):
                expanded = np.full((len(yq), len(np.unique(b.y_candidate))), 1e-12)
                expanded[:, self.model.classes_.astype(int)] = prediction
                prediction = expanded / expanded.sum(axis=1, keepdims=True)
            prediction = np.clip(prediction, 1e-12, 1.0)
            prediction = prediction / prediction.sum(axis=1, keepdims=True)
            ll = log_loss(yq, prediction, labels=np.arange(prediction.shape[1]))
            hard = prediction.argmax(axis=1)
            record: dict[str, Any] = {
                "utility": -float(ll),
                "logloss": float(ll),
                "accuracy": float(accuracy_score(yq, hard)),
                "roc_auc": np.nan,
                "rmse": np.nan,
                "mae": np.nan,
            }
            if prediction.shape[1] == 2:
                record["roc_auc"] = float(roc_auc_score(yq, prediction[:, 1]))
        else:
            prediction = np.asarray(self.model.predict(Xq), dtype=np.float64)
            rmse = math.sqrt(mean_squared_error(yq, prediction))
            record = {
                "utility": -float(rmse / max(b.y_candidate.std(), 1e-12)),
                "logloss": np.nan,
                "accuracy": np.nan,
                "roc_auc": np.nan,
                "rmse": float(rmse),
                "mae": float(mean_absolute_error(yq, prediction)),
            }
        elapsed = time.perf_counter() - started
        self.total_seconds += elapsed
        self.evaluations += 1
        self.checkpoint_path = str(self.model.model_path_)
        record["runtime_seconds"] = elapsed
        if return_prediction:
            record["prediction"] = prediction
        return record


def prediction_metrics(y: np.ndarray, pred: np.ndarray, task: str, y_scale: float) -> dict[str, float]:
    if task == "classification":
        ll = log_loss(y, pred, labels=np.arange(pred.shape[1]))
        out = {
            "utility": -float(ll),
            "logloss": float(ll),
            "accuracy": float(accuracy_score(y, pred.argmax(axis=1))),
            "roc_auc": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
        }
        if pred.shape[1] == 2:
            out["roc_auc"] = float(roc_auc_score(y, pred[:, 1]))
        return out
    rmse = math.sqrt(mean_squared_error(y, pred))
    return {
        "utility": -float(rmse / max(y_scale, 1e-12)),
        "logloss": np.nan,
        "accuracy": np.nan,
        "roc_auc": np.nan,
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y, pred)),
    }


def surrogate_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rho = spearmanr(y, pred).statistic
    return {
        "r2": float(r2_score(y, pred)),
        "spearman": float(rho) if np.isfinite(rho) else 0.0,
        "mae": float(mean_absolute_error(y, pred)),
    }


def fit_ridge(X_train: np.ndarray, y_train: np.ndarray, alphas: list[float]) -> Ridge:
    cv = min(5, max(2, len(y_train) // 50))
    search = GridSearchCV(Ridge(), {"alpha": alphas}, cv=cv, scoring="neg_mean_squared_error")
    search.fit(X_train, y_train)
    return search.best_estimator_


class FM(torch.nn.Module):
    def __init__(self, n: int, rank: int, interaction_only: bool = False):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.linear = None if interaction_only else torch.nn.Parameter(torch.zeros(n))
        self.v = torch.nn.Parameter(torch.randn(n, rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xv = x @ self.v
        interactions = 0.5 * ((xv * xv) - x @ (self.v * self.v)).sum(dim=1)
        out = self.bias + interactions
        if self.linear is not None:
            out = out + x @ self.linear
        return out

    def pair_matrix(self) -> np.ndarray:
        v = self.v.detach().cpu().numpy()
        b = v @ v.T
        np.fill_diagonal(b, 0.0)
        return b


class FeatureFM(torch.nn.Module):
    def __init__(self, z: np.ndarray, rank: int):
        super().__init__()
        self.register_buffer("z", torch.as_tensor(z, dtype=torch.float32))
        hidden = max(16, 2 * rank)
        self.g = torch.nn.Sequential(
            torch.nn.Linear(z.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, rank)
        )
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.linear = torch.nn.Parameter(torch.zeros(len(z)))

    def factors(self) -> torch.Tensor:
        return self.g(self.z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = self.factors()
        xv = x @ v
        inter = 0.5 * ((xv * xv) - x @ (v * v)).sum(dim=1)
        return self.bias + x @ self.linear + inter

    def pair_matrix(self) -> np.ndarray:
        v = self.factors().detach().cpu().numpy()
        b = v @ v.T
        np.fill_diagonal(b, 0.0)
        return b


class SignedBilinear(torch.nn.Module):
    def __init__(self, z: np.ndarray, rank: int):
        super().__init__()
        self.register_buffer("z", torch.as_tensor(z, dtype=torch.float32))
        self.u = torch.nn.Parameter(torch.randn(z.shape[1], rank) * 0.01)
        self.v = torch.nn.Parameter(torch.randn(z.shape[1], rank) * 0.01)
        self.bias = torch.nn.Parameter(torch.zeros(()))
        self.linear = torch.nn.Parameter(torch.zeros(len(z)))

    def pair_matrix_tensor(self) -> torch.Tensor:
        p, q = self.z @ self.u, self.z @ self.v
        b = 0.5 * (p @ q.T + q @ p.T)
        return b - torch.diag_embed(torch.diagonal(b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.pair_matrix_tensor()
        return self.bias + x @ self.linear + 0.5 * ((x @ b) * x).sum(dim=1)

    def pair_matrix(self) -> np.ndarray:
        return self.pair_matrix_tensor().detach().cpu().numpy()


class DeepSets(torch.nn.Module):
    def __init__(self, z: np.ndarray, hidden: int = 32):
        super().__init__()
        self.register_buffer("z", torch.as_tensor(z, dtype=torch.float32))
        self.phi = torch.nn.Sequential(
            torch.nn.Linear(z.shape[1], hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, hidden), torch.nn.ReLU()
        )
        self.rho = torch.nn.Sequential(torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x @ self.phi(self.z)
        return self.rho(pooled).squeeze(1)


@dataclass
class TorchFit:
    model: torch.nn.Module
    val_mse: float
    epochs: int
    target_mean: float
    target_scale: float

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            raw = self.model(torch.as_tensor(X, dtype=torch.float32)).cpu().numpy()
        return raw * self.target_scale + self.target_mean


def fit_torch_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
    weight_decay: float = 1e-3,
    max_epochs: int = 1200,
    patience: int = 100,
) -> TorchFit:
    torch.manual_seed(seed)
    # Constructors run before this function, so explicitly reset every learned
    # parameter after seeding. This makes reruns independent of process RNG history.
    for module in model.modules():
        if module is not model and hasattr(module, "reset_parameters"):
            module.reset_parameters()
    with torch.no_grad():
        if isinstance(model, FM):
            model.bias.zero_()
            if model.linear is not None:
                model.linear.zero_()
            model.v.normal_(mean=0.0, std=0.01)
        elif isinstance(model, FeatureFM):
            model.bias.zero_()
            model.linear.zero_()
        elif isinstance(model, SignedBilinear):
            model.bias.zero_()
            model.linear.zero_()
            model.u.normal_(mean=0.0, std=0.01)
            model.v.normal_(mean=0.0, std=0.01)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = model.to(device)
    train_i, val_i = train_test_split(np.arange(len(y)), test_size=0.15, random_state=seed)
    target_mean = float(y[train_i].mean())
    target_scale = float(max(y[train_i].std(), 1e-6))
    yn = ((y - target_mean) / target_scale).astype(np.float32)
    xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(yn, dtype=torch.float32, device=device)
    train_index = torch.as_tensor(train_i, dtype=torch.long, device=device)
    val_index = torch.as_tensor(val_i, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best = float("inf")
    stale = 0
    epoch = 0
    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(xt[train_index]), yt[train_index])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(torch.nn.functional.mse_loss(model(xt[val_index]), yt[val_index]))
        if val < best - 1e-6:
            best = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to("cpu")
    return TorchFit(model, best * target_scale * target_scale, epoch + 1, target_mean, target_scale)


def environment_record(bundle: DatasetBundle, evaluator: TabICLEvaluator | None = None) -> dict[str, Any]:
    import importlib.metadata as metadata

    packages = {}
    for name in ["tabicl", "tabpfn", "torch", "scikit-learn", "openml", "numpy", "pandas", "scipy"]:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    if torch.cuda.is_available():
        gpu = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return {
        "dataset": bundle.name,
        "task": bundle.task,
        "source_id": bundle.source_id,
        "full_rows": len(bundle.y),
        "features": bundle.X.shape[1],
        "candidate_rows": len(bundle.candidate_idx),
        "selector_rows": len(bundle.selector_idx),
        "test_rows": len(bundle.test_idx),
        "python": platform.python_version(),
        "packages": packages,
        "gpu": gpu,
        "checkpoint": evaluator.checkpoint_path if evaluator else None,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)
