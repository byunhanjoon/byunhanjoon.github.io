"""PriorDial: fixed warp marginals with tunable mechanism–warp dependence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from src.transforms import make_warp


MECHANISMS = ("linear", "additive", "threshold", "interaction", "partition", "periodic")
WARPS = ("identity", "affine", "signed_power", "asinh", "pwl", "sinh")


def population_coupling_mi(rho: float, n_families: int = len(MECHANISMS)) -> float:
    """Return mutual information of the ideal balanced coupling, in nats.

    The population scheduler draws ``C`` uniformly, then sets ``W=C`` with
    probability ``rho`` and otherwise draws ``W`` independently and uniformly.
    Both marginals stay uniform. This calibrates the dial's information strength;
    information is deliberately not assumed to be linear in ``rho``.
    """
    if not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]")
    if n_families < 2:
        raise ValueError("n_families must be at least two")
    diagonal = rho + (1.0 - rho) / n_families
    off_diagonal = (1.0 - rho) / n_families
    value = diagonal * np.log(n_families * diagonal)
    if off_diagonal > 0:
        value += (n_families - 1) * off_diagonal * np.log(n_families * off_diagonal)
    return float(value)


@dataclass(frozen=True)
class Episode:
    context_x: np.ndarray
    context_y: np.ndarray
    query_x: np.ndarray
    query_y: np.ndarray
    latent_context_z: np.ndarray
    latent_query_z: np.ndarray
    metadata: dict[str, Any]


def balanced_coupling_schedule(
    n_tasks: int,
    rho: float,
    rng: np.random.Generator,
    mechanisms: Iterable[str] = MECHANISMS,
    warps: Iterable[str] = WARPS,
) -> list[tuple[str, str, bool]]:
    """Return an exactly marginal-balanced schedule.

    The correlated and independent portions are each balanced. Correlated task count is
    rounded to the nearest multiple of the number of families; `effective_rho` is stored
    later rather than pretending the requested finite-sample proportion was exact.
    """
    mechanisms = tuple(mechanisms)
    warps = tuple(warps)
    if len(mechanisms) != len(warps):
        raise ValueError("one-to-one dial requires equal mechanism and warp counts")
    k = len(mechanisms)
    if n_tasks <= 0 or n_tasks % k:
        raise ValueError(f"n_tasks must be a positive multiple of {k}")
    if not 0 <= rho <= 1:
        raise ValueError("rho must lie in [0, 1]")
    n_correlated = int(np.clip(np.rint(rho * n_tasks / k) * k, 0, n_tasks))
    n_independent = n_tasks - n_correlated

    c_corr = np.tile(np.arange(k), n_correlated // k)
    corr = [(mechanisms[c], warps[c], True) for c in c_corr]

    # A cyclic near-product design avoids accidental finite-sample C--W dependence.
    # For each repetition, all mechanisms and warps appear once. Across repetitions,
    # offsets rotate, making every joint cell differ in count by at most one.
    independent = []
    for repetition in range(n_independent // k):
        offset = repetition % k
        for c in range(k):
            independent.append((mechanisms[c], warps[(c + offset) % k], False))
    schedule = corr + independent
    rng.shuffle(schedule)
    return schedule


class PriorDial:
    def __init__(
        self,
        seed: int,
        n_context: int = 64,
        n_query: int = 128,
        n_features: int = 8,
        task_type: str = "classification",
        informative_fraction: float = 0.75,
        correlation: float = 0.0,
        label_noise: float = 0.05,
        classification_logit_scale: float = 2.5,
    ) -> None:
        if task_type not in {"classification", "regression"}:
            raise ValueError("task_type must be classification or regression")
        if n_context < 8 or n_query < 1 or n_features < 2:
            raise ValueError("episode dimensions are too small")
        if not 0 < informative_fraction <= 1:
            raise ValueError("informative_fraction must lie in (0,1]")
        if not 0 <= correlation < 1:
            raise ValueError("correlation must lie in [0,1)")
        self.rng = np.random.default_rng(seed)
        self.seed = int(seed)
        self.n_context = int(n_context)
        self.n_query = int(n_query)
        self.n_features = int(n_features)
        self.task_type = task_type
        self.informative_fraction = float(informative_fraction)
        self.correlation = float(correlation)
        self.label_noise = float(label_noise)
        self.classification_logit_scale = float(classification_logit_scale)

    def _latent(self, n: int) -> np.ndarray:
        independent = self.rng.normal(size=(n, self.n_features))
        if self.correlation == 0:
            return independent
        shared = self.rng.normal(size=(n, 1))
        return np.sqrt(1 - self.correlation) * independent + np.sqrt(self.correlation) * shared

    def _mechanism(self, name: str, z: np.ndarray, state: dict[str, Any] | None = None):
        d_info = max(1, int(np.ceil(self.n_features * self.informative_fraction)))
        if state is None:
            state = {"d_info": d_info}
            if name == "linear":
                coef = self.rng.normal(size=d_info)
                mask = self.rng.random(d_info) < 0.65
                if not mask.any():
                    mask[self.rng.integers(d_info)] = True
                state["coef"] = coef * mask
            elif name == "additive":
                state["coef"] = self.rng.normal(size=d_info)
                state["phase"] = self.rng.uniform(-np.pi, np.pi, size=d_info)
            elif name == "threshold":
                state["coef"] = self.rng.choice([-1.0, 1.0], size=d_info)
                state["threshold"] = self.rng.uniform(-0.8, 0.8, size=d_info)
            elif name == "interaction":
                pairs = [(j, (j + 1) % d_info) for j in range(0, d_info, 2)]
                state["pairs"] = pairs
                state["coef"] = self.rng.normal(size=len(pairs))
            elif name == "partition":
                state["features"] = self.rng.integers(0, d_info, size=3)
                state["threshold"] = self.rng.uniform(-0.7, 0.7, size=3)
                state["leaf"] = self.rng.normal(size=8)
            elif name == "periodic":
                state["coef"] = self.rng.normal(size=d_info)
                state["frequency"] = self.rng.uniform(1.5, 4.0, size=d_info)
                state["phase"] = self.rng.uniform(-np.pi, np.pi, size=d_info)
            else:
                raise KeyError(name)

        zi = z[:, : int(state["d_info"])]
        if name == "linear":
            score = zi @ np.asarray(state["coef"])
        elif name == "additive":
            score = np.sum(
                np.asarray(state["coef"]) * np.tanh(zi + np.asarray(state["phase"]) / 3),
                axis=1,
            )
        elif name == "threshold":
            score = np.sum(
                np.asarray(state["coef"]) * (zi > np.asarray(state["threshold"])), axis=1
            )
        elif name == "interaction":
            score = sum(
                coef * zi[:, a] * zi[:, b]
                for coef, (a, b) in zip(state["coef"], state["pairs"], strict=True)
            )
        elif name == "partition":
            bits = np.zeros(z.shape[0], dtype=int)
            for bit, (feature, threshold) in enumerate(
                zip(state["features"], state["threshold"], strict=True)
            ):
                bits += (zi[:, feature] > threshold).astype(int) << bit
            score = np.asarray(state["leaf"])[bits]
        elif name == "periodic":
            score = np.sum(
                np.asarray(state["coef"])
                * np.sin(np.asarray(state["frequency"]) * zi + np.asarray(state["phase"])),
                axis=1,
            )
        else:  # pragma: no cover
            raise KeyError(name)
        return np.asarray(score, dtype=np.float64), state

    def generate(self, mechanism: str, warp: str, coupled: bool = False) -> Episode:
        z_all = self._latent(self.n_context + self.n_query)
        z_context = z_all[: self.n_context]
        z_query = z_all[self.n_context :]
        score_all, mechanism_state = self._mechanism(mechanism, z_all)
        context_score = score_all[: self.n_context]
        query_score = score_all[self.n_context :]
        scale = max(float(np.std(context_score)), 1e-8)

        if self.task_type == "classification":
            centered = self.classification_logit_scale * (score_all - np.median(context_score)) / scale
            probability = 1.0 / (1.0 + np.exp(-np.clip(centered, -20, 20)))
            y_all = self.rng.binomial(1, probability).astype(np.int64)
            if self.label_noise:
                flips = self.rng.random(y_all.size) < self.label_noise
                y_all[flips] = 1 - y_all[flips]
        else:
            y_all = (score_all - np.mean(context_score)) / scale
            y_all += self.rng.normal(0, self.label_noise, size=y_all.size)

        x_context = np.empty_like(z_context)
        x_query = np.empty_like(z_query)
        transform_states = []
        for j in range(self.n_features):
            transform = make_warp(warp, self.rng).fit(z_context[:, j])
            x_context[:, j] = transform.transform(z_context[:, j])
            x_query[:, j] = transform.transform(z_query[:, j])
            transform_states.append(transform.state_dict())

        return Episode(
            context_x=x_context,
            context_y=y_all[: self.n_context],
            query_x=x_query,
            query_y=y_all[self.n_context :],
            latent_context_z=z_context,
            latent_query_z=z_query,
            metadata={
                "generator_seed": self.seed,
                "mechanism": mechanism,
                "warp": warp,
                "coupled_draw": bool(coupled),
                "task_type": self.task_type,
                "n_context": self.n_context,
                "n_query": self.n_query,
                "n_features": self.n_features,
                "informative_fraction": self.informative_fraction,
                "correlation": self.correlation,
                "label_noise": self.label_noise,
                "classification_logit_scale": self.classification_logit_scale,
                "mechanism_state": _jsonable(mechanism_state),
                "transform_states": transform_states,
            },
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
