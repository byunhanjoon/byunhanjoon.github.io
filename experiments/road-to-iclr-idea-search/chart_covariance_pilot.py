"""Algebraic pilot for chart-covariant tabular first-layer updates.

The pilot compares two affine-equivalent bases of the same one-dimensional
piecewise-linear function space.  It verifies that a field-metric gradient
update preserves matched functions over several steps, while coordinatewise
Adam does not.  This is a mechanism check, not a performance benchmark.
"""

from __future__ import annotations

import json

import numpy as np
import torch


torch.set_default_dtype(torch.float64)


def bases(x: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knots = np.linspace(0.0, 1.0, bins + 1)
    left, right = knots[:-1], knots[1:]
    width = right - left
    cumulative = np.clip((x[:, None] - left) / width, 0.0, 1.0)
    local = np.empty_like(cumulative)
    local[:, 0] = 1.0 - cumulative[:, 0]
    local[:, 1:] = cumulative[:, :-1] - cumulative[:, 1:]
    return cumulative, local, width


def stiffness_cumulative(width: np.ndarray) -> np.ndarray:
    # Integral of phi'_a(x) phi'_b(x) dx for the cumulative ramp basis.
    return np.diag(1.0 / width)


def forward(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, head: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x @ weight.T + bias) @ head


def gradients(
    x: torch.Tensor,
    y: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    head: torch.Tensor,
    metric: torch.Tensor,
    regularization: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    for parameter in (weight, bias, head):
        parameter.requires_grad_(True)
    prediction = forward(x, weight, bias, head)
    loss = torch.mean((prediction - y) ** 2)
    penalty = regularization * torch.trace(weight @ metric @ weight.T)
    (loss + penalty).backward()
    return (
        weight.grad.detach().clone(),
        bias.grad.detach().clone(),
        head.grad.detach().clone(),
        loss.detach().clone(),
    )


def clear(*parameters: torch.Tensor) -> None:
    for parameter in parameters:
        parameter.grad = None
        parameter.requires_grad_(False)


def run_metric_updates(
    x_c: torch.Tensor,
    x_l: torch.Tensor,
    y: torch.Tensor,
    transform: torch.Tensor,
    metric_c: torch.Tensor,
    metric_l: torch.Tensor,
    steps: int,
) -> tuple[float, float]:
    generator = torch.Generator().manual_seed(7)
    weight_c = torch.randn((12, x_c.shape[1]), generator=generator) * 0.15
    weight_l = weight_c @ torch.linalg.inv(transform).T
    bias_c = torch.randn(12, generator=generator) * 0.03
    bias_l = bias_c.clone()
    head_c = torch.randn(12, generator=generator) * 0.1
    head_l = head_c.clone()
    momentum_c = torch.zeros_like(weight_c)
    momentum_l = torch.zeros_like(weight_l)
    learning_rate = 0.04
    regularization = 2e-4

    initial_gap = float(torch.max(torch.abs(
        forward(x_c, weight_c, bias_c, head_c) - forward(x_l, weight_l, bias_l, head_l)
    )))
    for _ in range(steps):
        grad_wc, grad_bc, grad_hc, _ = gradients(
            x_c, y, weight_c, bias_c, head_c, metric_c, regularization
        )
        clear(weight_c, bias_c, head_c)
        grad_wl, grad_bl, grad_hl, _ = gradients(
            x_l, y, weight_l, bias_l, head_l, metric_l, regularization
        )
        clear(weight_l, bias_l, head_l)

        direction_c = grad_wc @ torch.linalg.inv(metric_c)
        direction_l = grad_wl @ torch.linalg.inv(metric_l)
        momentum_c = 0.8 * momentum_c + direction_c
        momentum_l = 0.8 * momentum_l + direction_l
        weight_c -= learning_rate * momentum_c
        weight_l -= learning_rate * momentum_l
        # These parameters live outside the chart action and receive identical
        # gradients whenever the matched functions remain identical.
        bias_c -= learning_rate * grad_bc
        bias_l -= learning_rate * grad_bl
        head_c -= learning_rate * grad_hc
        head_l -= learning_rate * grad_hl

    final_gap = float(torch.max(torch.abs(
        forward(x_c, weight_c, bias_c, head_c) - forward(x_l, weight_l, bias_l, head_l)
    )))
    return initial_gap, final_gap


def run_adam_updates(
    x_c: torch.Tensor,
    x_l: torch.Tensor,
    y: torch.Tensor,
    transform: torch.Tensor,
    steps: int,
) -> float:
    generator = torch.Generator().manual_seed(7)
    weight_c = torch.nn.Parameter(torch.randn((12, x_c.shape[1]), generator=generator) * 0.15)
    weight_l = torch.nn.Parameter(weight_c.detach() @ torch.linalg.inv(transform).T)
    bias_c = torch.nn.Parameter(torch.randn(12, generator=generator) * 0.03)
    bias_l = torch.nn.Parameter(bias_c.detach().clone())
    head_c = torch.nn.Parameter(torch.randn(12, generator=generator) * 0.1)
    head_l = torch.nn.Parameter(head_c.detach().clone())
    optimizer_c = torch.optim.AdamW([weight_c, bias_c, head_c], lr=0.01, weight_decay=2e-4)
    optimizer_l = torch.optim.AdamW([weight_l, bias_l, head_l], lr=0.01, weight_decay=2e-4)

    for _ in range(steps):
        for optimizer, x, weight, bias, head in (
            (optimizer_c, x_c, weight_c, bias_c, head_c),
            (optimizer_l, x_l, weight_l, bias_l, head_l),
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((forward(x, weight, bias, head) - y) ** 2)
            loss.backward()
            optimizer.step()
    gap = torch.max(torch.abs(
        forward(x_c, weight_c, bias_c, head_c) - forward(x_l, weight_l, bias_l, head_l)
    ))
    return float(gap.detach())


def main() -> None:
    rng = np.random.default_rng(2026)
    train_x = np.sort(rng.uniform(0.0, 1.0, 512))
    evaluation_x = np.linspace(0.0, 1.0, 1001)
    cumulative_train, local_train, width = bases(train_x, bins=10)
    cumulative_eval, local_eval, _ = bases(evaluation_x, bins=10)

    mean_c = cumulative_train.mean(axis=0)
    mean_l = local_train.mean(axis=0)
    cumulative_train -= mean_c
    local_train -= mean_l
    cumulative_eval -= mean_c
    local_eval -= mean_l
    transform, *_ = np.linalg.lstsq(cumulative_train, local_train, rcond=1e-12)
    chart_residual = np.linalg.norm(cumulative_train @ transform - local_train) / np.linalg.norm(local_train)

    mass_c = cumulative_train.T @ cumulative_train / len(cumulative_train)
    stiffness_c = stiffness_cumulative(width)
    metric_c = mass_c + 2e-3 * stiffness_c
    metric_l = transform.T @ metric_c @ transform

    x_c = torch.from_numpy(cumulative_eval)
    x_l = torch.from_numpy(local_eval)
    transform_t = torch.from_numpy(transform)
    metric_c_t = torch.from_numpy(metric_c)
    metric_l_t = torch.from_numpy(metric_l)
    y = torch.from_numpy(
        np.sin(5.0 * evaluation_x) + 0.35 * (evaluation_x > 0.63)
    )

    initial_gap, metric_gap = run_metric_updates(
        x_c, x_l, y, transform_t, metric_c_t, metric_l_t, steps=25
    )
    adam_gap = run_adam_updates(x_c, x_l, y, transform_t, steps=25)
    print(json.dumps({
        "chart_reconstruction_relative_error": chart_residual,
        "matched_initial_prediction_max_gap": initial_gap,
        "field_metric_prediction_max_gap_after_25_steps": metric_gap,
        "adamw_prediction_max_gap_after_25_steps": adam_gap,
        "metric_condition_cumulative": float(np.linalg.cond(metric_c)),
        "metric_condition_local": float(np.linalg.cond(metric_l)),
    }, indent=2))


if __name__ == "__main__":
    main()
