"""First-layer blockwise adaptive optimizers for basis-sensitive feature blocks."""

from __future__ import annotations

from typing import Any, Iterable

import torch


class BlockAdaptiveOptimizer(torch.optim.Optimizer):
    """Adam-style updates with block-invariant second moments on one weight.

    Parameters outside the declared input-coordinate blocks receive ordinary
    coordinatewise Adam updates.  The expected weight layout is
    ``(output_channels, input_coordinates)``.
    """

    def __init__(
        self,
        parameter: torch.nn.Parameter,
        blocks: Iterable[Iterable[int]],
        *,
        method: str,
        lr: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        alpha: float = 0.0,
        eigenvalue_floor: float = 1e-8,
        normalization: str = "mean",
    ) -> None:
        if parameter.ndim != 2:
            raise ValueError(f"block optimizer requires a matrix parameter, got {parameter.shape}")
        if method not in {"block_scalar_adam", "block_adam", "matrix_adam", "soft_block_adam"}:
            raise ValueError(f"unsupported block method {method}")
        beta1, beta2 = betas
        defaults = dict(
            lr=float(lr),
            betas=(float(beta1), float(beta2)),
            eps=float(eps),
            weight_decay=float(weight_decay),
            alpha=float(alpha),
            eigenvalue_floor=float(eigenvalue_floor),
            normalization=str(normalization),
        )
        super().__init__([parameter], defaults)
        self.method = method
        self.blocks = [torch.as_tensor(list(indices), dtype=torch.long, device=parameter.device) for indices in blocks]
        covered = torch.zeros(parameter.shape[1], dtype=torch.bool, device=parameter.device)
        for indices in self.blocks:
            if len(indices) == 0:
                raise ValueError("empty optimizer block")
            if int(indices.min()) < 0 or int(indices.max()) >= parameter.shape[1]:
                raise ValueError("optimizer block index out of bounds")
            if covered[indices].any():
                raise ValueError("overlapping optimizer blocks")
            covered[indices] = True
        self.uncovered = torch.nonzero(~covered, as_tuple=False).flatten()

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        group = self.param_groups[0]
        parameter = group["params"][0]
        if parameter.grad is None:
            return loss
        gradient = parameter.grad
        if gradient.is_sparse:
            raise RuntimeError("sparse gradients are unsupported")
        state = self.state[parameter]
        if not state:
            state["step"] = 0
            state["first_moment"] = torch.zeros_like(parameter)
            state["coordinate_second_moment"] = torch.zeros_like(parameter)
            if self.method == "block_scalar_adam":
                state["block_second_moments"] = [
                    torch.zeros((), dtype=parameter.dtype, device=parameter.device) for _ in self.blocks
                ]
            elif self.method in {"block_adam", "soft_block_adam"}:
                state["block_second_moments"] = [
                    torch.zeros(parameter.shape[0], dtype=parameter.dtype, device=parameter.device)
                    for _ in self.blocks
                ]
            else:
                state["block_second_moments"] = [
                    torch.zeros((len(indices), len(indices)), dtype=torch.float64, device=parameter.device)
                    for indices in self.blocks
                ]
        state["step"] += 1
        step = int(state["step"])
        beta1, beta2 = group["betas"]
        bias1 = 1.0 - beta1**step
        bias2 = 1.0 - beta2**step
        first = state["first_moment"]
        first.mul_(beta1).add_(gradient, alpha=1.0 - beta1)
        first_hat = first / bias1
        coordinate = state["coordinate_second_moment"]
        coordinate.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        if group["weight_decay"]:
            parameter.mul_(1.0 - group["lr"] * group["weight_decay"])

        if len(self.uncovered):
            denominator = (coordinate[:, self.uncovered] / bias2).sqrt().add_(group["eps"])
            updated = parameter[:, self.uncovered].addcdiv(
                first_hat[:, self.uncovered], denominator, value=-group["lr"]
            )
            parameter.index_copy_(1, self.uncovered, updated)

        for block_index, indices in enumerate(self.blocks):
            block_gradient = gradient[:, indices]
            block_first = first_hat[:, indices]
            second = state["block_second_moments"][block_index]
            if self.method == "block_scalar_adam":
                statistic = block_gradient.square().mean()
                if group["normalization"] == "sum_over_sqrt_k":
                    statistic = block_gradient.square().sum() / (block_gradient.numel() ** 0.5)
                second.mul_(beta2).add_(statistic, alpha=1.0 - beta2)
                denominator = (second / bias2).sqrt().add(group["eps"])
                updated = parameter[:, indices].add(block_first / denominator, alpha=-group["lr"])
                parameter.index_copy_(1, indices, updated)
            elif self.method in {"block_adam", "soft_block_adam"}:
                statistic = block_gradient.square().mean(dim=1)
                if group["normalization"] == "sum_over_sqrt_k":
                    statistic = block_gradient.square().sum(dim=1) / (len(indices) ** 0.5)
                second.mul_(beta2).add_(statistic, alpha=1.0 - beta2)
                block_second = (second / bias2)[:, None]
                if self.method == "soft_block_adam":
                    coordinate_second = coordinate[:, indices] / bias2
                    mixed = group["alpha"] * coordinate_second + (1.0 - group["alpha"]) * block_second
                    denominator = mixed.sqrt().add_(group["eps"])
                else:
                    denominator = block_second.sqrt().add_(group["eps"])
                updated = parameter[:, indices].addcdiv(block_first, denominator, value=-group["lr"])
                parameter.index_copy_(1, indices, updated)
            else:
                gradient64 = block_gradient.to(torch.float64)
                statistic = gradient64.T @ gradient64 / max(block_gradient.shape[0], 1)
                if group["normalization"] == "sum_over_sqrt_k":
                    statistic = statistic * (block_gradient.shape[0] / (len(indices) ** 0.5))
                second.mul_(beta2).add_(statistic, alpha=1.0 - beta2)
                corrected = second / bias2
                eigenvalues, eigenvectors = torch.linalg.eigh(corrected)
                largest = float(torch.clamp(eigenvalues.max(), min=0.0).item())
                floor = max(group["eps"], group["eigenvalue_floor"] * max(largest, 1.0))
                inverse_sqrt = (
                    eigenvectors
                    @ torch.diag(torch.clamp(eigenvalues, min=floor).rsqrt())
                    @ eigenvectors.T
                )
                update = block_first.to(torch.float64) @ inverse_sqrt
                updated = parameter[:, indices].add(update.to(parameter.dtype), alpha=-group["lr"])
                parameter.index_copy_(1, indices, updated)
        return loss


def make_optimizers(
    model: torch.nn.Module,
    first_weight: torch.nn.Parameter,
    blocks: dict[str, list[int]],
    *,
    method: str,
    lr: float,
    weight_decay: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    alpha: float = 0.0,
    eigenvalue_floor: float = 1e-8,
    normalization: str = "mean",
) -> list[torch.optim.Optimizer]:
    if method == "adamw":
        return [torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2), eps=epsilon)]
    if method == "sgd_control":
        return [torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)]
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter is not first_weight and parameter.requires_grad
    ]
    ordinary = torch.optim.AdamW(
        other_parameters,
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        eps=epsilon,
    )
    block = BlockAdaptiveOptimizer(
        first_weight,
        [indices for _, indices in sorted(blocks.items(), key=lambda item: min(item[1]))],
        method=method,
        lr=lr,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
        alpha=alpha,
        eigenvalue_floor=eigenvalue_floor,
        normalization=normalization,
    )
    return [ordinary, block]


def zero_grad(optimizers: Iterable[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)


def step(optimizers: Iterable[torch.optim.Optimizer]) -> None:
    for optimizer in optimizers:
        optimizer.step()
