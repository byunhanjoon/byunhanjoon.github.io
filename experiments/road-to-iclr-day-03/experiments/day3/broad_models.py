"""Architectures, metrics, and scoped matrix optimizers for the broad study."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import tabm
import torch
from rtdl_revisiting_models import FTTransformerBackbone
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn

from .core import MLP, Prepared, ResNet


class DenseStemFTTransformer(nn.Module):
    """Official FT-Transformer backbone behind an exact dense affine stem."""

    def __init__(self, input_size: int, output_size: int, d_token: int = 32, n_tokens: int = 8):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_token = d_token
        self.first = nn.Linear(input_size, n_tokens * d_token)
        self.backbone = FTTransformerBackbone(
            d_out=output_size,
            n_blocks=2,
            d_block=d_token,
            attention_n_heads=4,
            attention_dropout=0.1,
            ffn_d_hidden=192,
            ffn_d_hidden_multiplier=None,
            ffn_dropout=0.1,
            ffn_activation="ReGLU",
            residual_dropout=0.0,
            n_tokens=None,
            linformer_kv_compression_ratio=None,
            linformer_kv_compression_sharing=None,
        )
        self.cls = nn.Parameter(torch.empty(1, 1, d_token))
        nn.init.normal_(self.cls, std=d_token**-0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.first(x).reshape(len(x), self.n_tokens, self.d_token)
        cls = self.cls.expand(len(x), -1, -1)
        return self.backbone(torch.cat((cls, tokens), dim=1))


class DenseStemTabM(nn.Module):
    """Official TabM backbone behind an exact dense affine stem."""

    def __init__(self, input_size: int, output_size: int, latent_size: int = 64):
        super().__init__()
        self.first = nn.Linear(input_size, latent_size)
        self.backbone = tabm.TabM.make(
            n_num_features=latent_size,
            cat_cardinalities=[],
            d_out=output_size,
            num_embeddings=None,
            n_blocks=2,
            d_block=192,
            dropout=0.1,
            k=8,
        )

    def forward_members(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(self.first(x), None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_members(x).mean(dim=1)


def make_model(name: str, input_size: int, output_size: int) -> nn.Module:
    if name == "mlp":
        return MLP(input_size, output_size, 256, 3, 0.1)
    if name == "resnet":
        return ResNet(input_size, output_size, 256, 3, 0.1)
    if name == "dense_stem_ft_transformer":
        return DenseStemFTTransformer(input_size, output_size)
    if name == "dense_stem_tabm":
        return DenseStemTabM(input_size, output_size)
    raise KeyError(name)


def member_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, task: str) -> torch.Tensor:
    if hasattr(model, "forward_members"):
        prediction = model.forward_members(x)
        if task == "binclass":
            target = y.float()[:, None].expand_as(prediction.squeeze(-1))
            return nn.functional.binary_cross_entropy_with_logits(prediction.squeeze(-1), target)
        if task == "multiclass":
            target = y.long()[:, None].expand(-1, prediction.shape[1])
            return nn.functional.cross_entropy(prediction.flatten(0, 1), target.flatten())
        target = y.float()[:, None].expand_as(prediction.squeeze(-1))
        return nn.functional.mse_loss(prediction.squeeze(-1), target)
    prediction = model(x)
    if task == "binclass":
        return nn.functional.binary_cross_entropy_with_logits(prediction.squeeze(-1), y.float())
    if task == "multiclass":
        return nn.functional.cross_entropy(prediction, y.long())
    return nn.functional.mse_loss(prediction.squeeze(-1), y.float())


def predictions(model: nn.Module, x: torch.Tensor, task: str) -> torch.Tensor:
    if hasattr(model, "forward_members"):
        raw = model.forward_members(x)
        if task == "binclass":
            probabilities = raw.squeeze(-1).sigmoid().mean(dim=1)
            return torch.logit(probabilities.clamp(1e-7, 1 - 1e-7))[:, None]
        if task == "multiclass":
            probabilities = raw.softmax(dim=-1).mean(dim=1)
            return probabilities.clamp_min(1e-12).log()
        return raw.mean(dim=1)
    return model(x)


def metrics(data: Prepared, logits: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if data.task == "binclass":
        scores = logits.reshape(-1)
        return {
            "roc_auc": float(roc_auc_score(y, scores)),
            "accuracy": float(accuracy_score(y, scores >= 0)),
            "primary": float(roc_auc_score(y, scores)),
        }
    if data.task == "multiclass":
        value = float(accuracy_score(y, logits.argmax(axis=1)))
        return {"accuracy": value, "primary": value}
    rmse_standard = float(np.sqrt(np.mean((logits.reshape(-1) - y) ** 2)))
    value = rmse_standard * data.y_scale
    return {"rmse": value, "primary": -value}


def inverse_power(matrix: torch.Tensor, power: float, ridge: float) -> torch.Tensor:
    eigen, vectors = torch.linalg.eigh(matrix.double())
    floor = max(float(eigen[-1].detach().cpu()) * ridge, 1e-14)
    values = eigen.clamp_min(floor).pow(-power)
    return ((vectors * values[None, :]) @ vectors.T).to(matrix.dtype)


def input_inverse(train: np.ndarray, ridge: float = 1e-8) -> torch.Tensor:
    augmented = np.column_stack((train.astype(np.float64), np.ones(len(train))))
    moment = torch.from_numpy(augmented.T @ augmented / len(augmented))
    return inverse_power(moment, 1.0, ridge)


def covariance_initialize(first: nn.Linear, train: np.ndarray, seed: int, ridge: float = 1e-8) -> None:
    covariance = torch.from_numpy(train.astype(np.float64).T @ train.astype(np.float64) / len(train))
    inverse_sqrt = inverse_power(covariance, 0.5, ridge).cpu().numpy()
    rng = np.random.default_rng(seed + 99173)
    gaussian = rng.normal(
        scale=math.sqrt(1.0 / max(3.0 * train.shape[1], 1)), size=first.weight.shape
    )
    with torch.no_grad():
        first.weight.copy_(torch.from_numpy(gaussian @ inverse_sqrt).to(first.weight))
        first.bias.zero_()


class ShampooOptimizer(torch.optim.Optimizer):
    """Full-model Adam-grafted Shampoo for dense parameters.

    Matrix parameters use the two Kronecker factors and inverse fourth roots;
    vectors use AdamW. Grafting preserves the Adam update norm while changing
    its matrix direction, matching the standard practical Shampoo comparison.
    """

    def __init__(
        self,
        parameters,
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        precondition_frequency: int = 10,
        max_precondition_dimension: int = 1024,
    ):
        super().__init__(parameters, dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precondition_dimension=max_precondition_dimension,
        ))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(gradient)
                    state["v"] = torch.zeros_like(gradient)
                    if gradient.ndim == 2 and max(gradient.shape) <= group["max_precondition_dimension"]:
                        state["left"] = torch.zeros(
                            gradient.shape[0], gradient.shape[0], device=gradient.device, dtype=gradient.dtype
                        )
                        state["right"] = torch.zeros(
                            gradient.shape[1], gradient.shape[1], device=gradient.device, dtype=gradient.dtype
                        )
                        state["left_root"] = None
                        state["right_root"] = None
                state["step"] += 1
                beta1, beta2 = group["betas"]
                state["m"].mul_(beta1).add_(gradient, alpha=1 - beta1)
                state["v"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                adam = (state["m"] / (1 - beta1 ** state["step"])) / (
                    (state["v"] / (1 - beta2 ** state["step"])).sqrt() + group["eps"]
                )
                if "left" in state:
                    beta = group["shampoo_beta"]
                    state["left"].lerp_(gradient @ gradient.T, 1 - beta)
                    state["right"].lerp_(gradient.T @ gradient, 1 - beta)
                    if state["left_root"] is None or state["step"] % group["precondition_frequency"] == 0:
                        state["left_root"] = inverse_power(state["left"], 0.25, 1e-8)
                        state["right_root"] = inverse_power(state["right"], 0.25, 1e-8)
                    update = state["left_root"] @ gradient @ state["right_root"]
                    update.mul_(adam.norm() / update.norm().clamp_min(1e-12))
                else:
                    update = adam
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
        return loss


class SOAPOptimizer(torch.optim.Optimizer):
    """Full-model SOAP: Adam in the evolving Shampoo eigenbasis.

    This follows the public ICLR 2025 implementation for matrix parameters,
    including first-step preconditioner initialization, QR eigenbasis refresh,
    and second-moment axis reordering. Vector parameters use AdamW. Algorithmic
    reference: https://github.com/nikhilvyas/SOAP (MIT license).
    """

    def __init__(
        self,
        parameters,
        lr: float = 3e-3,
        betas: tuple[float, float] = (0.95, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 1e-4,
        precondition_frequency: int = 10,
        max_precondition_dimension: int = 1024,
    ):
        super().__init__(parameters, dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precondition_dimension=max_precondition_dimension,
        ))

    @staticmethod
    def _project(value: torch.Tensor, ql: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
        return ql.T @ value @ qr

    @staticmethod
    def _back(value: torch.Tensor, ql: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
        return ql @ value @ qr.T

    @staticmethod
    def _eigenbasis(matrix: torch.Tensor) -> torch.Tensor:
        _, vectors = torch.linalg.eigh(matrix.double())
        return torch.flip(vectors, dims=(1,)).to(matrix.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]
                matrix = gradient.ndim == 2 and max(gradient.shape) <= group["max_precondition_dimension"]
                if not state:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(gradient)
                    state["v"] = torch.zeros_like(gradient)
                    if matrix:
                        state["left"] = torch.zeros(
                            gradient.shape[0], gradient.shape[0], device=gradient.device, dtype=gradient.dtype
                        )
                        state["right"] = torch.zeros(
                            gradient.shape[1], gradient.shape[1], device=gradient.device, dtype=gradient.dtype
                        )
                        state["left"].lerp_(gradient @ gradient.T, 0.05)
                        state["right"].lerp_(gradient.T @ gradient, 0.05)
                        state["ql"] = self._eigenbasis(state["left"])
                        state["qr"] = self._eigenbasis(state["right"])
                        # The public implementation intentionally skips the
                        # first parameter update to avoid using the same
                        # gradient to construct and consume an eigenbasis.
                        continue
                state["step"] += 1
                beta1, beta2 = group["betas"]
                if matrix:
                    projected = self._project(gradient, state["ql"], state["qr"])
                    state["m"].mul_(beta1).add_(projected, alpha=1 - beta1)
                    state["v"].mul_(beta2).addcmul_(projected, projected, value=1 - beta2)
                    correction = math.sqrt(1 - beta2 ** state["step"]) / (1 - beta1 ** state["step"])
                    update = self._back(
                        correction * state["m"] / (state["v"].sqrt() + group["eps"]),
                        state["ql"],
                        state["qr"],
                    )
                else:
                    state["m"].mul_(beta1).add_(gradient, alpha=1 - beta1)
                    state["v"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                    correction = math.sqrt(1 - beta2 ** state["step"]) / (1 - beta1 ** state["step"])
                    update = correction * state["m"] / (state["v"].sqrt() + group["eps"])
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                parameter.add_(update, alpha=-group["lr"])
                if matrix:
                    # Return the first moment to parameter coordinates before
                    # changing the eigenbasis, exactly as in public SOAP.
                    raw_m = self._back(state["m"], state["ql"], state["qr"])
                    state["left"].lerp_(gradient @ gradient.T, 1 - beta2)
                    state["right"].lerp_(gradient.T @ gradient, 1 - beta2)
                    if state["step"] % group["precondition_frequency"] == 0:
                        left_estimate = torch.diag(state["ql"].T @ state["left"] @ state["ql"])
                        right_estimate = torch.diag(state["qr"].T @ state["right"] @ state["qr"])
                        left_order = torch.argsort(left_estimate, descending=True)
                        right_order = torch.argsort(right_estimate, descending=True)
                        state["v"] = state["v"].index_select(0, left_order).index_select(1, right_order)
                        left_power = state["left"] @ state["ql"].index_select(1, left_order)
                        right_power = state["right"] @ state["qr"].index_select(1, right_order)
                        state["ql"], _ = torch.linalg.qr(left_power)
                        state["qr"], _ = torch.linalg.qr(right_power)
                    state["m"] = self._project(raw_m, state["ql"], state["qr"])
        return loss


@dataclass
class FirstUpdateDiagnostics:
    raw_gradient_norm: float = math.nan
    update_norm: float = math.nan
    preconditioner_condition: float = math.nan


class FirstLayerMatrixUpdater:
    """Manual augmented first-affine update; AdamW handles later parameters.

    `input_natural` is the exact right input factor. `first_layer_kfac` adds the
    left preactivation-gradient factor. Shampoo and SOAP are explicitly scoped
    comparisons on the first weight matrix, with the affine bias included in
    the augmented gradient state.
    """

    def __init__(
        self,
        first: nn.Linear,
        train: np.ndarray,
        method: str,
        learning_rate: float,
        ridge: float = 1e-8,
        beta: float = 0.95,
        precondition_frequency: int = 10,
    ):
        self.first = first
        self.method = method
        self.learning_rate = learning_rate
        self.ridge = ridge
        self.beta = beta
        self.frequency = precondition_frequency
        self.step_index = 0
        self.right_inverse = input_inverse(train, ridge)
        self.momentum = None
        self.delta: torch.Tensor | None = None
        self.diagnostics = FirstUpdateDiagnostics(
            preconditioner_condition=float(torch.linalg.cond(self.right_inverse).cpu())
        )
        self.L = None
        self.R = None
        self.m = None
        self.v = None
        self.ql = None
        self.qr = None
        self.hook = first.register_full_backward_hook(self._capture_delta)

    def _capture_delta(self, module, grad_input, grad_output):
        self.delta = grad_output[0].detach()

    def _gradient(self) -> torch.Tensor:
        return torch.cat((self.first.weight.grad, self.first.bias.grad[:, None]), dim=1)

    def _adam_shadow(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.m is None:
            self.m = torch.zeros_like(gradient)
            self.v = torch.zeros_like(gradient)
        self.m.mul_(0.9).add_(gradient, alpha=0.1)
        self.v.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
        correction1 = 1 - 0.9 ** max(self.step_index, 1)
        correction2 = 1 - 0.999 ** max(self.step_index, 1)
        return (self.m / correction1) / ((self.v / correction2).sqrt() + 1e-8)

    def _shampoo(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.L is None:
            self.L = torch.zeros(
                gradient.shape[0], gradient.shape[0], device=gradient.device, dtype=gradient.dtype
            )
            self.R = torch.zeros(
                gradient.shape[1], gradient.shape[1], device=gradient.device, dtype=gradient.dtype
            )
        self.L.lerp_(gradient @ gradient.T, 1 - self.beta)
        self.R.lerp_(gradient.T @ gradient, 1 - self.beta)
        left = inverse_power(self.L, 0.25, self.ridge)
        right = inverse_power(self.R, 0.25, self.ridge)
        update = left @ gradient @ right
        adam = self._adam_shadow(gradient)
        return update * (adam.norm() / update.norm().clamp_min(1e-12))

    def _soap(self, gradient: torch.Tensor) -> torch.Tensor:
        if self.L is None:
            self.L = torch.zeros(
                gradient.shape[0], gradient.shape[0], device=gradient.device, dtype=gradient.dtype
            )
            self.R = torch.zeros(
                gradient.shape[1], gradient.shape[1], device=gradient.device, dtype=gradient.dtype
            )
        self.L.lerp_(gradient @ gradient.T, 1 - self.beta)
        self.R.lerp_(gradient.T @ gradient, 1 - self.beta)
        refresh = self.ql is None or self.step_index % self.frequency == 0
        if refresh:
            old_ql, old_qr = self.ql, self.qr
            _, self.ql = torch.linalg.eigh(self.L.double())
            _, self.qr = torch.linalg.eigh(self.R.double())
            self.ql = torch.flip(self.ql, dims=(1,)).to(gradient.dtype)
            self.qr = torch.flip(self.qr, dims=(1,)).to(gradient.dtype)
            if self.m is not None and old_ql is not None:
                raw_m = old_ql @ self.m @ old_qr.T
                self.m = self.ql.T @ raw_m @ self.qr
                # A diagonal second moment cannot be exactly rotated. SOAP's
                # eigenvalue correction is approximated conservatively by its
                # mean on a full eigensystem refresh.
                self.v.fill_(float(self.v.mean()))
        projected = self.ql.T @ gradient @ self.qr
        if self.m is None:
            self.m = torch.zeros_like(projected)
            self.v = torch.zeros_like(projected)
        self.m.mul_(0.95).add_(projected, alpha=0.05)
        self.v.mul_(0.95).addcmul_(projected, projected, value=0.05)
        correction = math.sqrt(1 - 0.95 ** max(self.step_index, 1)) / (
            1 - 0.95 ** max(self.step_index, 1)
        )
        projected_update = correction * self.m / (self.v.sqrt() + 1e-8)
        return self.ql @ projected_update @ self.qr.T

    @torch.no_grad()
    def step(self, batch_size: int) -> None:
        self.step_index += 1
        gradient = self._gradient()
        right_inverse = self.right_inverse.to(gradient.device, gradient.dtype)
        if self.method == "input_natural":
            update = gradient @ right_inverse
        elif self.method == "first_layer_kfac":
            if self.delta is None:
                raise RuntimeError("K-FAC did not capture the first-layer output gradient")
            delta = self.delta.reshape(-1, self.delta.shape[-1]) * batch_size
            left_moment = delta.T @ delta / max(len(delta), 1)
            left_inverse = inverse_power(left_moment, 1.0, self.ridge)
            update = left_inverse @ gradient @ right_inverse
        elif self.method == "shampoo":
            update = self._shampoo(gradient)
        elif self.method == "soap":
            update = self._soap(gradient)
        else:
            raise ValueError(self.method)
        if self.method in ("input_natural", "first_layer_kfac"):
            if self.momentum is None:
                self.momentum = update.clone()
            else:
                self.momentum.mul_(0.9).add_(update)
            update = self.momentum
        if self.step_index == 1:
            self.diagnostics.raw_gradient_norm = float(gradient.norm().cpu())
            self.diagnostics.update_norm = float(update.norm().cpu())
        self.first.weight.add_(update[:, :-1], alpha=-self.learning_rate)
        self.first.bias.add_(update[:, -1], alpha=-self.learning_rate)

    def close(self) -> None:
        self.hook.remove()
