"""Small compatibility and timing check for the pinned official TACTiS code."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor" / "tactis"))

from tactis.model.tactis import TACTiS  # noqa: E402


def make_model() -> TACTiS:
    temporal_encoder = {
        "attention_layers": 3,
        "attention_heads": 3,
        "attention_dim": 16,
        "attention_feedforward_dim": 16,
        "dropout": 0.0,
    }
    return TACTiS(
        num_series=8,
        flow_series_embedding_dim=5,
        copula_series_embedding_dim=5,
        flow_input_encoder_layers=3,
        copula_input_encoder_layers=3,
        input_encoding_normalization=True,
        data_normalization="standardization",
        loss_normalization="series",
        positional_encoding={"dropout": 0.0},
        flow_temporal_encoder=temporal_encoder.copy(),
        copula_temporal_encoder=temporal_encoder.copy(),
        copula_decoder={
            "min_u": 0.01,
            "max_u": 0.99,
            "attentional_copula": {
                "attention_heads": 3,
                "attention_layers": 3,
                "attention_dim": 16,
                "mlp_layers": 3,
                "mlp_dim": 16,
                "resolution": 50,
            },
            "dsf_marginal": {
                "mlp_layers": 2,
                "mlp_dim": 8,
                "flow_layers": 2,
                "flow_hid_dim": 8,
            },
        },
    )


def count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    model = make_model().to(device)
    history = torch.randn(8, 8, 32, device=device)
    future = torch.randn(8, 8, 4, device=device)
    history_time = torch.arange(32, device=device).expand(8, -1).float()
    future_time = torch.arange(32, 36, device=device).expand(8, -1).float()

    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)
    start = time.perf_counter()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        marginal_logdet, _ = model.loss(history_time, history, future_time, future)
        loss = -marginal_logdet.mean()
        loss.backward()
        optimizer.step()
    stage1_seconds = time.perf_counter() - start
    stage1_parameters = count(model)

    model.initialize_stage2()
    model.to(device)
    copula_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith(
            (
                "copula_series_encoder",
                "copula_time_encoding",
                "copula_input_encoder",
                "copula_encoder",
                "decoder.copula",
            )
        )
    ]
    optimizer = torch.optim.AdamW(copula_parameters, lr=4e-4)
    start = time.perf_counter()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model.loss(history_time, history, future_time, future)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
    stage2_seconds = time.perf_counter() - start
    with torch.no_grad():
        samples = model.sample(16, history_time[:2], history[:2], future_time[:2])

    print(
        {
            "device": str(device),
            "stage1_parameters": stage1_parameters,
            "total_parameters": count(model),
            "stage1_seconds_per_step": stage1_seconds / 20,
            "stage2_seconds_per_step": stage2_seconds / 20,
            "stage2_loss": float(loss),
            "sample_shape": list(samples.shape),
            "samples_finite": bool(torch.isfinite(samples).all()),
        }
    )


if __name__ == "__main__":
    main()
