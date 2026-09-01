#!/usr/bin/env python3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> None:
    ridge = pd.read_csv(RESULTS / "ridge_screen.csv")
    controls = pd.read_csv(RESULTS / "basis_controls.csv")
    frame = pd.concat([ridge, controls])
    agg = frame.groupby(["domain", "seed", "method"]).test_mse.mean().unstack("method")
    methods = ["ple", "u_ple", "periodic", "code_rbf", "mpe_corrupt", "mpe_native", "mmpe_native"]
    domains = ["interval", "cycle", "tree", "nominal"]
    values = np.zeros((len(domains), len(methods)))
    for i, domain in enumerate(domains):
        x = agg.loc[domain]
        values[i] = [100 * ((x.ple - x[m]) / x.ple).mean() for m in methods]
    clipped = np.clip(values, -120, 120)
    fig, ax = plt.subplots(figsize=(10.2, 4.4))
    image = ax.imshow(clipped, cmap="RdYlGn", vmin=-100, vmax=100, aspect="auto")
    for i in range(len(domains)):
        for j in range(len(methods)):
            ax.text(j, i, f"{values[i,j]:+.1f}%", ha="center", va="center", fontsize=8,
                    color="white" if abs(clipped[i,j]) > 65 else "black")
    ax.set_xticks(range(len(methods)), [m.replace("_", "\n") for m in methods])
    ax.set_yticks(range(len(domains)), domains)
    ax.set_title("Mean ridge-test MSE reduction versus quantile PLE (12 seeds; 8 schemas)")
    fig.colorbar(image, ax=ax, label="gain vs Q-PLE (%)", shrink=0.85)
    fig.tight_layout()
    output = RESULTS / "figures/method_screen.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
