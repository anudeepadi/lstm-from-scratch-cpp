"""Plot actual printed checkpoints; does not train or modify the data.

Usage: python3 docs/plot_learning_curve.py
Optional plotting dependency: matplotlib==3.10.8
"""

from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

directory = Path(__file__).resolve().parent
log = (directory / "training-run.txt").read_text()
points = [(int(epoch), float(loss)) for epoch, loss in re.findall(
    r"Epoch (\d+)/\d+, Loss: ([\d.eE+-]+)", log
)]
if not points:
    raise ValueError("No epoch checkpoints found in training-run.txt")
epochs, losses = zip(*points)
fig, ax = plt.subplots(figsize=(8.5, 4.4), layout="constrained")
ax.plot(epochs, losses, "o-", color="#23695f", linewidth=2)
ax.set(xlabel="Epoch", ylabel="Mean online training loss (squared error)",
       title="LSTM from Scratch in C++ — one observed run")
ax.set_xticks(epochs)
ax.grid(alpha=0.2)
ax.spines[["top", "right"]].set_visible(False)
fig.text(0.5, -0.025,
         "Synthetic sine data • random initialization • training checkpoints only, no held-out evaluation",
         ha="center", fontsize=9, color="#555555")
fig.savefig(directory / "learning-curve.png", dpi=160, bbox_inches="tight")
plt.close(fig)
