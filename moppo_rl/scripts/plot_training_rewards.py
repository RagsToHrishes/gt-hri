"""Plot per-objective training curves from training_metrics.jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402


console = Console()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward components over training updates.")
    parser.add_argument(
        "--log-file",
        type=Path,
        required=True,
        help="Path to training_metrics.jsonl generated during training.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path (defaults to <log-file stem>_reward_curve.png).",
    )
    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="Overlay deterministic evaluation returns when available.",
    )
    return parser.parse_args()


def _load_metrics(log_path: Path) -> tuple[list[str], str, np.ndarray, np.ndarray, np.ndarray | None]:
    reward_names: list[str] = []
    env_id = "env"
    updates: list[int] = []
    train_returns: list[list[float]] = []
    eval_returns: list[Optional[list[float]]] = []

    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            record_type = record.get("type")
            if record_type == "metadata":
                reward_names = list(record.get("reward_names", []))
                env_id = record.get("env_id", env_id)
                continue
            if record_type != "metrics":
                continue
            updates.append(int(record["update"]))
            train_returns.append(list(record.get("avg_return", [])))
            eval_returns.append(record.get("eval_avg_return"))

    if not train_returns:
        raise ValueError(f"No training entries were found in {log_path}")

    train_matrix = np.asarray(train_returns, dtype=np.float32)
    eval_matrix: np.ndarray | None = None
    if any(entry is not None for entry in eval_returns):
        eval_matrix = np.full_like(train_matrix, np.nan)
        for idx, entry in enumerate(eval_returns):
            if entry is None:
                continue
            eval_matrix[idx, :] = np.asarray(entry, dtype=np.float32)

    if not reward_names:
        reward_names = [f"component_{i}" for i in range(train_matrix.shape[1])]

    return reward_names, env_id, np.asarray(updates, dtype=np.int32), train_matrix, eval_matrix


def _plot_rewards(
    *,
    reward_names: list[str],
    env_id: str,
    updates: np.ndarray,
    train_matrix: np.ndarray,
    eval_matrix: np.ndarray | None,
    include_eval: bool,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, name in enumerate(reward_names):
        ax.plot(updates, train_matrix[:, idx], label=f"Train · {name}", linewidth=2.0)
        if include_eval and eval_matrix is not None and np.isfinite(eval_matrix[:, idx]).any():
            ax.plot(
                updates,
                eval_matrix[:, idx],
                label=f"Eval · {name}",
                linestyle="--",
                linewidth=1.5,
            )

    ax.set_title(f"Reward components during training ({env_id})")
    ax.set_xlabel("Update")
    ax.set_ylabel("Mean episodic return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    console.print(f"[bold green]Saved reward plot to {output_path}[/bold green]")


def main() -> None:
    args = _parse_args()
    log_file = args.log_file
    if not log_file.exists():
        raise FileNotFoundError(f"Log file not found: {log_file}")
    output_path = args.output
    if output_path is None:
        output_path = log_file.with_name(f"{log_file.stem}_reward_curve.png")
    data = _load_metrics(log_file)
    _plot_rewards(
        reward_names=data[0],
        env_id=data[1],
        updates=data[2],
        train_matrix=data[3],
        eval_matrix=data[4],
        include_eval=args.include_eval,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
