"""Generate checkpoint overview figures that combine Pareto, reward, and hypervolume plots."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from rich.console import Console  # noqa: E402


console = Console()
DEFAULT_CHECKPOINT_ROOT = Path(__file__).resolve().parents[1] / "checkpoints"


@dataclass
class TrainingCurves:
    reward_names: list[str]
    env_id: str
    updates: np.ndarray
    returns: np.ndarray


@dataclass
class ParetoSweep:
    updates: np.ndarray
    hypervolumes: np.ndarray
    returns: list[np.ndarray]
    return_updates: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create combined overview figures for each checkpoint.")
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Directory containing checkpoint folders (defaults to the package checkpoints folder).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/checkpoint_overviews"),
        help="Destination directory for the generated PNG files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Resolution for the saved figures.",
    )
    return parser.parse_args()


def load_training_curves(checkpoint_dir: Path) -> TrainingCurves | None:
    log_path = checkpoint_dir / "training_metrics.jsonl"
    if not log_path.exists():
        console.print(f"[yellow]Skipping reward curves for {checkpoint_dir.name}: missing training_metrics.jsonl[/yellow]")
        return None

    reward_names: list[str] = []
    env_id = checkpoint_dir.name
    updates: list[int] = []
    avg_returns: list[list[float]] = []

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
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
            avg_returns.append(list(record.get("avg_return", [])))

    if not avg_returns:
        console.print(f"[yellow]No training entries found in {log_path}[/yellow]")
        return None

    return TrainingCurves(
        reward_names=reward_names,
        env_id=env_id,
        updates=np.asarray(updates, dtype=np.int32),
        returns=np.asarray(avg_returns, dtype=np.float32),
    )


def load_pareto_sweeps(pareto_dir: Path) -> ParetoSweep | None:
    if not pareto_dir.exists():
        console.print(f"[yellow]Missing pareto_eval directory: {pareto_dir}[/yellow]")
        return None

    update_ids: list[int] = []
    hypervolumes: list[float] = []
    returns: list[np.ndarray] = []
    return_updates: list[int] = []

    json_files = sorted(pareto_dir.glob("update_*_pareto.json"))
    if not json_files:
        console.print(f"[yellow]No Pareto JSON files found under {pareto_dir}[/yellow]")
        return None

    for json_file in json_files:
        with json_file.open("r", encoding="utf-8") as handle:
            record = json.load(handle)
        update = int(record.get("update", 0))
        hv = record.get("hypervolume")
        hypervolumes.append(float(hv) if hv is not None else np.nan)
        update_ids.append(update)
        sweep_returns = record.get("pareto_front_returns", []) or []
        for entry in sweep_returns:
            returns.append(np.asarray(entry, dtype=np.float32))
            return_updates.append(update)

    if not returns:
        console.print(f"[yellow]Pareto files under {pareto_dir} had no front entries[/yellow]")

    return ParetoSweep(
        updates=np.asarray(update_ids, dtype=np.int32),
        hypervolumes=np.asarray(hypervolumes, dtype=np.float32),
        returns=returns,
        return_updates=return_updates,
    )


def ensure_reward_names(existing: list[str], num_components: int) -> list[str]:
    if existing:
        return existing
    return [f"reward_{idx+1}" for idx in range(num_components)]


def plot_training_curves(ax: plt.Axes, curves: TrainingCurves | None) -> None:
    if curves is None:
        ax.text(0.5, 0.5, "No training_metrics data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    reward_names = ensure_reward_names(curves.reward_names, curves.returns.shape[1])
    for idx, name in enumerate(reward_names):
        ax.plot(curves.updates, curves.returns[:, idx], label=name, linewidth=1.6)
    ax.set_title(f"Training reward components ({curves.env_id})")
    ax.set_xlabel("Update")
    ax.set_ylabel("Mean episodic return")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def plot_hypervolume(ax: plt.Axes, sweeps: ParetoSweep | None) -> None:
    if sweeps is None or sweeps.updates.size == 0:
        ax.text(0.5, 0.5, "No hypervolume data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return
    ax.plot(sweeps.updates, sweeps.hypervolumes, marker="o", linewidth=1.8)
    ax.set_title("Hypervolume over updates")
    ax.set_xlabel("Update")
    ax.set_ylabel("Hypervolume")
    ax.grid(True, alpha=0.3)


def plot_radar_overlay(ax: plt.Axes, sweeps: ParetoSweep | None, reward_names: list[str]) -> None:
    if sweeps is None or not sweeps.returns:
        ax.text(0.5, 0.5, "No Pareto front data", ha="center", va="center", fontsize=11)
        ax.set_axis_off()
        return

    reward_dim = len(reward_names)
    if reward_dim == 0:
        reward_dim = len(sweeps.returns[0])
        reward_names = ensure_reward_names([], reward_dim)

    theta = np.linspace(0, 2 * np.pi, reward_dim, endpoint=False).tolist()
    theta += theta[:1]

    updates = np.asarray(sweeps.return_updates, dtype=np.int32)
    cmap = plt.get_cmap("plasma")
    norm = plt.Normalize(vmin=int(np.min(updates)), vmax=int(np.max(updates)))

    for values, update in zip(sweeps.returns, updates):
        if values.size != reward_dim:
            continue
        closed = np.concatenate([values, values[:1]])
        ax.plot(
            theta,
            closed,
            color=cmap(norm(update)),
            alpha=0.5,
            linewidth=1.1,
        )

    ax.set_xticks(theta[:-1])
    ax.set_xticklabels(reward_names, fontsize=9)
    ax.set_title("Pareto spider overlays")
    ax.grid(True, alpha=0.3)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    colorbar = plt.colorbar(sm, ax=ax, pad=0.1)
    colorbar.set_label("Pareto update", rotation=270, labelpad=14)


def iter_checkpoint_dirs(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Checkpoint root not found: {root}")
    for path in sorted(root.iterdir()):
        if path.is_dir():
            yield path


def build_overview_fig(
    *,
    checkpoint_dir: Path,
    curves: TrainingCurves | None,
    sweeps: ParetoSweep | None,
    output_path: Path,
    dpi: int,
) -> None:
    reward_dim = 0
    reward_names: list[str] = []
    if curves is not None:
        reward_dim = curves.returns.shape[1]
        reward_names = ensure_reward_names(curves.reward_names, reward_dim)
    elif sweeps is not None and sweeps.returns:
        reward_dim = sweeps.returns[0].size
        reward_names = ensure_reward_names([], reward_dim)

    fig = plt.figure(figsize=(18, 6), dpi=dpi)
    grid = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0])
    ax_radar = fig.add_subplot(grid[0, 0], projection="polar")
    ax_rewards = fig.add_subplot(grid[0, 1])
    ax_hv = fig.add_subplot(grid[0, 2])

    plot_radar_overlay(ax_radar, sweeps, reward_names)
    plot_training_curves(ax_rewards, curves)
    plot_hypervolume(ax_hv, sweeps)

    fig.suptitle(f"Checkpoint: {checkpoint_dir.name}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    console.print(f"[bold green]Saved overview to {output_path}[/bold green]")


def main() -> None:
    args = parse_args()
    root = args.checkpoints_root
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    processed = False
    for checkpoint_dir in iter_checkpoint_dirs(root):
        curves = load_training_curves(checkpoint_dir)
        sweeps = load_pareto_sweeps(checkpoint_dir / "pareto_eval")
        build_overview_fig(
            checkpoint_dir=checkpoint_dir,
            curves=curves,
            sweeps=sweeps,
            output_path=output_dir / f"{checkpoint_dir.name}_overview.png",
            dpi=args.dpi,
        )
        processed = True

    if not processed:
        console.print(f"[red]No checkpoint folders were found under {root}[/red]")


if __name__ == "__main__":
    main()
