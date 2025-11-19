"""Utilities for selecting the most appropriate accelerator device."""

from __future__ import annotations

from typing import Optional

import torch
from rich.console import Console


_AUTO_CHOICES = {"auto", "cuda_if_available"}


def resolve_device(preferred: Optional[str], *, console: Optional[Console] = None) -> str:
    """Resolve a user-provided device preference into a concrete torch device string."""

    def _log(message: str) -> None:
        if console is not None:
            console.print(message)

    if preferred is None or preferred.strip() == "":
        normalized = "auto"
    else:
        normalized = preferred.strip().lower()

    if normalized in _AUTO_CHOICES:
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            _log("[bold green]CUDA detected; running on GPU.[/bold green]")
            return "cuda"
        _log("[yellow]CUDA unavailable; falling back to CPU.[/yellow]")
        return "cpu"

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            _log(
                f"[yellow]Requested CUDA device '{preferred}' but CUDA is not available; "
                "using CPU instead.[/yellow]"
            )
            return "cpu"
        if ":" in normalized:
            suffix = normalized.split(":", 1)[1]
            if suffix.isdigit():
                index = int(suffix)
                if index >= torch.cuda.device_count():
                    _log(
                        f"[yellow]CUDA device index {index} unavailable; defaulting to cuda:0.[/yellow]"
                    )
                    return "cuda:0"
        return preferred

    return preferred if preferred is not None and preferred.strip() else "cpu"


__all__ = ["resolve_device"]
