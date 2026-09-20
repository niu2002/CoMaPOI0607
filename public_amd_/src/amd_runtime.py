"""Small, standalone helpers for a PyTorch ROCm/CPU demonstration."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
import platform
import time
from typing import Any, Callable, ContextManager

import torch


@dataclass(frozen=True)
class RuntimeInfo:
    """Runtime facts collected without assuming a GPU is installed."""

    python_version: str
    pytorch_version: str
    hip_version: str | None
    accelerator_available: bool
    device: str
    device_name: str | None
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def choose_device() -> torch.device:
    """Select PyTorch's CUDA-compatible accelerator, or safely fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def collect_runtime_info() -> RuntimeInfo:
    """Collect AMD/ROCm-relevant information and return friendly fallback status."""
    accelerator_available = torch.cuda.is_available()
    hip_version = getattr(torch.version, "hip", None)
    device = choose_device()
    device_name: str | None = None

    if accelerator_available:
        try:
            device_name = torch.cuda.get_device_name(device)
        except (AssertionError, RuntimeError) as exc:
            device_name = f"Unavailable ({exc.__class__.__name__})"

    if accelerator_available and hip_version:
        note = "ROCm/HIP runtime detected. PyTorch exposes it through torch.cuda for compatibility."
    elif accelerator_available:
        note = "A CUDA-compatible PyTorch accelerator is available; HIP runtime was not reported."
    else:
        note = "No PyTorch accelerator is available. The notebook will use CPU fallback."

    return RuntimeInfo(
        python_version=platform.python_version(),
        pytorch_version=torch.__version__,
        hip_version=hip_version,
        accelerator_available=accelerator_available,
        device=str(device),
        device_name=device_name,
        note=note,
    )


def synchronize_if_needed(device: torch.device) -> None:
    """Synchronize accelerator work so wall-clock measurements are meaningful."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory_if_needed(device: torch.device) -> None:
    """Reset peak allocation counters when the selected device supports them."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def memory_stats_mib(device: torch.device) -> dict[str, float] | None:
    """Return allocated/reserved/peak memory in MiB, or None on CPU."""
    if device.type != "cuda":
        return None
    scale = 1024 * 1024
    return {
        "allocated_mib": torch.cuda.memory_allocated(device) / scale,
        "reserved_mib": torch.cuda.memory_reserved(device) / scale,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / scale,
    }


def autocast_context(device: torch.device, enabled: bool = True) -> ContextManager[Any]:
    """Use float16 autocast on CUDA/ROCm and a no-op context elsewhere."""
    if device.type == "cuda" and enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def benchmark(
    step: Callable[[], None], device: torch.device, warmup_steps: int = 5, measured_steps: int = 20
) -> dict[str, float]:
    """Benchmark a supplied step after warmup; never fabricates measurements."""
    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and measured_steps must be positive.")

    for _ in range(warmup_steps):
        step()
    synchronize_if_needed(device)

    elapsed_seconds: list[float] = []
    for _ in range(measured_steps):
        synchronize_if_needed(device)
        start = time.perf_counter()
        step()
        synchronize_if_needed(device)
        elapsed_seconds.append(time.perf_counter() - start)

    average = sum(elapsed_seconds) / len(elapsed_seconds)
    return {
        "mean_latency_ms": average * 1000.0,
        "min_latency_ms": min(elapsed_seconds) * 1000.0,
        "max_latency_ms": max(elapsed_seconds) * 1000.0,
        "measured_steps": float(measured_steps),
    }

