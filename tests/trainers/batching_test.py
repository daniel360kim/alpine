"""
Tests for on-device batching utilities:
  - OnDeviceSignalLoader convergence vs DataLoader(BatchedNDSignalLoader)
  - Timing comparison between the two approaches
  - Factory routing logic of make_batched_signal_loader
"""

import time
import copy
from unittest.mock import patch

import torch
import numpy as np

from alpine.models.siren import Siren
from alpine.dataloaders.signal_dataloaders import (
    BatchedNDSignalLoader,
    OnDeviceSignalLoader,
    make_batched_signal_loader,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_signal(grid_dims=(64, 64), channels=1, seed=42):
    """Return a reproducible random signal as a numpy array."""
    rng = np.random.RandomState(seed)
    shape = tuple(grid_dims) + (channels,)
    return rng.rand(*shape).astype(np.float32)


def _train_loop(model, loader, device, n_iters=200):
    """Minimal training loop that mirrors alpine_base._fit_signal_dataloader."""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    final_loss = None
    for _ in range(n_iters):
        for batch in loader:
            coords = batch["input"].to(device)
            signal = batch["signal"].to(device)
            optimizer.zero_grad()
            out = model(coords)
            loss = loss_fn(out["output"], signal)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
    return final_loss


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_on_device_convergence():
    """Train identical SIRENs with OnDeviceSignalLoader and DataLoader(BatchedNDSignalLoader).
    Assert final losses are within 5e-3 of each other."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_dims = (64, 64)
    batch_size = 256
    n_iters = 200
    signal_np = _make_test_signal(grid_dims=grid_dims, channels=1)

    # Build two models with identical weights
    model_a = Siren(in_features=2, hidden_features=64, hidden_layers=2, out_features=1).to(device)
    model_b = copy.deepcopy(model_a)

    # Loader A: OnDeviceSignalLoader
    loader_a = OnDeviceSignalLoader(
        signal_np, grid_dims, batch_size, device, bounds=(-1, 1),
        normalize_signal=True, normalize_fn=None,
    )

    # Loader B: DataLoader(BatchedNDSignalLoader)
    dataset_b = BatchedNDSignalLoader(
        signal_np, grid_dims, bounds=(-1, 1), vectorized=True,
        normalize_signal=True, normalize_fn=None,
    )
    loader_b = torch.utils.data.DataLoader(
        dataset_b, batch_size=batch_size, shuffle=True, pin_memory=False,
    )

    loss_a = _train_loop(model_a, loader_a, device, n_iters)
    loss_b = _train_loop(model_b, loader_b, device, n_iters)

    diff = abs(loss_a - loss_b)
    print(f"[convergence] OnDevice loss={loss_a:.6f}  DataLoader loss={loss_b:.6f}  diff={diff:.6f}")
    assert diff < 5e-3, f"Loss difference {diff} exceeds 5e-3 threshold"
    print("[convergence] PASSED")


def test_on_device_timing():
    """Benchmark OnDeviceSignalLoader vs DataLoader(BatchedNDSignalLoader).
    Informational only — no assertion."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid_dims = (64, 64)
    batch_size = 256
    n_iters = 200
    signal_np = _make_test_signal(grid_dims=grid_dims, channels=1)

    model_a = Siren(in_features=2, hidden_features=64, hidden_layers=2, out_features=1).to(device)
    model_b = copy.deepcopy(model_a)

    # --- OnDeviceSignalLoader ---
    loader_a = OnDeviceSignalLoader(
        signal_np, grid_dims, batch_size, device, bounds=(-1, 1),
        normalize_signal=True, normalize_fn=None,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss_a = _train_loop(model_a, loader_a, device, n_iters)
    if device.type == "cuda":
        torch.cuda.synchronize()
    time_a = time.perf_counter() - t0

    # --- DataLoader(BatchedNDSignalLoader) ---
    dataset_b = BatchedNDSignalLoader(
        signal_np, grid_dims, bounds=(-1, 1), vectorized=True,
        normalize_signal=True, normalize_fn=None,
    )
    loader_b = torch.utils.data.DataLoader(
        dataset_b, batch_size=batch_size, shuffle=True, pin_memory=False,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    loss_b = _train_loop(model_b, loader_b, device, n_iters)
    if device.type == "cuda":
        torch.cuda.synchronize()
    time_b = time.perf_counter() - t0

    avg_a = time_a / n_iters * 1000  # ms
    avg_b = time_b / n_iters * 1000  # ms

    print()
    print(f"{'Metric':<25} {'OnDevice':>12} {'DataLoader':>12}")
    print("-" * 51)
    print(f"{'Total time (s)':<25} {time_a:>12.4f} {time_b:>12.4f}")
    print(f"{'Avg iter (ms)':<25} {avg_a:>12.4f} {avg_b:>12.4f}")
    print(f"{'Final loss':<25} {loss_a:>12.6f} {loss_b:>12.6f}")
    print()


def test_factory_routing():
    """Verify make_batched_signal_loader routes to the correct loader type."""

    grid_dims = (32, 32)
    signal_np = _make_test_signal(grid_dims=grid_dims, channels=1)
    batch_size = 128

    # Test 1: force_cpu=True → DataLoader
    result = make_batched_signal_loader(
        signal_np, grid_dims, batch_size, device="cpu", force_cpu=True,
    )
    assert isinstance(result, torch.utils.data.DataLoader), (
        f"Expected DataLoader with force_cpu=True, got {type(result)}"
    )
    print("[factory] force_cpu=True -> DataLoader  PASSED")

    # Test 2: device=cpu → DataLoader
    result = make_batched_signal_loader(
        signal_np, grid_dims, batch_size, device=torch.device("cpu"),
    )
    assert isinstance(result, torch.utils.data.DataLoader), (
        f"Expected DataLoader with CPU device, got {type(result)}"
    )
    print("[factory] device=cpu -> DataLoader  PASSED")

    # Test 3: CUDA with enough memory → OnDeviceSignalLoader
    if torch.cuda.is_available():
        result = make_batched_signal_loader(
            signal_np, grid_dims, batch_size, device=torch.device("cuda"),
        )
        assert isinstance(result, OnDeviceSignalLoader), (
            f"Expected OnDeviceSignalLoader on CUDA, got {type(result)}"
        )
        print("[factory] device=cuda (enough mem) -> OnDeviceSignalLoader  PASSED")
    else:
        print("[factory] CUDA not available — skipping on-device routing test")

    # Test 4: Patch mem_get_info to return 0 free memory → RuntimeError
    with patch("alpine.dataloaders.signal_dataloaders.torch.cuda.mem_get_info", return_value=(0, 0)):
        try:
            make_batched_signal_loader(
                signal_np, grid_dims, batch_size, device=torch.device("cuda"),
            )
            assert False, "Expected RuntimeError but no exception was raised"
        except RuntimeError as e:
            assert "force_cpu=True" in str(e), (
                f"Error message should contain 'force_cpu=True', got: {e}"
            )
            print(f"[factory] OOM -> RuntimeError with force_cpu=True hint  PASSED")

    print("[factory] ALL PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    test_on_device_convergence()
    test_on_device_timing()
    test_factory_routing()
    print("\nAll batching tests passed.")


if __name__ == "__main__":
    main()
