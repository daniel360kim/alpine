import math

import torch
import numpy as np
from typing import Union


class BatchedNDSignalLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        signal: Union[np.ndarray, torch.Tensor],
        grid_dims: tuple,
        bounds: tuple = (-1, 1),
        vectorized: bool = True,
        normalize_signal: bool = True,
        normalize_fn: callable = None,
    ):
        """_summary_

        Args:
            signal (Any): Indexible object containing the signal data. Can be numpy array, list, torch.Tensor, etc. Must be of shape grid_dims[0] x grid_dims[1] x ... x grid_dims[n] x (optional channels).
            grid_dims (tuple): _description_
            bounds (tuple, optional): _description_. Defaults to (-1, 1).
            vectorized (bool, optional): _description_. Defaults to True.
            normalize_signal (bool, optional): Min max normalization of the signal. Defaults to True.
            normalize_fn (callable, optional): custom callable function to normalize signal. Function will accept the signal as an argument. Defaults to None.
        """

        super(BatchedNDSignalLoader).__init__()
        self.grid_dims = grid_dims
        self.bounds = bounds
        self.vectorized = vectorized
        self.normalize_signal = normalize_signal
        self.normalize_fn = normalize_fn
        self.signal = self.setup_signal(signal)
        self.grid_tensor = self.build_coordinate_tensors()

    def setup_signal(self, signal):
        """
        Sets up the signal for the dataset. This includes reshaping, normalizing, and converting to a tensor if necessary. Used internally by BatchedNDSignalLoader.
        """
        assert np.prod(signal.shape[: len(self.grid_dims)]) == np.prod(
            self.grid_dims
        ), f"Signal shape {signal.shape} does not match grid dimensions {self.grid_dims}."
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)
        elif isinstance(signal, list):
            signal = torch.tensor(signal)
        elif not isinstance(signal, torch.Tensor):
            raise TypeError("Signal must be a numpy array, list, or torch.Tensor.")

        if self.normalize_signal:
            if self.normalize_fn is None:
                signal = (signal - signal.min()) / (signal.max() - signal.min())
            else:
                signal = self.normalize_fn(signal)

        if self.vectorized:
            signal = signal.reshape(np.prod(self.grid_dims), -1)

        return signal

    def build_coordinate_tensors(self):
        """Builds coordinate tensors based on the specified grid dimensions. Used internally by BatchedCoordinateDataset."""

        grid_axis = (
            torch.linspace(self.bounds[0], self.bounds[1], self.grid_dims[i])
            for i in range(len(self.grid_dims))
        )
        grid_meshgrids = torch.meshgrid(*grid_axis, indexing="ij")
        grid_tensor = torch.stack(grid_meshgrids, dim=-1)
        if self.vectorized:
            grid_tensor = grid_tensor.reshape(-1, len(self.grid_dims))
        return grid_tensor.float()

    def __len__(self):
        return np.prod(self.grid_dims)

    def __getitem__(self, idx):
        """Returns a batch of signal data based on the specified index.

        Args:
            idx (int): Index of the batch to be returned.

        Returns:
            torch.Tensor: Batch of signal data.
        """
        idx = (
            torch.unravel_index(torch.tensor(idx), self.grid_dims)
            if not self.vectorized
            else idx
        )
        coords = self.grid_tensor[idx]
        signal = self.signal[idx]

        return {"input": coords.float(), "signal": signal.float()}


class NDSignalLoader(torch.utils.data.Dataset):
    def __init__(
        self,
        signal: Union[np.ndarray, torch.Tensor],
        grid_dims: tuple,
        bounds: tuple = (-1, 1),
        vectorized: bool = True,
        normalize_signal: bool = True,
        normalize_fn: callable = None,
    ):
        """_summary_

        Args:
            signal (Any): Indexible object containing the signal data. Can be numpy array, list, torch.Tensor, etc.
            grid_dims (tuple): _description_
            bounds (tuple, optional): _description_. Defaults to (-1, 1).
            vectorized (bool, optional): _description_. Defaults to True.
            normalize_signal (bool, optional): Min max normalization of the signal. Defaults to True.
            normalize_fn (callable, optional): custom callable function to normalize signal. Function will accept the signal as an argument. Defaults to None.
        """

        super(NDSignalLoader).__init__()
        self.grid_dims = grid_dims
        self.bounds = bounds
        self.vectorized = vectorized
        self.normalize_signal = normalize_signal
        self.normalize_fn = normalize_fn
        self.signal = self.setup_signal(signal)
        self.grid_tensor = self.build_coordinate_tensors()

    def setup_signal(self, signal):
        """
        Sets up the signal for the dataset. This includes reshaping, normalizing, and converting to a tensor if necessary. Used internally by NDSignalLoader.
        """
        assert np.prod(signal.shape[: len(self.grid_dims)]) == np.prod(
            self.grid_dims
        ), f"Signal shape {signal.shape} does not match grid dimensions {self.grid_dims}."
        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)
        elif isinstance(signal, list):
            signal = torch.tensor(signal)
        elif not isinstance(signal, torch.Tensor):
            raise TypeError("Signal must be a numpy array, list, or torch.Tensor.")

        if self.normalize_signal:
            if self.normalize_fn is None:
                signal = (signal - signal.min()) / (signal.max() - signal.min())
            else:
                signal = self.normalize_fn(signal)

        if self.vectorized:
            signal = signal.reshape(np.prod(self.grid_dims), -1)

        return signal

    def build_coordinate_tensors(self):
        """Builds coordinate tensors based on the specified grid dimensions. Used internally by BatchedCoordinateDataset."""

        grid_axis = [
            torch.linspace(self.bounds[0], self.bounds[1], self.grid_dims[i])
            for i in range(len(self.grid_dims))
        ]

        grid_meshgrids = torch.meshgrid(*grid_axis, indexing="ij")
        grid_tensor = torch.stack(grid_meshgrids, dim=-1)
        if self.vectorized:
            grid_tensor = grid_tensor.reshape(-1, len(self.grid_dims))
        return grid_tensor.float()

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        """Returns a batch of signal data based on the specified index.

        Args:
            idx (int): Index of the batch to be returned.

        Returns:
            torch.Tensor: Batch of signal data.
        """
        coords = self.grid_tensor
        signal = self.signal

        return {"input": coords.float(), "signal": signal.float()}


class OnDeviceSignalLoader:
    """On-device signal loader that pre-pins the entire signal and coordinate
    tensors on the target device once, avoiding repeated CPU→GPU transfers
    during training.

    Unlike :class:`BatchedNDSignalLoader`, this class is **not** a
    ``torch.utils.data.Dataset`` and does not need to be wrapped in a
    ``torch.utils.data.DataLoader``.  It is directly iterable and yields
    ``{"input": ..., "signal": ...}`` dicts with tensors already resident on
    the target device.

    Args:
        signal (Union[np.ndarray, torch.Tensor]): Signal data of shape
            ``grid_dims[0] x ... x grid_dims[n] x (optional channels)``.
        grid_dims (tuple): Spatial dimensions of the signal grid.
        batch_size (int): Number of samples per batch.
        device: Target ``torch.device`` (e.g. ``"cuda:0"``).
        bounds (tuple, optional): Coordinate range. Defaults to ``(-1, 1)``.
        normalize_signal (bool, optional): Whether to min-max normalize the
            signal. Defaults to ``True``.
        normalize_fn (callable, optional): Custom normalization function that
            accepts and returns the signal tensor. Defaults to ``None``.
    """

    def __init__(
        self,
        signal: Union[np.ndarray, torch.Tensor],
        grid_dims: tuple,
        batch_size: int,
        device,
        bounds: tuple = (-1, 1),
        normalize_signal: bool = True,
        normalize_fn: callable = None,
    ):
        self.grid_dims = grid_dims
        self.bounds = bounds
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.normalize_signal = normalize_signal
        self.normalize_fn = normalize_fn

        self.signal = self._setup_signal(signal)
        self.coords = self._build_coordinate_tensors()
        self.N = self.coords.shape[0]

        # Move both tensors to the target device once
        self.coords = self.coords.to(self.device)
        self.signal = self.signal.to(self.device)

    def _setup_signal(self, signal):
        """Sets up the signal: validates shape, converts to tensor, normalizes,
        and reshapes to ``(N, C)``."""
        assert np.prod(signal.shape[: len(self.grid_dims)]) == np.prod(
            self.grid_dims
        ), f"Signal shape {signal.shape} does not match grid dimensions {self.grid_dims}."

        if isinstance(signal, np.ndarray):
            signal = torch.from_numpy(signal)
        elif isinstance(signal, list):
            signal = torch.tensor(signal)
        elif not isinstance(signal, torch.Tensor):
            raise TypeError("Signal must be a numpy array, list, or torch.Tensor.")

        if self.normalize_signal:
            if self.normalize_fn is None:
                signal = (signal - signal.min()) / (signal.max() - signal.min())
            else:
                signal = self.normalize_fn(signal)

        signal = signal.reshape(np.prod(self.grid_dims), -1)
        return signal.float()

    def _build_coordinate_tensors(self):
        """Builds a flat coordinate tensor of shape ``(N, D)``."""
        grid_axis = [
            torch.linspace(self.bounds[0], self.bounds[1], self.grid_dims[i])
            for i in range(len(self.grid_dims))
        ]
        grid_meshgrids = torch.meshgrid(*grid_axis, indexing="ij")
        grid_tensor = torch.stack(grid_meshgrids, dim=-1)
        grid_tensor = grid_tensor.reshape(-1, len(self.grid_dims))
        return grid_tensor.float()

    def __len__(self) -> int:
        return math.ceil(self.N / self.batch_size)

    def __iter__(self):
        perm = torch.randperm(self.N, device=self.device)
        for start in range(0, self.N, self.batch_size):
            idx = perm[start : start + self.batch_size]
            yield {"input": self.coords[idx], "signal": self.signal[idx]}

    @staticmethod
    def estimate_memory_bytes(
        signal: Union[np.ndarray, torch.Tensor], grid_dims: tuple
    ) -> int:
        """Estimates the total GPU memory (in bytes, float32) required to hold
        both the coordinate tensor ``(N, D)`` and the signal tensor ``(N, C)``
        on device.

        Args:
            signal (Union[np.ndarray, torch.Tensor]): The signal array/tensor.
            grid_dims (tuple): Spatial grid dimensions.

        Returns:
            int: Estimated memory in bytes.
        """
        N = int(np.prod(grid_dims))
        D = len(grid_dims)
        C = int(np.prod(signal.shape[len(grid_dims) :]))
        return N * (D + C) * 4


def make_batched_signal_loader(
    signal: Union[np.ndarray, torch.Tensor],
    grid_dims: tuple,
    batch_size: int,
    device,
    bounds: tuple = (-1, 1),
    normalize_signal: bool = True,
    normalize_fn: callable = None,
    force_cpu: bool = False,
    memory_safety_factor: float = 0.8,
):
    """Factory that returns the best available batched loader for the given
    signal and device.

    When the target device is CUDA and enough GPU memory is available, an
    :class:`OnDeviceSignalLoader` is returned so that all batching happens
    entirely on-device.  Otherwise, a standard
    ``torch.utils.data.DataLoader(BatchedNDSignalLoader(...))`` is returned.

    Args:
        signal (Union[np.ndarray, torch.Tensor]): Signal data.
        grid_dims (tuple): Spatial grid dimensions.
        batch_size (int): Samples per batch.
        device: Target ``torch.device``.
        bounds (tuple, optional): Coordinate range. Defaults to ``(-1, 1)``.
        normalize_signal (bool, optional): Min-max normalize. Defaults to ``True``.
        normalize_fn (callable, optional): Custom normalization. Defaults to ``None``.
        force_cpu (bool, optional): Force CPU-based ``DataLoader`` path.
            Defaults to ``False``.
        memory_safety_factor (float, optional): Fraction of free GPU memory
            considered usable. Defaults to ``0.8``.

    Returns:
        Union[OnDeviceSignalLoader, torch.utils.data.DataLoader]: A batched
        loader yielding ``{"input": ..., "signal": ...}`` dicts.

    Raises:
        RuntimeError: If the signal does not fit in GPU memory and
            ``force_cpu`` is ``False``.
    """
    device = torch.device(device)

    if force_cpu or device.type == "cpu":
        dataset = BatchedNDSignalLoader(
            signal,
            grid_dims,
            bounds=bounds,
            vectorized=True,
            normalize_signal=normalize_signal,
            normalize_fn=normalize_fn,
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, pin_memory=False
        )

    # CUDA path
    required = OnDeviceSignalLoader.estimate_memory_bytes(signal, grid_dims)
    available = torch.cuda.mem_get_info(device)[0] * memory_safety_factor

    if required <= available:
        return OnDeviceSignalLoader(
            signal,
            grid_dims,
            batch_size,
            device,
            bounds=bounds,
            normalize_signal=normalize_signal,
            normalize_fn=normalize_fn,
        )

    raise RuntimeError(
        f"Signal requires {required} bytes but only {int(available)} bytes "
        f"({memory_safety_factor:.0%} of free GPU memory) are available. "
        f"Pass force_cpu=True to fall back to CPU-based DataLoader batching."
    )