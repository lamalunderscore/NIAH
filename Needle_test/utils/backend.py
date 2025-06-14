"""Implementing the BackEnd class."""

from typing import Callable, Dict, Union

import torch


class BackEnd:
    """Backend monitor for different devices.

    Attributes:
        allocated_memory (Union[Dict[str, float], float]): The currently allocated memory.
        device (str): The device that is used.

    """

    def __init__(self):
        if torch.cuda.is_available():
            self._allocated_memory = lambda: {
                str(i): torch.cuda.memory_allocated(torch.device(f"cuda:{i}")) / 1024**2
                for i in range(torch.cuda.device_count())
            }
            self.empty_cache: Callable[[], None] = torch.cuda.empty_cache
            self._device = "cuda"
        elif torch.backends.mps.is_available():
            self._allocated_memory = torch.mps.current_allocated_memory() / 1024**2
            self.empty_cache: Callable[[], None] = torch.mps.empty_cache
            self._device = "mps"
        else:
            print("Warning, CPU monitoring not implemented.")
            self._allocated_memory = lambda: 0
            self.empty_cache = lambda: None
            self._device = "cpu"

    @property
    def allocated_memory(self) -> Union[Dict[str, float], float]:
        return self._allocated_memory()

    @property
    def device(self) -> str:
        return self._device
