import torch

from typing import Any
from abc import ABC, abstractmethod

class StreamContextBase(ABC):
    r"""Base stream context class abstraction for multi backends stream to herit from
    """
    def __init__(self) -> None:
        pass


    @abstractmethod
    def __enter__(self):
        raise NotImplementedError(f"unimplement __enter__ in stream context base class")


    @abstractmethod
    def __exit__(self, type: Any, value: Any, traceback: Any):
        raise NotImplementedError(f"unimplement __exit__ in stream context base class")
