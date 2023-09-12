from abc import ABC, abstractmethod
from typing import Any


class StreamContextBase(ABC):
    r"""Base stream context class abstraction for multi backends stream to herit from"""

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError()

    @abstractmethod
    def __exit__(self, type: Any, value: Any, traceback: Any):
        raise NotImplementedError()
