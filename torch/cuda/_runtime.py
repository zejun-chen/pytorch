from typing import Dict, Union, Optional

import torch
from torch._device_runtime import GPUFunction, register_function_for_device

_device_t = Union[torch.device, str, int, None]

class CudaFunction(GPUFunction):

    stream_func = torch.cuda.stream
    current_stream_func = torch.cuda.current_stream
    set_stream_func = torch.cuda.set_stream
    set_stream_by_id_func = torch.cuda.set_stream_by_id

    class Event:
        def __new__(cls, *args, **kwargs):
            return torch.cuda.Event(*args, **kwargs)

    class device:
        def __new__(cls, device: _device_t):
            return torch.cuda.device(device)

    @staticmethod
    def current_device() -> int:
        return torch.cuda.current_device()

    @staticmethod
    def set_device(device: _device_t):
        torch.cuda.set_device(device)

    @staticmethod
    def device_count() -> int:
        return torch.cuda.device_count()

    @staticmethod
    def is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def stream(stream: torch.Stream):
        return torch.cuda.stream(stream)

    @staticmethod
    def current_stream(device: Optional[_device_t] = None):
        return torch.cuda.current_stream(device)

    @staticmethod
    def set_stream(stream: torch.Stream):
        torch.cuda.set_stream(stream)

    @staticmethod
    def set_stream_by_id(stream_id: int, device_index: int, device_type: int):
        torch.cuda.set_stream_by_id(stream_id, device_index, device_type)

    @staticmethod
    def get_raw_stream(device: int):
        from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
        return get_cuda_stream(device)

    @staticmethod
    def synchronize(device: _device_t = None):
        return torch.cuda.synchronize(device)

    @staticmethod
    def get_device_properties(device: _device_t = None):
        return torch.cuda.get_device_properties(device)

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        major, min = torch.cuda.get_device_capability(device)
        return major * 10 + min

register_function_for_device("cuda", CudaFunction)
