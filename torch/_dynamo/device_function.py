from typing import Dict, Union, Optional, List, Callable
import warnings

import torch

_device_t = Union[torch.device, str, int, None]

class GPUFunction:
    """
    This is a simple device runtime abstraction for Inductor. It enables custom
    backends to be integrated with Inductor in a device-agnostic semantic.
    """
    class Event:
        def __new__(cls, *args, **kwargs):
            raise NotImplementedError()

    class device:
        def __new__(cls, device: _device_t):
            raise NotImplementedError()

    @staticmethod
    def current_device():
        raise NotImplementedError()

    @staticmethod
    def set_device(device: _device_t):
        raise NotImplementedError()

    @staticmethod
    def device_count():
        raise NotImplementedError()

    @staticmethod
    def is_available():
        raise NotImplementedError()

    @staticmethod
    def stream(stream: torch.Stream):
        raise NotImplementedError()

    @staticmethod
    def current_stream(device: Optional[_device_t] = None):
        raise NotImplementedError()

    @staticmethod
    def set_stream(stream: torch.Stream):
        raise NotImplementedError()

    @staticmethod
    def set_stream_by_id(stream_id: int, device_index: int, device_type: int):
        raise NotImplementedError()

    @staticmethod
    def get_raw_stream():
        raise NotImplementedError()

    @staticmethod
    def synchronize(device: _device_t = None):
        raise NotImplementedError()

    @staticmethod
    def get_device_properties(device: _device_t = None):
        raise NotImplementedError()

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        raise NotImplementedError()

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

device_functions: Dict[str, GPUFunction] = {}

'''
This container is used to contain all stream-related functions for dynamo capture
'''
stream_function_container: Dict[str, Dict[str, Callable]] = {}

def _register_stream_function_for_device(device: str, device_function: GPUFunction):
    stream_function_container['stream'] = {device: getattr(device_function, 'stream_func', Callable)}
    stream_function_container['set_stream'] = {device: getattr(device_function, 'set_stream_func', Callable)}
    stream_function_container['current_stream'] = {device: getattr(device_function, 'current_stream_func', Callable)}
    stream_function_container['set_stream_by_id'] = {device: getattr(device_function, 'set_stream_by_id_func', Callable)}

def register_function_for_device(device: str, device_function: GPUFunction):
    if device in device_functions:
        warnings.warn(
            "device {device} has been register already"
        )
    device_functions[device] = device_function
    _register_stream_function_for_device(device, device_function)

def get_function_for_device(device: str):
    return device_functions[device] if device in device_functions else None

def get_registered_device_functions():
    return device_functions.items()

register_function_for_device("cuda", CudaFunction)
