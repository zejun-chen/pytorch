import torch
from torch.device_interface import (
    caching_worker_current_devices,
    caching_worker_device_properties,
    DeviceInterface,
)

if torch.cuda._is_compiled():
    from torch._C import _cuda_getCurrentRawStream as get_cuda_stream
else:
    get_cuda_stream = None

from typing import Union

_device_t = Union[torch.device, str, int, None]


class CudaInterface(DeviceInterface):
    Event = torch.cuda.Event
    device = torch.cuda.device
    Stream = torch.cuda.Stream

    class Worker:
        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["cuda"] = device

        @staticmethod
        def current_device() -> int:
            if "cuda" in caching_worker_current_devices:
                return caching_worker_current_devices["cuda"]
            return torch.cuda.current_device()

        @staticmethod
        def get_device_properties(device: _device_t = None):
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "cuda"
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = CudaInterface.Worker.current_device()

            if "cuda" not in caching_worker_device_properties:
                device_prop = [
                    torch.cuda.get_device_properties(i)
                    for i in range(torch.cuda.device_count())
                ]
                caching_worker_device_properties["cuda"] = device_prop

            return caching_worker_device_properties["cuda"][device]

    current_device = staticmethod(torch.cuda.current_device)
    set_device = staticmethod(torch.cuda.set_device)
    device_count = staticmethod(torch.cuda.device_count)
    stream = staticmethod(torch.cuda.stream)
    current_stream = staticmethod(torch.cuda.current_stream)
    set_stream = staticmethod(torch.cuda.set_stream)
    set_stream_by_id = staticmethod(torch.cuda.set_stream_by_id)
    synchronize = staticmethod(torch.cuda.synchronize)
    get_device_properties = staticmethod(torch.cuda.get_device_properties)
    get_raw_stream = staticmethod(get_cuda_stream)

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        major, min = torch.cuda.get_device_capability(device)
        return major * 10 + min
