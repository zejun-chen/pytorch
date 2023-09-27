from torch.device_interface import DeviceInterface, get_interface_for_device

# TODO: OOB. torch.cuda.current_stream()  -->  torch.gpu.current_stream(device_type='cuda')
# TODO: OOB. torch.cuda.Event()  -->  torch.gpu.Event(device_type='cuda')
# TODO: OOB. torch.cuda.synchronize()  -->  torch.gpu.synchronize(device_type='cuda')
# TODO: name change?
# TODO: location change?
class gpu:
    def __getattr__(self, name):
        if not hasattr(DeviceInterface, name):
            raise RuntimeError("device interface cannot find runtime API for {}".format(name))

    class Stream:
        def __init__(self, device_type: str, device=None, priority=0, **kwargs):
            return get_interface_for_device(device_type).Stream(device, priority, **kwargs)

    class Event:
        def __init__(self, device_type: str, enable_timing=False, blocking=False, interprocess=False):
            return get_interface_for_device(device_type).Event(enable_timing, blocking, interprocess)

    @staticmethod
    def current_device(device_type: str):
        return get_interface_for_device(device_type).current_device()

    @staticmethod
    def set_device(device_type: str, device):
        get_interface_for_device(device_type).set_device(device)

    @staticmethod
    def device_count(device_type: str):
        return get_interface_for_device(device_type).device_count()

    @staticmethod
    def is_available(device_type: str) -> bool:
        return get_interface_for_device(device_type).is_available()

    @staticmethod
    def stream(device_type: str, stream):
        return get_interface_for_device(device_type).stream(stream)

    @staticmethod
    def current_stream(device_type: str):
        return get_interface_for_device(device_type).current_stream()

    @staticmethod
    def set_stream(device_type: str, stream):
        get_interface_for_device(device_type).set_stream(stream)

    @staticmethod
    def set_stream_by_id(device_type: str, stream_id: int, device_index: int, device_type_id: int):
        get_interface_for_device(device_type).set_stream_by_id(stream_id, device_index, device_type_id)

    @staticmethod
    def get_raw_stream(device_type: str):
        return get_interface_for_device(device_type).get_raw_stream()

    @staticmethod
    def synchronize(device_type: str, device):
        return get_interface_for_device(device_type).synchronize(device)

    @staticmethod
    def get_device_properties(device_type: str, device):
        return get_interface_for_device(device_type).get_device_properties(device)

    @staticmethod
    def get_compute_capability(device_type: str, device):
        return get_interface_for_device(device_type).get_compute_capability(device)
