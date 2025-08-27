from functools import lru_cache

from .import_utils import is_torch_mps_available, is_torch_cuda_available, _LazyModule


@lru_cache
def get_available_devices() -> frozenset[str]:
    """
    Returns a frozenset of devices available for the current PyTorch installation.
    """
    devices = {"cpu"}  # `cpu` is always supported as a device in PyTorch

    if is_torch_cuda_available():
        devices.add("cuda")

    if is_torch_mps_available():
        devices.add("mps")

    return frozenset(devices)