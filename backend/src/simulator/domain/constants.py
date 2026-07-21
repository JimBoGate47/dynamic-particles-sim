from enum import Enum

import torch


class DeviceType(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


DEVICE_MAP: dict[DeviceType, torch.device] = {
    DeviceType.CPU: torch.device("cpu"),
    DeviceType.CUDA: torch.device("cuda"),
    DeviceType.AUTO: torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}
