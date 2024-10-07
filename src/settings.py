import torch


DEVICE: str = "cpu"
DTYPE: torch.dtype = torch.float32

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)
torch.set_float32_matmul_precision("medium")



