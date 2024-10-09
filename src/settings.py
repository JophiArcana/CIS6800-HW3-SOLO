import torch


DEVICE: str = "cuda"
DTYPE: torch.dtype = torch.float32

torch.set_default_device(DEVICE)
torch.set_default_dtype(DTYPE)




