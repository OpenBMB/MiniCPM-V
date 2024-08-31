import torch

torch.set_grad_enabled(False)
torch.manual_seed(1234)
from .base import BaseModel
from .minicpm_v import MiniCPM_V, MiniCPM_Llama3_V, MiniCPM_V_2_6
