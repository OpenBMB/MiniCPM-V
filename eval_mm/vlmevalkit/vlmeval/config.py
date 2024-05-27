from vlmeval.vlm import *
from vlmeval.api import *
from functools import partial

ungrouped = {
    'MiniCPM-V':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V'),
    'MiniCPM-V-2':partial(MiniCPM_V, model_path='openbmb/MiniCPM-V-2'),
    'MiniCPM-Llama3-V-2_5':partial(MiniCPM_Llama3_V, model_path='openbmb/MiniCPM-Llama3-V-2_5'),
}

supported_VLM = {}

model_groups = [
    ungrouped
]

for grp in model_groups:
    supported_VLM.update(grp)

