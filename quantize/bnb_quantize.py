"""
the script will use bitandbytes to quantize the MiniCPM-Llama3-V-2_5 model.
the be quantized model can be finetuned by MiniCPM-Llama3-V-2_5 or not.
you only need to set the model_path 、save_path and run bash code 

cd MiniCPM-V
python quantize/bnb_quantize.py

you will get the quantized model in save_path、quantized_model test time and gpu usage
"""


import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import time
import torch
import GPUtil
import os

assert torch.cuda.is_available(),"CUDA is not available, but this code requires a GPU."

device = 'cuda'  # Select GPU to use
model_path = '/root/ld/ld_model_pretrained/MiniCPM-Llama3-V-2_5' # Model download path
save_path = '/root/ld/ld_model_pretrain/MiniCPM-Llama3-V-2_5_int4' # Quantized model save path
image_path = './assets/airplane.jpeg'


# Create a configuration object to specify quantization parameters
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Whether to perform 4-bit quantization
    load_in_8bit=False,  # Whether to perform 8-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Computation precision setting
    bnb_4bit_quant_storage=torch.uint8,  # Storage format for quantized weights
    bnb_4bit_quant_type="nf4",  # Quantization format, here using normally distributed int4
    bnb_4bit_use_double_quant=True,  # Whether to use double quantization, i.e., quantizing zeropoint and scaling parameters
    llm_int8_enable_fp32_cpu_offload=False,  # Whether LLM uses int8, with fp32 parameters stored on the CPU
    llm_int8_has_fp16_weight=False,  # Whether mixed precision is enabled
    llm_int8_skip_modules=["out_proj", "kv_proj", "lm_head"],  # Modules not to be quantized
    llm_int8_threshold=6.0  # Outlier value in the llm.int8() algorithm, distinguishing whether to perform quantization based on this value
)

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    device_map=device,  # Allocate model to device
    quantization_config=quantization_config,
    trust_remote_code=True
)

gpu_usage = GPUtil.getGPUs()[0].memoryUsed  
start=time.time()
response = model.chat(
    image=Image.open(image_path).convert("RGB"),
    msgs=[
        {
            "role": "user",
            "content": "What is in this picture?"
        }
    ],
    tokenizer=tokenizer
) # 模型推理
print('Output after quantization:',response)
print('Inference time after quantization:',time.time()-start)
print(f"GPU memory usage after quantization: {round(gpu_usage/1024,2)}GB")

"""
Expected output:

    Output after quantization: This picture contains specific parts of an airplane, including wings, engines, and tail sections. These components are key parts of large commercial aircraft.
    The wings support lift during flight, while the engines provide thrust to move the plane forward. The tail section is typically used for stabilizing flight and plays a role in airline branding.
    The design and color of the airplane indicate that it belongs to Air China, likely a passenger aircraft due to its large size and twin-engine configuration.
    There are no markings or insignia on the airplane indicating the specific model or registration number; such information may require additional context or a clearer perspective to discern.
    Inference time after quantization: 8.583992719650269 seconds
    GPU memory usage after quantization: 6.41 GB
"""

# Save the model and tokenizer
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path, safe_serialization=True)
tokenizer.save_pretrained(save_path)