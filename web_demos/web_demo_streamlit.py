import gradio as gr
from PIL import Image
import traceback
import re
import torch
import argparse
from transformers import AutoModel, AutoTokenizer

# README, How to run demo on different devices
# For Nvidia GPUs support BF16 (like A100, H100, RTX3090)
# python web_demo.py --device cuda --dtype bf16

# For Nvidia GPUs do NOT support BF16 (like V100, T4, RTX2080)
# python web_demo.py --device cuda --dtype fp16

# For Mac with MPS (Apple silicon or AMD GPUs).
# PYTORCH_ENABLE_MPS_FALLBACK=1 python web_demo.py --device mps --dtype fp16

# Argparser
parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--device', type=str, default='cuda', help='cuda or mps')
parser.add_argument('--dtype', type=str, default='bf16', help='bf16 or fp16')
args = parser.parse_args()
device = args.device
assert device in ['cuda', 'mps']
if args.dtype == 'bf16':
    if device == 'mps':
        print('Warning: MPS does not support bf16, will use fp16 instead')
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
else:
    dtype = torch.float16

# Load model
model_path = 'openbmb/MiniCPM-V-2'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = model.to(device=device, dtype=dtype)
model.eval()

ERROR_MSG = "Error, please retry"
model_name = 'MiniCPM-V 2.0'

form_radio = {'choices': ['Beam Search', 'Sampling'],'value': 'Sampling','interactive': True,'label': 'Decode Type'}
# Beam Form
num_beams_slider = {'minimum': 0,'maximum': 5,'value': 3,'step': 1,'interactive': True,'label': 'Num Beams'}
repetition_penalty_slider = {'minimum': 0,'maximum': 3,'value': 1.2,'step': 0.01,'interactive': True,'label': 'Repetition Penalty'}
repetition_penalty_slider2 = {'minimum': 0,'maximum': 3,'value': 1.05,'step': 0.01,'interactive': True,'label': 'Repetition Penalty'}

# Handling both max_new_tokens_slider and top_p_slider
max_new_tokens_slider = {'minimum': 1,'maximum': 4096,'value': 1024,'step': 1,'interactive': True,'label': 'Max New Tokens'}
top_p_slider = {'minimum': 0,'maximum': 1,'value': 0.8,'step': 0.05,'interactive': True,'label': 'Top P'}
top_k_slider = {'minimum': 0,'maximum': 200,'value': 100,'step': 1,'interactive': True,'label': 'Top K'}
temperature_slider = {'minimum': 0, 'maximum': 2,'value': 0.7,'step': 0.05,'interactive': True,'label': 'Temperature'}

def create_component(params, comp='Slider'):
    if comp == 'Slider':
        return gr.Slider(minimum=params['minimum'],maximum=params['maximum'],value=params['value'],step=params['step'],interactive=params['interactive'],label=params['label'])
    elif comp == 'Radio':
        return gr.Radio(choices=params['choices'],value=params['value'],interactive=params['interactive'],label=params['label'])
    elif comp == 'Button':
        return gr.Button(value=params['value'],interactive=True)

def chat(img, msgs, ctx, params=None, vision_hidden_states=None):
    default_params = {"num_beams":3, "repetition_penalty": 1.2, "max_new_tokens": 1024}
    if params is None:
        params = default_params
    if img is None:
        return -1, "Error, invalid image, please upload a new image", None, None
    try:
        image = img.convert('RGB')
        answer, context, _ = model.chat(image=image,msgs=msgs,context=None,tokenizer=tokenizer,**params)
        res = re.sub(r'(<box>.*</box>)', '', answer)
        res = res.replace('<ref>', '')
        res = res.replace('</ref>', '')
        res = res.replace('<box>', '')
        answer = res.replace('</box>', '')
        return 0, answer, None, None
    except Exception as err:
        print(err)
        traceback.print_exc()
        return -1, ERROR_MSG, None, None

def upload_img(image, _chatbot, _app_session):
    image = Image.fromarray(image)
    _app_session['sts']=None
    _app_session['ctx']=[]
    _app_session['img']=image 
    _chatbot.append(('', 'Image uploaded successfully, you can talk to me now'))
    return _chatbot, _app_session

def respond(_question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty, repetition_penalty_2, top_p, top_k, temperature):
    if _app_cfg.get('ctx', None) is None:
        _chat_bot.append((_question, 'Please upload an image to start'))
        return '', _chat_bot, _app_cfg
    _context = _app_cfg['ctx'].copy()
    if _context:
        _context.append({"role": "user", "content": _question})
    else:
        _context = [{"role": "user", "content": _question}] 
    print('<User>:', _question)
    if params_form == 'Beam Search':
        params = {'sampling': False,'num_beams': num_beams,'repetition_penalty': repetition_penalty,"max_new_tokens": 896 }
    else:
        params = {'sampling': True,'top_p': top_p,'top_k': top_k,'temperature': temperature,'repetition_penalty': repetition_penalty_2,"max_new_tokens": 896 }
    code, _answer, _, sts = chat(_app_cfg['img'], _context, None, params)
    print('<Assistant>:', _answer)
    _context.append({"role": "assistant", "content": _answer}) 
    _chat_bot.append((_question, _answer))
    if code == 0:
        _app_cfg['ctx']=_context
        _app_cfg['sts']=sts
    return '', _chat_bot, _app_cfg

def regenerate_button_clicked(_question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty, repetition_penalty_2, top_p, top_k, temperature):
    return respond(_question, _chat_bot, _app_cfg, params_form, num_beams, repetition_penalty, repetition_penalty_2, top_p, top_k, temperature)