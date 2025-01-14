## MiniCPM-V 1.0


> Archive at：2024-05-19

MiniCPM-V 1.0 is an efficient version with promising performance for deployment. The model is built based on SigLip-400M and [MiniCPM-2.4B](https://github.com/OpenBMB/MiniCPM/), connected by a perceiver resampler. Notable features of MiniCPM-V 1.0 include:

- ⚡️ **High Efficiency.** 

  MiniCPM-V 1.0 can be **efficiently deployed on most GPU cards and personal computers**, and **even on end devices such as mobile phones**. In terms of visual encoding, we compress the image representations into 64 tokens via a perceiver resampler, which is significantly fewer than other LMMs based on MLP architecture (typically > 512 tokens). This allows MiniCPM-V 1.0 to operate with **much less memory cost and higher speed during inference**.

- 🔥 **Promising Performance.** 

  MiniCPM-V 1.0 achieves **state-of-the-art performance** on multiple benchmarks (including MMMU, MME, and MMbech, etc) among models with comparable sizes, surpassing existing LMMs built on Phi-2. It even **achieves comparable or better performance than the 9.6B Qwen-VL-Chat**.

- 🙌 **Bilingual Support.** 

  MiniCPM-V 1.0 is **the first end-deployable LMM supporting bilingual multimodal interaction in English and Chinese**. This is achieved by generalizing multimodal capabilities across languages, a technique from the ICLR 2024 spotlight [paper](https://arxiv.org/abs/2308.12038).

### Evaluation

<div align="center">

<table style="margin: 0px auto;">
<thead>
  <tr>
    <th align="left">Model</th>
    <th>Size</th>
    <th nowrap="nowrap" >Visual Tokens</th>
    <th>MME</th>
    <th nowrap="nowrap" >MMB dev (en)</th>
    <th nowrap="nowrap" >MMB dev (zh)</th>
    <th nowrap="nowrap" >MMMU val</th>
    <th nowrap="nowrap" >CMMMU val</th>
  </tr>
</thead>
<tbody align="center">
  <tr>
    <td align="left">LLaVA-Phi</td>
    <td align="right">3B</td>
    <td>576</td>
    <td>1335</td>
    <td>59.8</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left">MobileVLM</td>
    <td align="right">3B</td>
    <td>144</td>
    <td>1289</td>
    <td>59.6</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >Imp-v1</td>
    <td align="right">3B</td>
    <td>576</td>
    <td>1434</td>
    <td>66.5</td>
    <td>- </td>
    <td>- </td>
    <td>- </td>
  </tr>
  <tr>
    <td  nowrap="nowrap" align="left" >Qwen-VL-Chat</td>
    <td align="right" >9.6B</td>
    <td>256</td>
    <td>1487</td>
    <td>60.6 </td>
    <td>56.7 </td>
    <td>35.9 </td>
    <td>30.7 </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" >CogVLM</td>
    <td align="right">17.4B </td>
    <td>1225</td>
    <td>1438 </td>
    <td>63.7 </td>
    <td>53.8 </td>
    <td>32.1 </td>
    <td>- </td>
  </tr>
  <tr>
    <td nowrap="nowrap" align="left" ><b>MiniCPM-V 1.0</b></td>
    <td align="right">3B </td>
    <td>64</td>
    <td>1452 </td>
    <td>67.9 </td>
    <td>65.3 </td>
    <td>37.2 </td>
    <td>32.1 </td>
  </tr>
</tbody>
</table>

</div>

### Examples

We deploy MiniCPM-V 1.0 on end devices. The demo video is the raw screen recording on a OnePlus 9R without edition.

<table align="center">
    <p align="center">
      <img src="assets/gif_cases/蛇_cn.gif" width=36%/>
      <img src="assets/gif_cases/Mushroom_en.gif" width=36%/>
    </p>
</table>

## Install

1. Clone this repository and navigate to the source folder

```bash
git clone https://github.com/OpenBMB/OmniLMM.git
cd OmniLMM
```

2. Create conda environment

```Shell
conda create -n OmniLMM python=3.10 -y
conda activate OmniLMM
```

3. Install dependencies

```shell
pip install -r requirements.txt
```

## Inference

### Model Zoo
| Model                | Description       | Download Link |
|:----------------------|:-------------------|:---------------:|
| MiniCPM-V 1.0  | The efficient version for end device deployment.    |  [🤗](https://huggingface.co/openbmb/MiniCPM-V) &nbsp;&nbsp; [<img src="./assets/modelscope_logo.png" width="20px"></img>](https://modelscope.cn/models/OpenBMB/MiniCPM-V/files) |


### Multi-turn Conversation
Please refer to the following codes to run `MiniCPM-V 1.0`.

<div align="center">
<img src="assets/worldmap_ck.jpg" width="500px">
</div>


```python
from chat import OmniLMMChat, img2base64

chat_model = OmniLMMChat('openbmb/MiniCPM-V')

im_64 = img2base64('./assets/worldmap_ck.jpg')

# First round chat 
msgs = [{"role": "user", "content": "What is interesting about this image?"}]

inputs = {"image": im_64, "question": json.dumps(msgs)}
answer = chat_model.chat(inputs)
print(answer)

# Second round chat 
# pass history context of multi-turn conversation
msgs.append({"role": "assistant", "content": answer})
msgs.append({"role": "user", "content": "Where is China in the image"})

inputs = {"image": im_64, "question": json.dumps(msgs)}
answer = chat_model.chat(inputs)
print(answer)
```


### Inference on Mac
<details>
<summary>Click to view example, MiniCPM-V 1.0 can run on Mac with MPS (Apple silicon or AMD GPUs). </summary>

```python
# test.py
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.to(device='mps', dtype=torch.float16)

tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V', trust_remote_code=True)
model.eval()

image = Image.open('./assets/worldmap_ck.jpg').convert('RGB')
question = 'What is interesting about this image?'
msgs = [{'role': 'user', 'content': question}]

answer, context, _ = model.chat(
    image=image,
    msgs=msgs,
    context=None,
    tokenizer=tokenizer,
    sampling=True
)
print(answer)
```
Run with command:
```shell
PYTORCH_ENABLE_MPS_FALLBACK=1 python test.py
```
</details>

### Deployment on Mobile Phone

Currently MiniCPM-V 1.0 can be deployed on mobile phones with Android and Harmony operating systems. 🚀 Try it out [here](https://github.com/OpenBMB/mlc-MiniCPM).
