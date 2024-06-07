# MiniCPM-V Finetuning


We offer the official scripts for easy finetuning of the pretrained **MiniCPM-Llama3-V 2.5** and **MiniCPM-V 2.0** on downstream tasks. Our finetune scripts use transformers Trainer and DeepSpeed by default.

### Data preparation

To prepare your finetuning data, you should formulate each sample as a dictionary consisting of an id, an image path list with an image, and a list of conversations. Then save data samples in JSON files.

For the vision-language example with image, you are required to provide **\<image\>** to define the position to insert the image embeddings. If you don't provide \<image\>, the image will be placed at the front of the conversation.

<details>
  <summary>
    <b>vision-language example (vl_finetune_data.json) with 1 samples.</b>
  </summary>

```
  [
    {
      "id": "0",
      "image": 'path/to/image_0.jpg',
      "conversations": [
            {
              'role': 'user', 
              'content': '<image>\nHow many desserts are on the white plate?'
            }, 
            {
                'role': 'assistant', 
                'content': 'There are three desserts on the white plate.'
            },   
            {
                'role': 'user', 
                'content': 'What type of desserts are they?'
            },
            {
                'role': 'assistant', 
                'content': 'The desserts are cakes with bananas and pecans on top. They share similarities with donuts, but the presence of bananas and pecans differentiates them.'
            }, 
            {
                'role': 'user', 
                'content': 'What is the setting of the image?'}, 
            {
                'role': 'assistant', 
                'content': 'The image is set on a table top with a plate containing the three desserts.'
            },
        ]
    },
  ]
```

</details>

### Full-parameter finetuning

Full-parameter parameter finetuning requires updating all parameters of LLM in the whole training process. Please specify the correct MODEL path, DATA path and LLM_TYPE in the shell scripts.

```shell
MODEL="openbmb/MiniCPM-Llama3-V-2_5" # or openbmb/MiniCPM-V-2
DATA="path/to/trainging_data" # json file
EVAL_DATA="path/to/test_data" # json file
LLM_TYPE="llama3" # if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm
```

To launch your training, run the following script:

```
sh finetune_ds.sh
```

Specially, Llama3 has a different chat_template for training and inference, we modified the chat_template for training, so please take care to restore the chat_template when inference on the training ckpt.

### LoRA finetuning

The LoRA allows light-weight model tuning with only a small subset of parameters updated. We provide the LoRA implementation based on `peft`. To launch your training, run the following script:

```
sh finetune_lora.sh
```

After training, you could load the model with the path to the adapter. We advise you to use absolute path for your pretrained model. This is because LoRA only saves the adapter and the absolute path in the adapter configuration json file is used for finding out the pretrained model to load.

```
from peft import AutoPeftModelForCausalLM

path_to_adapter="path_to_adapter"

model = AutoPeftModelForCausalLM.from_pretrained(
    # path to the output directory
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()

vpm_resampler_embedtokens_weight = torch.load(f"{path_to_adapter}/vpm_resampler_embedtokens.pt")

msg = model.load_state_dict(vpm_resampler_embedtokens_weight, strict=False)
```


### Model Fine-tuning Memory Usage Statistics

The following table presents the memory usage of the model when fine-tuning using NVIDIA A100 (80GiB) GPUs under different numbers of GPUs. The fine-tuning was performed with the DeepSpeed Zero-3 optimization, Gradient Checkpointing techniques and offloading optimizer as well as parameters memory to cpu, with a maximum length set to 2048 and batch size set to 1. You refer to [deepspeed zero stage](https://huggingface.co/docs/transformers/v4.41.2/en/deepspeed#select-a-zero-stage) to reduce memory cost.

| Fine-tuning Method | GPUs: 2 | GPUs: 4 | GPUs: 8 |
|--------------------|---------|---------|---------|
| LoRA Fine-tuning   | 14.4 GiB| 13.6 GiB|   13.1 GiB   |
| Full Parameters Fine-tuning | 16.0 GiB | 15.8 GiB | 15.63GiB |

### Notes
- **Fine-tuning Method**: Displays two different fine-tuning strategies, LoRA fine-tuning and Full parameters fine-tuning.
- **Number of GPUs**: The table lists the memory usage for configurations with 2, 4, and 8 GPUs.
- **Memory Usage**: Expressed in GiB, this shows the required memory for each fine-tuning method under corresponding GPU configurations.
- **Out of memory**: Indicates that the memory was insufficient for full parameters fine-tuning under the current GPU configurations.

### Finetuning FAQs

<details>
<summary>Q:When you encounter Out of Memory (OOM) issues during training large models, you can try the following methods to resolve or mitigate the issue:</summary>

A：When you face Out of Memory (OOM) issues during training large models, the following strategies may help resolve or mitigate the problem:
#### Adjust Model Hyperparameters
- **Reduce `max_model_length`**: Decreasing the maximum sequence length the model processes can significantly reduce the memory required for each operation. For example, reducing the maximum length from 2048 to 1200 or another value suitable for your dataset.
```
--model_max_length 1200

```
- **Lower `batch_size`**: Reducing the amount of data processed in each batch helps decrease memory consumption.
```
--batch_size 1
 ```
- **Reduce the number of slices (`slice`)**: When handling large datasets such as large images files, reducing the number of slices processed each time can lower memory requirements.
```
--max_slice_nums 9 
```

#### Reduce Training Model Parameters
- **Do not train VPM (Visual Processing Module)**: You can adjust hyperparameters in the finetune script to opt out of training the visual processing module to save memory.
```
--tune_vision false
```
- **Use LoRA finetuning**: Refer to the [LoRA finetuning](#LoRA-finetuning) section.

#### Optimize with DeepSpeed
- **Configure DeepSpeed Zero Stage 2**: Use the following configuration to offload optimizer parameters to the CPU, reducing memory pressure on the GPU:
  ```json
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
  }
- **Configure DeepSpeed Zero Stage 3**：Further offload model parameters and optimizer parameters to the CPU, further reducing GPU memory usage:
```json
"zero_optimization": {
  "stage": 3,
  "offload_optimizer": {
    "device": "cpu",
    "pin_memory": true
  },
  "offload_param": {
    "device": "cpu",
    "pin_memory": true
  }
}
```
You can visit [huggingface deepspeed](https://huggingface.co/docs/transformers/deepspeed) to find out more about how to use DeepSpeed.
</details>
<details>
<summary>Q: Encounter an error while using the AutoPeftModelForCausalLM to load a checkpoint that has undergone lora fine-tuning</summary>

A: The error as described in [issues 168](https://github.com/OpenBMB/MiniCPM-V/issues/168) occurs because the model lacks `get_input_embeddings` and `set_input_embeddings` methods. Follow these steps to resolve this issue: 

1.**Reload the Fine-Tuned Model:** Make sure you correctly load the checkpoint that has been fine-tuned using lora techniques. Use the following code example to guide you:
   ```python
   from peft import AutoPeftModelForCausalLM

   model = AutoPeftModelForCausalLM.from_pretrained(
       'path_to_your_fine_tuned_checkpoint',  # Path to your fine-tuned checkpoint directory
       output='output/minicpmv2_lora',
       device_map='auto',
       trust_remote_code=True
   ).eval()
   ```
  2.**Update the `model_minicpmv.py` File:**
   - **Verification:** Make sure you verify and update your `model_minicpmv.py` file to ensure it is the latest version.
   - **Update Hugging Face Library Code:** If the issue persists after updating the file, consider updating the related code in the Hugging Face library.
   - **Direct File Copy:** For a quick resolution, directly download and copy the latest `model_minicpmv.py` file into your project. This file is available from the following sources:
     - [MiniCPM-Llama3-V-2_5 on Hugging Face](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5/tree/main)
     - [MiniCPM-V-2 on Hugging Face](https://huggingface.co/openbmb/MiniCPM-V-2)
</details>

<details>
<summary>Q: How do I use the `flash_attention_2` implementation when loading a pretrained model?</summary>

A: If your environment supports `flash_attn2`, you can add an argument `_attn_implementation="flash_attention_2"` when using the `AutoModel.from_pretrained` method to load a model. For example:

```python
model = AutoModel.from_pretrained('model_name', _attn_implementation="flash_attention_2")
```
</details>

<details>
<summary>Q: What if our data is resized to 512? Can we use the original image size instead?</summary>

A: Our model supports up to 1344x1344 lossless encoding. If you are currently resizing your images to 512, you might want to try using the original image sizes instead. Our system automatically includes a high-definition image encoding scheme by default.

</details>

<details>
<summary>Q: What should we do if we encounter out-of-memory (OOM) errors?</summary>

A: If you experience OOM issues, consider reducing the batch size (`bs`). To maintain an equivalent total batch size, you can adjust the `gradient_accumulation_steps` setting. This approach allows you to manage memory usage effectively while still processing the desired amount of data per training step.
</details>

<details>
<summary>Q: How can we determine the maximum length for our training data, and what if we do not want to train the vision encoder?</summary>

A: I recommend using this function [here](https://github.com/OpenBMB/MiniCPM-V/blob/main/finetune/dataset.py#L220) to sample the length of your training data. Note that the `input_ids` length includes the image portion. Once you determine the maximum length, you can specify it in the startup command using `--model_max_length xxx`.

Additionally, if you prefer not to train the vision encoder, you can add `--tune_vision false` to your command.

</details>

<details>
<summary>Q: How can we adjust training hyperparameters when using LoRA to train our model?</summary>

A: You can refer to the [LoRA documentation](https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig) for guidance on adjusting your training hyperparameters when using LoRA. This documentation provides detailed information on configuring various parameters specific to the LoRA adaptation technique.
</details>

#### Customizing Hyperparameters
To tailor the training process according to your specific requirements, you can adjust various hyperparameters. For comprehensive documentation on available hyperparameters and their functionalities, you can refer to the [official Transformers documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) and [Lora documentation](https://huggingface.co/docs/peft/en/package_reference/lora#peft.LoraConfig). Experimentation and fine-tuning of these parameters are essential for achieving optimal model performance tailored to your specific task and dataset.
