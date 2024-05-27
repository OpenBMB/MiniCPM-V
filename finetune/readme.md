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
sh finetune_ds_lora.sh
```

After training, you could load the model with the path to the adapter. We advise you to use absolute path for your pretrained model. This is because LoRA only saves the adapter and the absolute path in the adapter configuration json file is used for finding out the pretrained model to load.

```
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    # path to the output directory
    path_to_adapter,
    device_map="auto",
    trust_remote_code=True
).eval()
```

### Finetuning FAQs
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
