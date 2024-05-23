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


#### Customizing Hyperparameters
To tailor the training process according to your specific requirements, you can adjust various hyperparameters. For comprehensive documentation on available hyperparameters and their functionalities, you can refer to the [official Transformers documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments). Experimentation and fine-tuning of these parameters are essential for achieving optimal model performance tailored to your specific task and dataset.
