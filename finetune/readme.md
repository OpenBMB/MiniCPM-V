# Minicpm-V2 Finetuning

<div align="center">

[English](README.md) 

</div>

We offer the official scripts for easy finetuning of the pretrained minicpm-v2 model on downstream tasks. Our finetune scripts use DeepSpeed by default.

### Data preparation

To prepare your finetuning data, you should (1) formulate each sample as a dictionary consisting of an id, an image path list with an image (optional, not required for pure-text example), and a list of conversations, and (2) save data samples in JSON files.

For the vision-language example with image, you are required to define placeholder(s) <ImageHere> to define the position to insert the image embeddings.

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

Full-parameter parameter finetuning requires updating all parameters of LLM in the whole training process. To launch your training, run the following script:

```
sh finetune_ds.sh
```
#### Customizing Hyperparameters
To tailor the training process according to your specific requirements, you can adjust various hyperparameters. For comprehensive documentation on available hyperparameters and their functionalities, you can refer to the [official Transformers documentation](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments). Experimentation and fine-tuning of these parameters are essential for achieving optimal model performance tailored to your specific task and dataset.
### LoRA finetuning

**This part is still unfinished, and we will complete it as soon as possible.**
