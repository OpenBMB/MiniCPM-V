import warnings
import pandas as pd
from abc import abstractmethod
from ..smp import *
from .image_base import ImageBaseDataset


class MMGenBench(ImageBaseDataset):

    prompt_list = [
        """
# Role
You are an expert in the field of image understanding, focusing on the \
understanding of images and generating the image caption-prompt.

# Definition Explanation
image caption-prompt: Refers to the caption or description of an image, \
used to provide to a Text-to-Image model to generate a new image.
Text-to-Image model: Can generate a new image based on the provided image \
caption-prompt, such as stable diffusion 3, flux, and other image generation models.

# Task Description
Generate an image caption-prompt based on the input image.

# Key Points and Requirements
1. Accurately understand the input image and precisely generate an image caption-prompt.
2. The generated image caption-prompt, when provided to the Text-to-Image model, requires the \
Text-to-Image model to generate a new image that is as consistent as possible with the input image.
3. The generated image caption-prompt must conform to the preferences of the Text-to-Image model.
4. The generated image caption-prompt should describe the input image in as much \
detail as possible, and it should be between 20 to 60 words.

# Output Format
A string, that is the image caption-prompt. No extra output needed.
"""
    ]
    TYPE = 'GenerateImgPrompt'
    DATASET_URL = {
        'MMGenBench-Test': 'https://huggingface.co/datasets/lerogo/MMGenBench/resolve/main/MMGenBench-Test.tsv',
        'MMGenBench-Domain': 'https://huggingface.co/datasets/lerogo/MMGenBench/resolve/main/MMGenBench-Domain.tsv',
    }
    PROMPT_MAP = {
        'MMGenBench-Test': prompt_list[0],
        'MMGenBench-Domain': prompt_list[0],
    }
    DATASET_MD5 = {
        'MMGenBench-Test': "94f8dac6bbf7c20be403f99adeaa73da",
        'MMGenBench-Domain': "5c10daf6e2c5f08bdfb0701aa6db86bb",
    }

    def __init__(self, dataset='MMGenBench', **kwargs):
        super().__init__(dataset, **kwargs)
        warnings.warn('This dataset is for inference only and does not support direct output of evaluation results.\n')
        warnings.warn('Please refer to "https://github.com/lerogo/MMGenBench" for more evaluation information.\n')

    def load_data(self, dataset):
        data = super().load_data(dataset)
        if 'question' not in data:
            data['question'] = [(
                self.PROMPT_MAP[dataset]
            )] * len(data)
        return data

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        warnings.warn('This evaluation method is not supported.\n')
        warnings.warn('Please refer to "https://github.com/lerogo/MMGenBench" for more evaluation information.\n')
        return None
