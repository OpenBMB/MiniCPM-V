import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def dataset(monkeypatch):
    path = Path(__file__).resolve().parents[1] / 'finetune/dataset.py'
    spec = importlib.util.spec_from_file_location('minicpm_dataset_images_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def capture_conversation(conversations, *args):
        return {'conversations': conversations}

    # Tokenization is downstream of the image/placeholder alignment under test.
    monkeypatch.setattr(module, 'conversation_to_ids', capture_conversation)
    return module


def tokenizer():
    return SimpleNamespace(
        im_start='<im_start>', unk_token='<unk>', im_end='<im_end>',
        slice_start='<slice_start>', slice_end='<slice_end>',
    )


def transform(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()


def preprocess(dataset, images, messages, **kwargs):
    return dataset.preprocess(
        images, messages, tokenizer(), transform, query_nums=2,
        llm_type='llama3', **kwargs,
    )


def colors(result):
    return [tuple(image[:, 0, 0].tolist()) for image in result['pixel_values']]


def test_image_tensors_follow_placeholder_order_not_dictionary_order(dataset):
    images = {'<image_00>': Image.new('RGB', (28, 28), 'red'), '<image_01>': Image.new('RGB', (28, 28), 'blue')}
    messages = [{'role': 'user', 'content': '<image_01> then <image_00>'}, {'role': 'assistant', 'content': 'blue then red'}]
    result = preprocess(dataset, images, messages)
    assert colors(result) == [(0, 0, 255), (255, 0, 0)]
    assert messages[0]['content'] == '<image_01> then <image_00>'


def test_each_repeated_placeholder_has_corresponding_image_features(dataset):
    images = {'<image_00>': Image.new('RGB', (28, 28), 'red'), '<image_01>': Image.new('RGB', (28, 28), 'blue')}
    messages = [
        {'role': 'user', 'content': '<image_01> <image_00>'},
        {'role': 'assistant', 'content': 'two images'},
        {'role': 'user', 'content': 'Look at <image_01> again'},
        {'role': 'assistant', 'content': 'blue'},
    ]
    assert colors(preprocess(dataset, images, messages)) == [(0, 0, 255), (255, 0, 0), (0, 0, 255)]


def test_unused_dictionary_images_do_not_create_unmatched_features(dataset):
    images = {'<image_00>': Image.new('RGB', (28, 28), 'red'), '<image_01>': Image.new('RGB', (28, 28), 'blue')}
    messages = [{'role': 'user', 'content': '<image_01>'}, {'role': 'assistant', 'content': 'blue'}]
    assert colors(preprocess(dataset, images, messages)) == [(0, 0, 255)]


def test_sliced_image_groups_are_reordered_as_complete_units(dataset):
    images = {'<image_00>': Image.new('RGB', (84, 56), 'red'), '<image_01>': Image.new('RGB', (28, 28), 'blue')}
    messages = [{'role': 'user', 'content': '<image_01> <image_00>'}, {'role': 'assistant', 'content': 'blue then red'}]
    config = {'patch_size': 14, 'max_slice_nums': 4, 'scale_resolution': 28}
    result = preprocess(dataset, images, messages, slice_config=config)
    _, patches, _ = dataset.slice_image(images['<image_00>'], 4, 28, 14)
    expected_red_count = 1 + sum(len(row) for row in patches)
    assert colors(result) == [(0, 0, 255)] + [(255, 0, 0)] * expected_red_count


def test_batch_vision_patch_sizes_follow_reordered_images(dataset):
    images = {'<image_00>': Image.new('RGB', (42, 28), 'red'), '<image_01>': Image.new('RGB', (28, 56), 'blue')}
    messages = [{'role': 'user', 'content': '<image_01> <image_00>'}, {'role': 'assistant', 'content': 'blue then red'}]
    result = preprocess(dataset, images, messages, batch_vision=True)
    assert result['tgt_sizes'].tolist() == [[4, 2], [2, 3]]
    assert colors(result) == [(0, 0, 255), (255, 0, 0)]


def test_original_in_order_multi_image_input_is_unchanged(dataset):
    images = {'<image_00>': Image.new('RGB', (28, 28), 'red'), '<image_01>': Image.new('RGB', (28, 28), 'blue')}
    messages = [{'role': 'user', 'content': '<image_00> <image_01>'}, {'role': 'assistant', 'content': 'red then blue'}]
    assert colors(preprocess(dataset, images, messages)) == [(255, 0, 0), (0, 0, 255)]


def test_single_image_implicit_prefix_is_unchanged(dataset):
    images = {'<image>': Image.new('RGB', (28, 28), 'red')}
    messages = [{'role': 'user', 'content': 'What color?'}, {'role': 'assistant', 'content': 'red'}]
    result = preprocess(dataset, images, messages)
    assert colors(result) == [(255, 0, 0)]
    assert result['conversations'][0]['content'].startswith('<im_start><unk><unk><im_end>')
