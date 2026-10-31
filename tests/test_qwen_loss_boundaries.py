import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def dataset():
    path = Path(__file__).resolve().parents[1] / 'finetune/dataset.py'
    spec = importlib.util.spec_from_file_location('minicpm_dataset_loss_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    """Small ChatML tokenizer with ordinary 'assistant' tokens in message text."""

    tokens = {'<|im_start|>': 1, '<|im_end|>': 2, 'assistant': 3, 'user': 4}
    empty_thinking = [7, 8, 9, 8]

    def __init__(self, ids, rendered):
        self.ids = ids
        self.rendered = rendered

    def convert_tokens_to_ids(self, token):
        return self.tokens[token]

    def encode(self, text, add_special_tokens=False):
        assert text == '<think>\n\n</think>\n\n'
        assert add_special_tokens is False
        return list(self.empty_thinking)

    def apply_chat_template(self, chat, tokenize, **kwargs):
        return self.ids if tokenize else self.rendered


def make_conversation():
    return [{'role': 'user', 'content': 'question'}, {'role': 'assistant', 'content': 'answer'}]


def test_literal_assistant_in_answer_does_not_expose_empty_thinking_prefix(dataset):
    # <im_start>user\nQ<im_end>\n<im_start>assistant\n<think>\n\n</think>\n\nA assistant B<im_end>
    ids = [1, 4, 5, 20, 2, 5, 1, 3, 5, 7, 8, 9, 8, 21, 3, 22, 2]
    tokenizer = Tokenizer(ids, '<think>\n\n</think>\n\nA assistant B')
    actual_ids, mask, _ = dataset.conversation_to_ids_qwen2(make_conversation(), tokenizer)
    np.testing.assert_array_equal(actual_ids, ids)
    np.testing.assert_array_equal(mask, [1] * 13 + [0] * 4)


def test_empty_thinking_in_user_message_does_not_skip_answer_tokens(dataset):
    ids = [1, 4, 5, 7, 8, 9, 8, 2, 5, 1, 3, 5, 21, 22, 23, 24, 25, 2]
    tokenizer = Tokenizer(ids, 'user: <think>\n\n</think>\n\nassistant: first five answer tokens')
    _, mask, _ = dataset.conversation_to_ids_qwen2(make_conversation(), tokenizer)
    np.testing.assert_array_equal(mask, [1] * 12 + [0] * 6)


def test_empty_thinking_in_earlier_turn_does_not_skip_last_answer(dataset):
    ids = [1, 3, 5, 7, 8, 9, 8, 21, 2, 5, 1, 4, 5, 20, 2, 5, 1, 3, 5, 22, 23, 24, 25, 26, 2]
    tokenizer = Tokenizer(ids, '<think>\n\n</think>\n\nold answer and a new answer')
    _, mask, _ = dataset.conversation_to_ids_qwen2(make_conversation(), tokenizer)
    expected = np.ones(len(ids), dtype=np.int8)
    expected[3:9] = 0
    expected[19:25] = 0
    np.testing.assert_array_equal(mask, expected)


def test_final_empty_thinking_prefix_remains_excluded(dataset):
    ids = [1, 4, 5, 20, 2, 5, 1, 3, 5, 7, 8, 9, 8, 21, 2]
    tokenizer = Tokenizer(ids, '<think>\n\n</think>\n\nanswer')
    _, mask, _ = dataset.conversation_to_ids_qwen2(make_conversation(), tokenizer)
    np.testing.assert_array_equal(mask, [1] * 13 + [0] * 2)


def test_plain_chatml_answer_mask_is_unchanged(dataset):
    ids = [1, 4, 5, 20, 2, 5, 1, 3, 5, 21, 22, 2]
    tokenizer = Tokenizer(ids, 'plain answer')
    _, mask, _ = dataset.conversation_to_ids_qwen2(make_conversation(), tokenizer)
    np.testing.assert_array_equal(mask, [1] * 9 + [0] * 3)
