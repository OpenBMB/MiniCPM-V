import importlib.util
from pathlib import Path

import pytest
import torch
from PIL import Image


@pytest.fixture
def dataset_module():
    path = Path(__file__).resolve().parents[1] / 'finetune/dataset.py'
    spec = importlib.util.spec_from_file_location('minicpm_dataset_recovery_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_dataset(module, rows):
    return module.SupervisedDataset(rows, transform=None, tokenizer=None, slice_config=None)


def valid_row(tmp_path):
    path = tmp_path / 'image.png'
    Image.new('RGB', (2, 2), (12, 34, 56)).save(path)
    return {'image': str(path), 'conversations': []}


def preprocessing(images, conversations, *args, **kwargs):
    assert images['<image>'].getpixel((0, 0)) == (12, 34, 56)
    return {
        'input_ids': torch.tensor([1, 2]),
        'position_ids': torch.tensor([0, 1]),
        'target': torch.tensor([-100, 2]),
        'pixel_values': [],
        'tgt_sizes': [],
        'image_bound': [],
    }


def test_corrupt_record_recovers_without_out_of_range_random_index(dataset_module, tmp_path, monkeypatch):
    rows = [{'image': str(tmp_path / 'missing.png'), 'conversations': []}, valid_row(tmp_path)]
    monkeypatch.setattr(dataset_module, 'preprocess', preprocessing)
    # An inclusive upper bound currently draws len(dataset), then recurses again.
    draws = []

    def choose_upper(low, high):
        draws.append((low, high))
        if len(draws) > 2:
            raise RuntimeError('Unbounded invalid-index recovery')
        return high

    monkeypatch.setattr(dataset_module.random, 'randint', choose_upper)
    result = make_dataset(dataset_module, rows)[0]
    assert result['input_ids'].tolist() == [1, 2]
    assert result['attention_mask'].tolist() == [True, True]


def test_all_invalid_records_raise_with_original_cause(dataset_module, tmp_path, monkeypatch):
    rows = [{'image': str(tmp_path / f'missing-{i}.png'), 'conversations': []} for i in range(3)]
    attempts = []
    original_open = dataset_module.Image.open

    def track_open(path):
        attempts.append(path)
        return original_open(path)

    monkeypatch.setattr(dataset_module.Image, 'open', track_open)
    monkeypatch.setattr(dataset_module.random, 'randint', lambda low, high: 0)
    with pytest.raises(RuntimeError, match='Unable to load any of 3 training samples') as error:
        make_dataset(dataset_module, rows)[0]
    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert attempts == [row['image'] for row in rows]


def test_requested_valid_record_is_not_replaced(dataset_module, tmp_path, monkeypatch):
    monkeypatch.setattr(dataset_module, 'preprocess', preprocessing)
    result = make_dataset(dataset_module, [valid_row(tmp_path)])[0]
    assert result['labels'].tolist() == [-100, 2]


@pytest.mark.parametrize('index', [-2, 1, 2])
def test_out_of_range_index_is_not_resampled(dataset_module, tmp_path, index):
    with pytest.raises(IndexError):
        make_dataset(dataset_module, [valid_row(tmp_path)])[index]


def test_empty_dataset_reports_index_error(dataset_module):
    with pytest.raises(IndexError):
        make_dataset(dataset_module, [])[0]
