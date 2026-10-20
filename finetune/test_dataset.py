import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataset

TOKENIZER = SimpleNamespace(
    im_start="<image>",
    im_end="</image>",
    slice_start="<slice>",
    slice_end="</slice>",
    im_id_start="<image_id>",
    im_id_end="</image_id>",
    unk_token="<unk>",
)


def slice_config(max_slice_nums):
    return {"patch_size": 14, "max_slice_nums": max_slice_nums, "scale_resolution": 448}


@pytest.mark.parametrize(
    "sizes, config",
    [
        ([(336, 336), (1000, 1000), (300, 200)], slice_config(9)),
        ([(1000, 1000), (1000, 1000)], slice_config(1)),
        ([(336, 336), (300, 200)], None),
    ],
)
def test_every_image_gets_an_id(monkeypatch, sizes, config):
    prompts = []
    monkeypatch.setattr(
        dataset,
        "conversation_to_ids",
        lambda conversations, *args: prompts.append(conversations[0]["content"]) or {},
    )
    images = {f"<image_{i:02d}>": Image.new("RGB", size) for i, size in enumerate(sizes)}
    conversations = [
        {"role": "user", "content": " ".join(images)},
        {"role": "assistant", "content": "ok"},
    ]
    dataset.preprocess(
        images,
        conversations,
        TOKENIZER,
        lambda img: torch.zeros(3, img.size[1], img.size[0]),
        query_nums=2,
        slice_config=config,
        llm_type="qwen",
    )
    for idx in range(len(sizes)):
        assert f"<image_id>{idx}</image_id><image><unk><unk></image>" in prompts[0]
    assert prompts[0].count("<image_id>") == len(sizes)
