import copy
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import patch


DATASET_PATH = Path(__file__).with_name("dataset.py")


class Tokenizer:
    im_start = "<im_start>"
    im_end = "<im_end>"
    unk_token = "<unk>"


def load_dataset_module():
    torch = ModuleType("torch")
    torch_nn = ModuleType("torch.nn")
    torch_nn_utils = ModuleType("torch.nn.utils")
    torch_nn_utils_rnn = ModuleType("torch.nn.utils.rnn")
    torch_utils = ModuleType("torch.utils")
    torch_utils_data = ModuleType("torch.utils.data")

    class Dataset:
        pass

    torch.Tensor = object
    torch_nn_utils_rnn.pad_sequence = lambda *args, **kwargs: None
    torch_utils_data.Dataset = Dataset
    torch_nn.utils = torch_nn_utils
    torch_nn_utils.rnn = torch_nn_utils_rnn
    torch.utils = torch_utils
    torch_utils.data = torch_utils_data

    pil = ModuleType("PIL")
    pil_image = ModuleType("PIL.Image")
    pil.Image = pil_image

    transformers = ModuleType("transformers")
    transformers.AutoProcessor = object
    transformers.AutoTokenizer = object

    spec = importlib.util.spec_from_file_location("minicpm_dataset", DATASET_PATH)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "torch": torch,
            "torch.nn": torch_nn,
            "torch.nn.utils": torch_nn_utils,
            "torch.nn.utils.rnn": torch_nn_utils_rnn,
            "torch.utils": torch_utils,
            "torch.utils.data": torch_utils_data,
            "PIL": pil,
            "PIL.Image": pil_image,
            "transformers": transformers,
        },
    ):
        spec.loader.exec_module(module)
    return module


class PreprocessSingleImagePlacementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = load_dataset_module()

    def setUp(self):
        self.tokenizer = Tokenizer()
        self.image_placeholder = "<im_start><unk><unk><im_end>"

    def preprocess_and_capture(self, conversations):
        captured = {}

        def capture_conversation_to_ids(updated_conversations, *args):
            captured["conversations"] = copy.deepcopy(updated_conversations)
            return {"input_ids": [], "position_ids": [], "target": [], "image_bound": []}

        self.dataset.conversation_to_ids = capture_conversation_to_ids
        self.dataset.preprocess(
            {"<image>": object()},
            conversations,
            self.tokenizer,
            transform=lambda image: image,
            query_nums=2,
            max_length=32,
        )
        return captured["conversations"]

    def test_places_single_image_at_later_placeholder_turn(self):
        conversations = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Please inspect <image> now"},
            {"role": "assistant", "content": "Second answer"},
        ]

        updated = self.preprocess_and_capture(conversations)

        self.assertEqual(updated[0]["content"], "First question")
        self.assertEqual(
            updated[2]["content"],
            f"Please inspect {self.image_placeholder} now",
        )
        self.assertEqual(
            "\n".join(message["content"] for message in updated).count(
                self.image_placeholder
            ),
            1,
        )

    def test_keeps_front_fallback_without_placeholder(self):
        conversations = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
        ]

        updated = self.preprocess_and_capture(conversations)

        self.assertEqual(
            updated[0]["content"],
            f"{self.image_placeholder}\nFirst question",
        )


if __name__ == "__main__":
    unittest.main()
