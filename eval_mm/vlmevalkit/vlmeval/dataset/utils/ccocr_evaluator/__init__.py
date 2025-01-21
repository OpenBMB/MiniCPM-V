from .kie_evaluator import KieEvaluator
from .doc_parsing_evaluator import ParsingEvaluator
from .ocr_evaluator import OcrEvaluator
from .common import summary


evaluator_map_info = {
    "kie": KieEvaluator("kie"),
    "doc_parsing": ParsingEvaluator("doc_parsing"),
    "multi_lan_ocr": OcrEvaluator("multi_lan_ocr"),
    "multi_scene_ocr": OcrEvaluator("multi_scene_ocr")
}
