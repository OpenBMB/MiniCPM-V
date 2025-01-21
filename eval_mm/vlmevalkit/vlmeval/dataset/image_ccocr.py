# flake8: noqa

import os
import re
import tempfile
from functools import partial
import pandas as pd

from .image_base import ImageBaseDataset
from ..smp import *

# should be the same as  FAIL_MSG definded in vlmeval/inference.py
FAIL_MSG = 'Failed to obtain answer via API.'


class CCOCRDataset(ImageBaseDataset):
    TYPE = 'VQA'
    DATASET_URL_MODELSCOPE = {
        "CCOCR_DocParsing_DocPhotoChn": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/doc_parsing/doc/doc_photo_chn_75.tsv",
        "CCOCR_DocParsing_DocPhotoEng": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/doc_parsing/doc/doc_photo_eng_75.tsv",
        "CCOCR_DocParsing_DocScanChn": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/doc_parsing/doc/doc_scan_chn_75.tsv",
        "CCOCR_DocParsing_DocScanEng": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/doc_parsing/doc/doc_scan_eng_75.tsv",
        "CCOCR_DocParsing_TablePhotoChn": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/doc_parsing/table/table_photo_chn_75.tsv",
        "CCOCR_DocParsing_TablePhotoEng": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/doc_parsing/table/table_photo_eng_75.tsv",
        "CCOCR_DocParsing_TableScanChn": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/doc_parsing/table/table_scan_chn_75.tsv",
        "CCOCR_DocParsing_TableScanEng": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/doc_parsing/table/table_scan_eng_75.tsv",
        "CCOCR_DocParsing_MolecularHandwriting": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/doc_parsing/molecular/molecular_handwriting_100.tsv",
        "CCOCR_DocParsing_FormulaHandwriting": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/doc_parsing/formula/formula_handwriting_100.tsv",
        "CCOCR_Kie_Sroie2019Word": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/kie/constrained_category/sroie2019_word_347.tsv",
        "CCOCR_Kie_Cord": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/kie/constrained_category/CORD_100.tsv",
        "CCOCR_Kie_EphoieScut": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/kie/constrained_category/EPHOIE_SCUT_311.tsv",
        "CCOCR_Kie_Poie": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/kie/constrained_category/POIE_250.tsv",
        "CCOCR_Kie_ColdSibr": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/kie/open_category/COLD_SIBR_400.tsv",
        "CCOCR_Kie_ColdCell": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/kie/open_category/COLD_CELL_600.tsv",
        "CCOCR_MultiLanOcr_Arabic": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_lan_ocr/Arabic/Arabic_150.tsv",
        "CCOCR_MultiLanOcr_French": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_lan_ocr/French/French_150.tsv",
        "CCOCR_MultiLanOcr_German": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_lan_ocr/German/German_150.tsv",
        "CCOCR_MultiLanOcr_Italian": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_lan_ocr/Italian/Italian_150.tsv",
        "CCOCR_MultiLanOcr_Japanese": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_lan_ocr/Japanese/Japanese_150.tsv",
        "CCOCR_MultiLanOcr_Korean": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_lan_ocr/Korean/Korean_150.tsv",
        "CCOCR_MultiLanOcr_Portuguese": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_lan_ocr/Portuguese/Portuguese_150.tsv",
        "CCOCR_MultiLanOcr_Russian": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_lan_ocr/Russian/Russian_150.tsv",
        "CCOCR_MultiLanOcr_Spanish": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_lan_ocr/Spanish/Spanish_150.tsv",
        "CCOCR_MultiLanOcr_Vietnamese": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_lan_ocr/Vietnamese/Vietnamese_150.tsv",
        "CCOCR_MultiSceneOcr_Cord": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/document_text/CORD_100.tsv",
        "CCOCR_MultiSceneOcr_Funsd": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/document_text/FUNSD_50.tsv",
        "CCOCR_MultiSceneOcr_Iam": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/document_text/IAM_50.tsv",
        "CCOCR_MultiSceneOcr_ZhDoc": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/document_text/zh_doc_100.tsv",
        "CCOCR_MultiSceneOcr_ZhHandwriting": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/document_text/zh_handwriting_50.tsv",
        "CCOCR_MultiSceneOcr_Hieragent": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/scene_text/Hieragent_100.tsv",
        "CCOCR_MultiSceneOcr_Ic15": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/scene_text/IC15_500.tsv",
        "CCOCR_MultiSceneOcr_Inversetext": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/scene_text/InverseText_500.tsv",
        "CCOCR_MultiSceneOcr_Totaltext": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/scene_text/TotalText_300.tsv",
        "CCOCR_MultiSceneOcr_ZhScene": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/scene_text/zh_scene_450.tsv",
        "CCOCR_MultiSceneOcr_UgcLaion": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/ugc_text/ugc_laion_400.tsv",
        "CCOCR_MultiSceneOcr_ZhDense": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/ugc_text/zh_dense_50.tsv",
        "CCOCR_MultiSceneOcr_ZhVertical": "https://www.modelscope.cn/datasets/Qwen/CC-OCR/resolve/master/multi_scene_ocr/ugc_text/zh_vertical_100.tsv"
    }

    DATASET_URL_HUGGINGFACE = {
        "CCOCR_DocParsing_DocPhotoChn": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/doc_parsing/doc/doc_photo_chn_75.tsv",
        "CCOCR_DocParsing_DocPhotoEng": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/doc_parsing/doc/doc_photo_eng_75.tsv",
        "CCOCR_DocParsing_DocScanChn": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/doc_parsing/doc/doc_scan_chn_75.tsv",
        "CCOCR_DocParsing_DocScanEng": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/doc_parsing/doc/doc_scan_eng_75.tsv",
        "CCOCR_DocParsing_TablePhotoChn": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/doc_parsing/table/table_photo_chn_75.tsv",
        "CCOCR_DocParsing_TablePhotoEng": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/doc_parsing/table/table_photo_eng_75.tsv",
        "CCOCR_DocParsing_TableScanChn": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/doc_parsing/table/table_scan_chn_75.tsv",
        "CCOCR_DocParsing_TableScanEng": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/doc_parsing/table/table_scan_eng_75.tsv",
        "CCOCR_DocParsing_MolecularHandwriting": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/doc_parsing/molecular/molecular_handwriting_100.tsv",
        "CCOCR_DocParsing_FormulaHandwriting": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/doc_parsing/formula/formula_handwriting_100.tsv",
        "CCOCR_Kie_Sroie2019Word": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/kie/constrained_category/sroie2019_word_347.tsv",
        "CCOCR_Kie_Cord": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/kie/constrained_category/CORD_100.tsv",
        "CCOCR_Kie_EphoieScut": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/kie/constrained_category/EPHOIE_SCUT_311.tsv",
        "CCOCR_Kie_Poie": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/kie/constrained_category/POIE_250.tsv",
        "CCOCR_Kie_ColdSibr": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/kie/open_category/COLD_SIBR_400.tsv",
        "CCOCR_Kie_ColdCell": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/kie/open_category/COLD_CELL_600.tsv",
        "CCOCR_MultiLanOcr_Arabic": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_lan_ocr/Arabic/Arabic_150.tsv",
        "CCOCR_MultiLanOcr_French": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_lan_ocr/French/French_150.tsv",
        "CCOCR_MultiLanOcr_German": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_lan_ocr/German/German_150.tsv",
        "CCOCR_MultiLanOcr_Italian": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_lan_ocr/Italian/Italian_150.tsv",
        "CCOCR_MultiLanOcr_Japanese": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_lan_ocr/Japanese/Japanese_150.tsv",
        "CCOCR_MultiLanOcr_Korean": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_lan_ocr/Korean/Korean_150.tsv",
        "CCOCR_MultiLanOcr_Portuguese": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_lan_ocr/Portuguese/Portuguese_150.tsv",
        "CCOCR_MultiLanOcr_Russian": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_lan_ocr/Russian/Russian_150.tsv",
        "CCOCR_MultiLanOcr_Spanish": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_lan_ocr/Spanish/Spanish_150.tsv",
        "CCOCR_MultiLanOcr_Vietnamese": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_lan_ocr/Vietnamese/Vietnamese_150.tsv",
        "CCOCR_MultiSceneOcr_Cord": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/document_text/CORD_100.tsv",
        "CCOCR_MultiSceneOcr_Funsd": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/document_text/FUNSD_50.tsv",
        "CCOCR_MultiSceneOcr_Iam": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/document_text/IAM_50.tsv",
        "CCOCR_MultiSceneOcr_ZhDoc": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/document_text/zh_doc_100.tsv",
        "CCOCR_MultiSceneOcr_ZhHandwriting": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/document_text/zh_handwriting_50.tsv",
        "CCOCR_MultiSceneOcr_Hieragent": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/scene_text/Hieragent_100.tsv",
        "CCOCR_MultiSceneOcr_Ic15": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/scene_text/IC15_500.tsv",
        "CCOCR_MultiSceneOcr_Inversetext": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/scene_text/InverseText_500.tsv",
        "CCOCR_MultiSceneOcr_Totaltext": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/scene_text/TotalText_300.tsv",
        "CCOCR_MultiSceneOcr_ZhScene": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/scene_text/zh_scene_450.tsv",
        "CCOCR_MultiSceneOcr_UgcLaion": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/ugc_text/ugc_laion_400.tsv",
        "CCOCR_MultiSceneOcr_ZhDense": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/ugc_text/zh_dense_50.tsv",
        "CCOCR_MultiSceneOcr_ZhVertical": "https://huggingface.co/datasets/wulipc/CC-OCR/resolve/main/multi_scene_ocr/ugc_text/zh_vertical_100.tsv"
    }

    # define data path
    DATASET_URL = DATASET_URL_MODELSCOPE
    DATASET_MD5 = {
        "CCOCR_DocParsing_DocPhotoChn": "9039dcbb31830d413261a95cfa29d97f",
        "CCOCR_DocParsing_DocPhotoEng": "2ca0824881e1d7317626f2a19d902989",
        "CCOCR_DocParsing_DocScanChn": "9e265c8aa760ebdf5c3bf9e892d55492",
        "CCOCR_DocParsing_DocScanEng": "77d04637be3def86dbc2ce37ba64a704",
        "CCOCR_DocParsing_TablePhotoChn": "c4dc85252ddad2b43a03a67b1d1ae983",
        "CCOCR_DocParsing_TablePhotoEng": "02ab75d6169da0cd2ece9ce0ae14a479",
        "CCOCR_DocParsing_TableScanChn": "f1f79959fdd01127df7377c9d46722f2",
        "CCOCR_DocParsing_TableScanEng": "794903c7acf52bfe956eefba2166d14b",
        "CCOCR_DocParsing_MolecularHandwriting": "30b7f7679b713ce000a939eca7b4078f",
        "CCOCR_DocParsing_FormulaHandwriting": "e03047776ce5e79a61ae1c057e2a348e",
        "CCOCR_Kie_Sroie2019Word": "3287d99a8e86a99b74171fa5a70f9acb",
        "CCOCR_Kie_Cord": "ab297cadcbc7158884a301c366f3330a",
        "CCOCR_Kie_EphoieScut": "bb8fa3ba7ea91cbf17be0904956ad3f3",
        "CCOCR_Kie_Poie": "882b64317989ecbfed6518051cdffb14",
        "CCOCR_Kie_ColdSibr": "109d5dad8b7081fb6a2f088e963196d4",
        "CCOCR_Kie_ColdCell": "7b44c45b4d7d768d1dbdc08872fe7d3a",
        "CCOCR_MultiLanOcr_Arabic": "e9a3f2bb9298d0b882ebc7a98980c3f3",
        "CCOCR_MultiLanOcr_French": "729407ed2036c22e602eff645eddd40c",
        "CCOCR_MultiLanOcr_German": "96fc2edae747f0ec95b0a6f9bf723022",
        "CCOCR_MultiLanOcr_Italian": "29a508fa5d5a5e767497dd69e2430ebb",
        "CCOCR_MultiLanOcr_Japanese": "bbcca96ccf25fff63597c2ab4f3ebb1f",
        "CCOCR_MultiLanOcr_Korean": "0f55dbd24eba5edc189c91e124411641",
        "CCOCR_MultiLanOcr_Portuguese": "a6fcf8831775a61aa631c0cf1c422ae7",
        "CCOCR_MultiLanOcr_Russian": "19d2f84062a1699d3e9333912bd6b303",
        "CCOCR_MultiLanOcr_Spanish": "f5a0cfa9f2ae4115c91c7b362034e591",
        "CCOCR_MultiLanOcr_Vietnamese": "bf1cd4e83d91767f4906f81550cec8b9",
        "CCOCR_MultiSceneOcr_Cord": "92943f0ccb4c5a196c574222e76759a0",
        "CCOCR_MultiSceneOcr_Funsd": "229cc38d193edd00f4383610e98ee873",
        "CCOCR_MultiSceneOcr_Iam": "d897a6d6c3880c65e752ec11b211204c",
        "CCOCR_MultiSceneOcr_ZhDoc": "303682cc16c8bb51b2b896f8ceb8bd38",
        "CCOCR_MultiSceneOcr_ZhHandwriting": "faa298d366bc05e5cfb39e334afb8eff",
        "CCOCR_MultiSceneOcr_Hieragent": "6f132cdd0473d7cc145c3e3a08957dd6",
        "CCOCR_MultiSceneOcr_Ic15": "3d94869f312a41d53d0578a06a2fb1f2",
        "CCOCR_MultiSceneOcr_Inversetext": "e141d424a0c4cf9579064428a270f13d",
        "CCOCR_MultiSceneOcr_Totaltext": "ca1daf81d49eeb57ef844b72a23c2e62",
        "CCOCR_MultiSceneOcr_ZhScene": "9295152a66e6f117db8bfbb20a9013e6",
        "CCOCR_MultiSceneOcr_UgcLaion": "8e9ea1fbf9d56532157e807eabf39b21",
        "CCOCR_MultiSceneOcr_ZhDense": "de8f48ee0c8a2cf8ed7f2b3a81e6322d",
        "CCOCR_MultiSceneOcr_ZhVertical": "4892b4aec6e7fd11e39aaea23712709b"
    }

    # It returns a DataFrame
    def evaluate(self, eval_file, **judge_kwargs):
        """
        """
        df = load(eval_file)
        dict_list = df.to_dict(orient='records')

        required_colume_list = ['answer', 'prediction', "category", "image_name", "l2-category", "split"]
        for required_colume in required_colume_list:
            assert required_colume in df, "required_colume: {} NOT found".format(required_colume)

        gt_info, ptd_info = {}, {}
        for data_info in dict_list:
            image_name = data_info['image_name']
            gt_info[image_name] = data_info['answer']

            # warning the FAIL samples
            if data_info['prediction'] != FAIL_MSG:
                ptd_info[image_name] = data_info['prediction']

        # assert eval_file is a single dataset
        group_name = set([str(x) for x in df['category']]).pop()
        op_name = set([str(x) for x in df['l2-category']]).pop()
        data_name = set([str(x) for x in df['split']]).pop()

        data_info = {"op": op_name, "group": group_name, "dataset": data_name,  "num": len(gt_info)}
        try:
            from .utils.ccocr_evaluator import evaluator_map_info as ccocr_evaluator_map
        except ImportError as err:
            import warnings
            warnings.warn('The dependency of CCOCR evaluator is not properly installed')
            warnings.warn(f'{type(err)}: {err}')
        eval_func = ccocr_evaluator_map.get(group_name, None)
        if eval_func is None:
            raise ValueError("error: evaluator not defined for: {}".format(group_name))
        meta_info, eval_info = eval_func(ptd_info, gt_info, **data_info)

        output_info = {"meta": meta_info, "evaluation": eval_info, "config": data_info}
        result_file = os.path.splitext(os.path.abspath(eval_file))[0] + "_eval.json"
        dump(output_info, result_file)

        # update global status for summary
        # warning: the evaluate function should NOT run in parallel
        all_status_info = {}
        global_status_path = os.path.join(os.path.dirname(eval_file), "status.json")
        if os.path.exists(global_status_path):
            with open(global_status_path, "r") as f:
                all_status_info = json.load(f)
        all_status_info[data_name] = output_info
        with open(global_status_path, "w") as f:
            json.dump(all_status_info, f, ensure_ascii=False, indent=4)
        return eval_info.get("summary")
