import os
import sys
import json
import re
from collections import Counter

# local import
from .common import BaseMetric


def token_normalize(token_text, is_lower=False, is_alphanum_only=False):
    """
    """
    if is_lower:
        token_text = token_text.lower()
    if is_alphanum_only:
        token_text = re.sub('[^A-Za-z0-9]+', '', token_text)
    return token_text


def text_normalize_and_tokenize(text, is_keep_blank=True, is_lower=True, is_alphanum_only=False):
    text = text.replace("\t", " ").replace("\n", " ").replace("###", "").replace("***", "")
    text = re.sub(r'\s+', ' ', text)
    if not is_keep_blank:
        text = text.replace(" ", "")
    text_tokens = text.split(" ") if is_keep_blank else list(text)
    text_token_normalized = [token_normalize(t, is_lower, is_alphanum_only) for t in text_tokens]
    text_token_normalized = [x for x in text_token_normalized if len(x) > 0]
    return text_token_normalized


def evaluate_single_sample(gts, preds):
    right_num = 0
    gt_counter_info = dict(Counter(gts))
    pdt_counter_info = dict(Counter(preds))
    for gt_token, gt_count in gt_counter_info.items():
        pred_count = pdt_counter_info.get(gt_token, 0)
        right_num += min(gt_count, pred_count)
    return right_num


def calculate_metrics(response_info, gt_info, is_verbose=False):
    """
    """
    macro_recall_list, macro_precision_list, macro_f1_list = [], [], []
    total_gt_num, total_pred_num, total_right_num = 0, 0, 0
    for file_name, fullbox_gts in gt_info.items():
        fullbox_preds = response_info.get(file_name, [])
        right_num = evaluate_single_sample(fullbox_gts, fullbox_preds)
        total_right_num += right_num
        total_gt_num += len(fullbox_gts)
        total_pred_num += len(fullbox_preds)

        macro_recall = right_num / (len(fullbox_gts) + 1e-9)
        macro_precision = right_num / (len(fullbox_preds) + 1e-9)
        macro_f1 = 2 * macro_recall * macro_precision / (macro_recall + macro_precision + 1e-9)
        macro_recall_list.append(macro_recall)
        macro_precision_list.append(macro_precision)
        macro_f1_list.append(macro_f1)

    # marco
    final_macro_recall = sum(macro_recall_list) / (len(macro_recall_list) + 1e-9)
    final_macro_precision = sum(macro_precision_list) / (len(macro_precision_list) + 1e-9)
    final_macro_f1 = sum(macro_f1_list) / (len(macro_f1_list) + 1e-9)

    # micro
    recall_acc = total_right_num / (total_gt_num + 1e-9)
    preci_acc = total_right_num / (total_pred_num + 1e-9)
    hmean = 2 * recall_acc * preci_acc / (recall_acc + preci_acc + 1e-9)
    vbs_eval_result = {
        'macro_recall': final_macro_recall, 'macro_precision': final_macro_precision, 'macro_f1_score': final_macro_f1,
        'micro_recall': recall_acc, 'micro_precision': preci_acc, 'mirco_f1_score': hmean
    }
    eval_result = vbs_eval_result if is_verbose else {'macro_f1_score': final_macro_f1, 'mirco_f1_score': hmean}
    return eval_result


class OcrEvaluator(BaseMetric):
    def response_post_func(self, response_text, **kwargs):
        return response_text

    def evaluate(self, response_info, gt_info, **kwargs):
        # hard code here
        dataset_name = kwargs['dataset']
        is_word_level, is_lower, is_alphanum_only = True, True, False
        if dataset_name in ["Arabic", "Japanese", "Korean"] or "zh" in dataset_name:
            is_word_level = False
        if "multi_scene_ocr" in self.group_name and is_word_level:
            is_alphanum_only = True
        eval_config = {"word_level": is_word_level, "alphanum_only": is_alphanum_only, "lowercase": is_lower}

        image_pdt_info, image_gt_info = {}, {}
        for file_name, gt_src in gt_info.items():
            pred_src = response_info.get(file_name, "")
            pdt_token_list = text_normalize_and_tokenize(
                str(pred_src).strip(), is_word_level, is_lower, is_alphanum_only)
            gt_token_list = text_normalize_and_tokenize(
                str(gt_src).strip(), is_word_level, is_lower, is_alphanum_only)
            image_pdt_info[file_name] = pdt_token_list
            image_gt_info[file_name] = gt_token_list
        eval_result = calculate_metrics(image_pdt_info, image_gt_info, is_verbose=False)
        return {"summary": eval_result, "metric_config": eval_config}


if __name__ == '__main__':
    pass
