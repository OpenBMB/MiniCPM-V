
"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import json
import os
import sys
import re
import time
from typing import Any, Dict, List, Tuple, Union

import zss
from zss import Node
from collections import Counter
from nltk import edit_distance

# local import
from .common import BaseMetric


def flatten(data: dict):
    """
    Convert Dictionary into Non-nested Dictionary
    Example:
        input(dict)
            {
                "menu": [
                    {"name" : ["cake"], "count" : ["2"]},
                    {"name" : ["juice"], "count" : ["1"]},
                ]
            }
        output(list)
            [
                ("menu.name", "cake"),
                ("menu.count", "2"),
                ("menu.name", "juice"),
                ("menu.count", "1"),
            ]
    """
    flatten_data = list()

    def _flatten(value, key=""):
        if type(value) is dict:
            for child_key, child_value in value.items():
                _flatten(child_value, f"{key}.{child_key}" if key else child_key)
        elif type(value) is list:
            for value_item in value:
                _flatten(value_item, key)
        else:
            flatten_data.append((key, value))

    _flatten(data)
    return flatten_data


def update_cost(node1: Node, node2: Node):
    """
    Update cost for tree edit distance.
    If both are leaf node, calculate string edit distance between two labels (special token '<leaf>' will be ignored).
    If one of them is leaf node, cost is length of string in leaf node + 1.
    If neither are leaf node, cost is 0 if label1 is same with label2 othewise 1
    """
    label1 = node1.label
    label2 = node2.label
    label1_leaf = "<leaf>" in label1
    label2_leaf = "<leaf>" in label2
    if label1_leaf and label2_leaf:
        return edit_distance(label1.replace("<leaf>", ""), label2.replace("<leaf>", ""))
    elif not label1_leaf and label2_leaf:
        return 1 + len(label2.replace("<leaf>", ""))
    elif label1_leaf and not label2_leaf:
        return 1 + len(label1.replace("<leaf>", ""))
    else:
        return int(label1 != label2)


def insert_and_remove_cost(node: Node):
    """
    Insert and remove cost for tree edit distance.
    If leaf node, cost is length of label name.
    Otherwise, 1
    """
    label = node.label
    if "<leaf>" in label:
        return len(label.replace("<leaf>", ""))
    else:
        return 1


def normalize_dict(data: Union[Dict, List, Any]):
    """
    Sort by value, while iterate over element if data is list
    """
    # if not data:
    #     return {}

    if isinstance(data, dict):
        new_data = dict()
        for key in sorted(data.keys(), key=lambda k: (len(k), k)):
            value = normalize_dict(data[key])
            if value:
                if not isinstance(value, list):
                    value = [value]
                new_data[key] = value

    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            new_data = []
            for item in data:
                item = normalize_dict(item)
                if item:
                    new_data.append(item)
        else:
            new_data = [str(item).strip() for item in data if type(item) in {str, int, float} and str(item).strip()]
    else:
        new_data = [str(data).strip()]
    return new_data


def cal_f1_all(preds, answers):
    """
    Calculate global F1 accuracy score (field-level, micro-averaged) by counting all true positives,
    false negatives and false positives
    """
    metric_info, error_info = {}, {}
    total_tp, total_fn_or_fp = 0, 0
    for file_name, answer in answers.items():
        sample_error_info = {"fp": [], "fn": [], "tp": []}
        pred = preds.get(file_name, {})
        pred, answer = flatten(normalize_dict(pred)), flatten(normalize_dict(answer))
        for field in pred:
            field_name = field[0]
            if field_name not in metric_info:
                metric_info[field_name] = {"total_tp": 0, "total_fn_or_fp": 0}
            if field in answer:
                total_tp += 1
                metric_info[field_name]["total_tp"] += 1
                sample_error_info["tp"].append(field)
                answer.remove(field)
            else:
                total_fn_or_fp += 1
                metric_info[field_name]["total_fn_or_fp"] += 1
                sample_error_info["fp"].append(field)

        total_fn_or_fp += len(answer)
        for field in answer:
            field_name = field[0]
            if field_name not in metric_info:
                metric_info[field_name] = {"total_tp": 0, "total_fn_or_fp": 0}
            metric_info[field_name]["total_fn_or_fp"] += 1
            sample_error_info["fn"].append(field)

        sample_error_num = sum([len(v) for k, v in sample_error_info.items() if k != "tp"])
        if sample_error_num > 0:
            sample_error_info["error_num"] = sample_error_num
            error_class_list = ["counter_" + x[0] for x in (sample_error_info["fn"] + sample_error_info["fp"])]
            counter = Counter(error_class_list)
            sample_error_info["error_info"] = dict(counter)
            error_info[file_name] = sample_error_info

    # summary
    for field_name, field_info in metric_info.items():
        field_tp, field_fn_or_fp = field_info["total_tp"], field_info["total_fn_or_fp"]
        metric_info[field_name]["acc"] = field_tp / (field_tp + field_fn_or_fp / 2 + 1e-6)

    print("donut_evaluator: total_tp: {}, total_fn_or_fp: {}, ptd_num: {}, gt_num: {}".format(total_tp, total_fn_or_fp,
                                                                                              len(preds), len(answers)))
    error_info = {k: v for k, v in
                  sorted(error_info.items(), key=lambda item: item[1].get("error_num", 0), reverse=True)}
    metric_info = {k: v for k, v in
                   sorted(metric_info.items(), key=lambda item: item[1].get("total_fn_or_fp", 0), reverse=True)}
    return total_tp / (total_tp + total_fn_or_fp / 2 + 1e-6), metric_info, error_info


def construct_tree_from_dict(data: Union[Dict, List], node_name: str = None):
    """
    Convert Dictionary into Tree

    Example:
        input(dict)

            {
                "menu": [
                    {"name" : ["cake"], "count" : ["2"]},
                    {"name" : ["juice"], "count" : ["1"]},
                ]
            }

        output(tree)
                                 <root>
                                   |
                                 menu
                                /    \
                         <subtree>  <subtree>
                        /      |     |      \
                     name    count  name    count
                    /         |     |         \
              <leaf>cake  <leaf>2  <leaf>juice  <leaf>1
     """
    if node_name is None:
        node_name = "<root>"

    node = Node(node_name)

    if isinstance(data, dict):
        for key, value in data.items():
            kid_node = construct_tree_from_dict(value, key)
            node.addkid(kid_node)
    elif isinstance(data, list):
        if all(isinstance(item, dict) for item in data):
            for item in data:
                kid_node = construct_tree_from_dict(
                    item,
                    "<subtree>",
                )
                node.addkid(kid_node)
        else:
            for item in data:
                node.addkid(Node(f"<leaf>{item}"))
    else:
        raise Exception(data, node_name)
    return node


def cal_acc(pred: dict, answer: dict):
    """
    Calculate normalized tree edit distance(nTED) based accuracy.
    1) Construct tree from dict,
    2) Get tree distance with insert/remove/update cost,
    3) Divide distance with GT tree size (i.e., nTED),
    4) Calculate nTED based accuracy. (= max(1 - nTED, 0 ).
    """
    pred = construct_tree_from_dict(normalize_dict(pred))
    answer = construct_tree_from_dict(normalize_dict(answer))
    val1 = zss.distance(
        pred,
        answer,
        get_children=zss.Node.get_children,
        insert_cost=insert_and_remove_cost,
        remove_cost=insert_and_remove_cost,
        update_cost=update_cost,
        return_operations=False,
    )
    val2 = zss.distance(
        construct_tree_from_dict(normalize_dict({})),
        answer,
        get_children=zss.Node.get_children,
        insert_cost=insert_and_remove_cost,
        remove_cost=insert_and_remove_cost,
        update_cost=update_cost,
        return_operations=False,
    )
    return max(0, 1 - val1 / val2)


def cal_acc_all(pred_info, answer_info):
    acc_info, error_info = {}, {}
    for file_name, answer in answer_info.items():
        # if file_name not in pred_info:
        #     print("---> error: pdt not found: {}".format(file_name))
        #     continue
        pred = pred_info.get(file_name, {})
        acc = cal_acc(pred, answer)
        acc_info[file_name] = acc
        if acc < 1.0:
            error_info[file_name] = {"acc": acc, "pred": pred, "answer": answer}

    error_info = {k: v for k, v in sorted(error_info.items(), key=lambda item: item[1].get("acc", 0))}
    acc_averge = sum(list(acc_info.values())) / (len(acc_info) + 1e-6)
    return acc_averge, error_info


def normalize_values_of_nested_dict(d, normalize_func):
    """
    """
    if isinstance(d, dict):
        return {k: normalize_values_of_nested_dict(v, normalize_func) for k, v in d.items()}
    elif isinstance(d, list):
        return [normalize_values_of_nested_dict(x, normalize_func) if isinstance(x, dict) else x for x in d]
    elif isinstance(d, str):
        return normalize_func(d)
    else:
        return d


def eval_donut(pdt_info, gt_info, normalize_func=None, data_name=None):
    """
    """
    if normalize_func is not None:
        print("--> info: normalize_func executed.")
        pdt_info = normalize_values_of_nested_dict(pdt_info, normalize_func)
        gt_info = normalize_values_of_nested_dict(gt_info, normalize_func)

    f1_score, class_eval_info, error_info = cal_f1_all(pdt_info, gt_info)
    acc_average, acc_error_info = cal_acc_all(pdt_info, gt_info)
    eval_info = {"f1_score": f1_score, "acc": acc_average, "class_f1_score": class_eval_info,
                 "f1_error_info": error_info, "acc_error_info": acc_error_info}
    print(data_name, "f1_score", f1_score, "acc", acc_average)
    return eval_info


def post_process_to_json(qwen_info_str, file_name=None):
    try:
        if "```json" in qwen_info_str:
            if "```" not in qwen_info_str:
                qwen_info_str += "```"
            qwen_info_group = re.search(r'```json(.*?)```', qwen_info_str, re.DOTALL)
            json_str = qwen_info_group.group(1).strip().replace("\n", "")
        else:
            json_str = qwen_info_str.strip().replace("\n", "")
        json_data = json.loads(json_str)
        return json_data
    except Exception as err:  # noqa: F841
        return None


def fullwidth_to_halfwidth(text):
    # 全角转半角
    result = ''
    for char in text:
        code_point = ord(char)
        # 全角空格直接转化
        if code_point == 0x3000:
            code_point = 0x0020
        # 其他全角字符（除空格）转换为半角
        elif 0xFF01 <= code_point <= 0xFF5E:
            code_point -= 0xFEE0
        result += chr(code_point)
    result = result.replace("、", ",")
    return result


def remove_unnecessary_spaces(text):
    # 去掉中文字符之间的空格
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])', '', text)
    # 去掉中文和英文、数字之间的空格
    text = re.sub(r'(?<=[\u4e00-\u9fff])\s+(?=[a-zA-Z0-9])', '', text)
    text = re.sub(r'(?<=[a-zA-Z0-9])\s+(?=[\u4e00-\u9fff])', '', text)
    # 去掉符号前的不必要空格，保留符号后的一个空格
    text = re.sub(r'(?<![0-9])\s*([,.!?:;])\s*', r'\1 ', text)  # 非数字前后的符号
    # 在数字和英文之间添加空格
    text = re.sub(r'(?<=[0-9])(?=[a-zA-Z])', ' ', text)
    text = re.sub(r'(?<=[a-zA-Z])(?=[0-9])', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


class KieEvaluator(BaseMetric):
    def response_post_func(self, response_text, **kwargs):
        response_text = post_process_to_json(response_text, file_name=kwargs.get('file_name', None))
        return response_text

    def normalize_func(self, text, **kwargs):
        halfwidth_text = fullwidth_to_halfwidth(str(text))
        cleaned_text = remove_unnecessary_spaces(halfwidth_text)
        return cleaned_text

    def evaluate(self, response_info, gt_info, **kwargs):
        """
        response_info: dict: {"file_name_1": response, "file_name_2": gt}
        gt_info: dict: {"file_name_1": gt, "file_name_2": gt}
        kwargs: dataset index config: {'dataset': 'kie_benchmark_POIE', 'group': 'kie', 'op': 'poie', 'num': 250}
        """
        # gt should be a dict for kie task, fix for VLMEvalKit
        for image_name, label_content in gt_info.items():
            if isinstance(label_content, str):
                gt_info[image_name] = json.loads(label_content)

        response_info = normalize_values_of_nested_dict(response_info, self.normalize_func)
        gt_info = normalize_values_of_nested_dict(gt_info, self.normalize_func)

        f1_score, class_eval_info, error_info = cal_f1_all(response_info, gt_info)
        acc_average, acc_error_info = cal_acc_all(response_info, gt_info)

        # summary info
        summary_info = {"f1_score": f1_score, "acc": acc_average}
        eval_info = {"summary": summary_info, "class_f1_score": class_eval_info,
                     "f1_error_info": error_info, "acc_error_info": acc_error_info}
        return eval_info


if __name__ == '__main__':
    pass
