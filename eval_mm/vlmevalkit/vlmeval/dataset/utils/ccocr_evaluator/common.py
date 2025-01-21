import os
import json
import time
import sys
from abc import abstractmethod
from tabulate import tabulate


def pick_response_text(json_path):
    """
    """
    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)
    except Exception as e:
        print("--> file error: msg: {}, path: {}".format(e, json_path))
        return None

    for required_key in ["model_name", "response"]:
        if required_key not in json_data:
            print("--> required key not exists, name: {}, path: {}".format(required_key, json_path))
            return None

    model_name = json_data["model_name"]
    model_response = json_data["response"]

    response_text = None
    if model_name.startswith("gpt") or model_name.startswith("o1"):
        response_text = model_response.get("data", {}).get("response", {}).get("choices", [{}])[0].get("message", {}).get("content", None)  # noqa: E501
    elif model_name.startswith("local_"):
        response_text = model_response
    else:
        if model_name.startswith("claude"):
            content_list = model_response.get("content", None)
        elif model_name.startswith("gemini"):
            content_list = model_response.get("candidates", [{}])[0].get("content", {}).get("parts", None)
        elif model_name.startswith("qwen"):
            content_list = model_response.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", None)  # noqa: E501
        else:
            raise NotImplementedError("The pick_response_text NOT implemented for model: {}".format(model_name))

        if isinstance(content_list, list) and len(content_list) > 0:
            response_text = content_list[0].get("text", None)

    if response_text is None:
        print("--> [error][{}] text pick error, path: {}".format(model_name, json_path))
    return response_text


def load_response_from_dir(res_dir):
    """
    """
    response_info = {}
    for file_name in os.listdir(res_dir):
        file_path = os.path.abspath(os.path.join(res_dir, file_name))
        if not file_name.endswith(".json"):
            print("--> skip: result file should be a json: but got: {}".format(file_path))
            continue

        response_text = pick_response_text(file_path)
        if response_text is None:
            continue

        file_name_wo_ext, ext = os.path.splitext(file_name)
        response_info[file_name_wo_ext] = response_text
    return response_info


class BaseMetric(object):
    """ BaseMetric """
    """ OCRMetric """
    def __init__(self, group_name, **kwargs):
        self.group_name = group_name
        self.kwargs = kwargs

    def response_post_func(self, response_text, **kwargs):
        return response_text

    @abstractmethod
    # Given the prediction and gt, return the evaluation results in the format of a dictionary
    # results should contain a 'summary' key, for example:
    # {
    #     "summary": {
    #         "f1-score": 99.99,
    #         "metric_name": "metric_value"  # used for summary，only metric info could be placed in this dict.
    #     },
    #     "your other info": "xxx"
    # }
    def evaluate(self, response_info, gt_info, normalize_func=None, **kwargs):
        pass

    def __call__(self, pdt_res_dir, gt_info, with_response_ratio=True, **kwargs):
        if isinstance(pdt_res_dir, dict):
            raw_response_info = pdt_res_dir
        elif os.path.exists(pdt_res_dir) and os.path.isdir(pdt_res_dir):
            raw_response_info = load_response_from_dir(pdt_res_dir)
        else:
            return ValueError("invalid input: response dict or folder are required, but got {}".format(pdt_res_dir))

        post_error_list, response_info = [], {}
        response_error_list = list(gt_info.keys() - raw_response_info.keys())
        for file_name, single_pdt_str in raw_response_info.items():
            single_pdt_str = self.response_post_func(single_pdt_str, **kwargs)
            if single_pdt_str is None:
                post_error_list.append(file_name)
                continue
            response_info[file_name] = single_pdt_str

        meta_info = {
            "gt_total_num": len(gt_info), "pdt_total_num": len(response_info),
            "post_error_list": post_error_list, "response_error_list": response_error_list,
        }
        eval_info = self.evaluate(response_info, gt_info, **kwargs)

        # add response_success_ratio
        if "summary" in eval_info and with_response_ratio:
            success_ratio = (len(response_info) + len(post_error_list)) / (len(gt_info) + 1e-9)
            eval_info["summary"].update({"response_success_ratio": success_ratio})
        return meta_info, eval_info


def summary(index_path, exp_dir_base, is_weighted_sum=False):
    """
    """
    with open(index_path, "r") as f:
        data_list = json.load(f)

    all_data_info = {}
    for data_info_item in data_list:
        data_name = data_info_item["dataset"]
        if not data_info_item.get("release", True):
            continue
        all_data_info[data_name] = data_info_item
    dataset_list = list(all_data_info.keys())
    summary_path = summary_multi_exp(exp_dir_base, dataset_list, is_weighted_sum=is_weighted_sum)
    return summary_path


def summary_multi_exp(exp_dir_base, dataset_list=None, is_weighted_sum=False):
    """
    """
    if dataset_list is None:
        all_dataset_name = []
        for exp_name in os.listdir(exp_dir_base):
            dir_status_path = os.path.join(exp_dir_base, exp_name, "status.json")
            if not os.path.exists(dir_status_path):
                continue
            with open(dir_status_path, "r") as f:
                data_status_info = json.load(f)
            all_dataset_name.extend(data_status_info.keys())
        dataset_list = sorted(set(all_dataset_name))

    # summary main code
    all_evaluate_info, _ = {}, 0
    for exp_name in os.listdir(exp_dir_base):
        dir_status_path = os.path.join(exp_dir_base, exp_name, "status.json")
        if not os.path.exists(dir_status_path):
            print("--> skip: status.json not exist: {}".format(dir_status_path))
            continue

        with open(dir_status_path, "r") as f:
            all_status_info = json.load(f)

        for data_name in dataset_list:
            total_num = all_status_info.get(data_name, {}).get("config", {}).get("num", "-1")
            summary_info = all_status_info.get(data_name, {}).get("evaluation", {}).get("summary", {})
            for metric_name, metric_value in summary_info.items():
                if metric_name not in all_evaluate_info:
                    all_evaluate_info[metric_name] = {}
                if exp_name not in all_evaluate_info[metric_name]:
                    all_evaluate_info[metric_name][exp_name] = {}
                all_evaluate_info[metric_name][exp_name][data_name] = (metric_value, total_num)

    all_table_md = []
    for metric_name, metric_info in all_evaluate_info.items():
        formatted_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time()))
        summary_line_list = []
        summary_key_name = "summary(weighted)" if is_weighted_sum else "summary"
        summary_head = [f"exp_name({metric_name}_{formatted_time})"] + dataset_list + [summary_key_name]
        for exp_name, data_eval_info in metric_info.items():
            summary_line = [exp_name, ]

            all_metric_value = 0
            is_summary_valid, all_total_num, all_weighted_metric = True, 0, 0
            for data_name in dataset_list:
                metric_value, total_num = data_eval_info.get(data_name, ("-1", "-1"))
                summary_line.append("{:.2f}".format(float(metric_value) * 100))
                if str(metric_value) == "-1" or str(metric_value) == "-1":
                    is_summary_valid = False
                    continue

                all_total_num += float(total_num)
                all_weighted_metric += float(total_num) * float(metric_value)
                all_metric_value += float(metric_value)

            summary_value_valid = ((all_weighted_metric / (all_total_num + 1e-9)) * 100) if is_weighted_sum \
                else (all_metric_value / (len(dataset_list) + 1e-9) * 100)
            summary_value = "-" if not is_summary_valid else "{:.2f}".format(summary_value_valid)
            summary_line.append(summary_value)
            summary_line_list.append(summary_line)

        md_table_info = tabulate(summary_line_list, headers=summary_head, tablefmt='pipe')
        all_table_md.append(md_table_info)

    print("\n\n".join(all_table_md))
    summary_path = os.path.abspath(os.path.join(exp_dir_base, "summary.md"))
    with open(summary_path, "w") as f:
        f.write("\n\n".join(all_table_md))
    return summary_path


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python {} exp_base_dir".format(__file__))
        exit(-1)
    else:
        print('--> info: {}'.format(sys.argv))
        exp_base_dir = sys.argv[1]

    summary_path = summary_multi_exp(exp_base_dir, dataset_list=None, is_weighted_sum=False)
    print("--> info: summary saved at : {}".format(summary_path))
    print("happy coding.")
