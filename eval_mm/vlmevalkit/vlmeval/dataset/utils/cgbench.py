from ...smp import *
from .multiple_choice import extract_answer_from_item
import pandas as pd
import numpy as np
import re

FAIL_MSG = "Failed to obtain answer via API."

frame_tmpl = "frame-{}-of-{}.jpg"

sys_prompt_open_eval_step_1 = (
    "You will be provided with a question, a model's prediction, and the ground "
    "truth answer for this question.\n"
    "Your task is to judge whether the model's prediction is correct based on the "
    "meaning of the two texts.\n"
    "In most cases, this can be done by determining if the meaning of the model's "
    "prediction is consistent with, or contains, the ground truth answer. However, "
    "in some cases where the two texts differ, it may represent different "
    "descriptions of the same visual scene, in which case visual information is "
    "needed for further judgment.\n"
    "Therefore, I hope you:\n"
    "- Output 0, if the model's prediction and the ground truth answer are neither "
    "consistent nor related by inclusion, with fundamentally different meanings.\n"
    "- Output 1, if the meaning of the model's prediction and the ground truth "
    "answer is consistent, or if the model's prediction meaningfully contains the "
    "ground truth answer.\n"
    "- Output 2, if the model's prediction and ground truth are not consistent or "
    "inclusive, but may be different descriptions of the same visual scene, "
    "requiring visual information for further judgment.\n"
    "Only output the answer in the following format:\n\n"
    '```json\n{"result": choice}\n```\n\n'
    "The choice is either 0, 1, or 2 as specified above."
)

sys_prompt_open_eval_step_2 = (
    "You will be provided with a question, a model's prediction, and the sampling "
    "frames of the clue intervals related to this question.\n"
    "Your task is to determine whether the model has answered the question "
    "correctly based on the visual information provided.\n"
    "Therefore, I hope you:\n"
    "- Output 0, if the model's prediction does not correctly answer the question.\n"
    "- Output 1, if the model's prediction correctly answers the question.\n"
    "Only output the answer in the following format without output extra "
    "explanation:\n\n"
    '```json\n{"result": choice}\n```\n\n'
    "The choice is either 0 or 1 as specified above."
)

FAIL_MSG = "Failed to obtain answer via API."

# '10-20', '20-30', '30-40', '40-50', '50-60'
DURATIONS = ["0 ~ 10", "10 ~ 20", "20 ~ 30", "30 ~ 40", "40 ~ 50", "50 ~ 60", "60+"]

DOMAINS = [
    "Life Record",
    "Music & TV show",
    "Instruction & Knowledge",
    "Driving",
    "Embodied Expert",
    "Humor/funny",
    "Electonic/Social Gaming",
    "Security & Health",
    "Sports & Exercise",
    "Special Scenes",
    "Art & Culture",
    "GUI",
    "News",
    "Animal & Pet",
]

SUB_CATEGORIES = [
    "Time Cognition",
    "Hallucination",
    "Entity Perception",
    "2D Spatial Perception",
    "Time Perception",
    "Scene Perception",
    "Text Perception",
    "Event Cognition",
    "Entity Cognition",
    "Text Cognition",
    "Event Perception",
    "Scene Cognition",
]


def get_dimention_rating_open_ended(data_path):
    # 读取数据
    df = load(data_path)

    df = df[df["score"] != -1]

    # 将秒转换为分钟并分配到对应区间
    df["duration_minutes"] = df["duration"] / 60
    df["duration_range"] = pd.cut(
        df["duration_minutes"], bins=[-np.inf, 10, 20, 30, 40, 50, 60, np.inf], labels=DURATIONS
    )

    # 初始化结果字典
    result = {
        "overall": 0,
        "duration": {k: 0 for k in DURATIONS},
        "domain": {k: 0 for k in DOMAINS},
        "sub_category": {k: 0 for k in SUB_CATEGORIES},
    }

    # Overall
    result["overall"] = round(df["score"].mean(), 4)

    # Duration
    for dur in DURATIONS:
        dur_scores = df[df["duration_range"] == dur]["score"]
        result["duration"][dur] = round(dur_scores.mean(), 4) if not dur_scores.empty else 0

    # Domain
    for domain in DOMAINS:
        domain_scores = df[df["domain"] == domain]["score"]
        result["domain"][domain] = round(domain_scores.mean(), 4) if not domain_scores.empty else 0

    # Sub-category
    for sub_cat in SUB_CATEGORIES:
        sub_cat_scores = df[df["sub_category"] == sub_cat]["score"]
        result["sub_category"][sub_cat] = round(sub_cat_scores.mean(), 4) if not sub_cat_scores.empty else 0

    return result


def get_dimention_rating_mcq_grouding(data_path):

    # 读取数据
    df = load(data_path)

    # df.loc[(df['task_mode'] == 'miou') & (df['score'] == -1), 'score'] = 0

    df = df[df["score"] != -1]

    # 将秒转换为分钟并分配到对应区间
    df["duration_minutes"] = df["duration"] / 60
    df["duration_range"] = pd.cut(
        df["duration_minutes"], bins=[-np.inf, 10, 20, 30, 40, 50, 60, np.inf], labels=DURATIONS
    )

    # 初始化结果字典
    result = {
        metric: {
            "overall": 0,
            "duration": {k: 0 for k in DURATIONS},
            "domain": {k: 0 for k in DOMAINS},
            "sub_category": {k: 0 for k in SUB_CATEGORIES},
        }
        for metric in ["long_acc", "clue_acc", "miou", "CRR", "acc@iou", "rec@iou"]
    }

    # 计算基础指标
    for metric in ["long_acc", "clue_acc", "miou"]:
        metric_df = df[df["task_mode"] == metric]

        # Overall
        result[metric]["overall"] = round(metric_df["score"].mean(), 4)

        # Duration
        for dur in DURATIONS:
            dur_scores = metric_df[metric_df["duration_range"] == dur]["score"]
            result[metric]["duration"][dur] = round(dur_scores.mean(), 4) if not dur_scores.empty else 0

        # Domain
        for domain in DOMAINS:
            domain_scores = metric_df[metric_df["domain"] == domain]["score"]
            result[metric]["domain"][domain] = round(domain_scores.mean(), 4) if not domain_scores.empty else 0

        # Sub-category
        for sub_cat in SUB_CATEGORIES:
            sub_cat_scores = metric_df[metric_df["sub_category"] == sub_cat]["score"]
            result[metric]["sub_category"][sub_cat] = round(sub_cat_scores.mean(), 4) if not sub_cat_scores.empty else 0

    # 计算复合指标 CRR
    def calculate_crr(scores):
        long_acc = scores[scores["task_mode"] == "long_acc"]["score"].mean()
        clue_acc = scores[scores["task_mode"] == "clue_acc"]["score"].mean()
        return round(min(long_acc, clue_acc) / clue_acc, 4) if clue_acc != 0 else 0

    # Overall CRR
    result["CRR"]["overall"] = calculate_crr(df)

    # Duration CRR
    for dur in DURATIONS:
        dur_df = df[df["duration_range"] == dur]
        result["CRR"]["duration"][dur] = calculate_crr(dur_df)

    # Domain CRR
    for domain in DOMAINS:
        domain_df = df[df["domain"] == domain]
        result["CRR"]["domain"][domain] = calculate_crr(domain_df)

    # Sub-category CRR
    for sub_cat in SUB_CATEGORIES:
        sub_cat_df = df[df["sub_category"] == sub_cat]
        result["CRR"]["sub_category"][sub_cat] = calculate_crr(sub_cat_df)

    # 计算 acc@iou
    def calculate_acc_at_iou_threshold(scores, threshold):

        miou_qids = set(scores[scores["task_mode"] == "miou"]["qid"])

        long_acc_qids = set(scores[scores["task_mode"] == "long_acc"]["qid"])

        valid_qids = miou_qids & long_acc_qids

        miou_positive = set(scores[(scores["task_mode"] == "miou") & (scores["score"] > threshold)]["qid"])

        long_acc_positive = scores[
            (scores["task_mode"] == "long_acc") & (scores["qid"].isin(miou_positive)) & (scores["score"] == 1)
        ]

        acc_at_iou_threshold = len(long_acc_positive) / len(valid_qids) if len(valid_qids) > 0 else 0
        return round(acc_at_iou_threshold, 4)

    def calculate_acc_at_iou(scores):
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        acc_at_iou_values = [calculate_acc_at_iou_threshold(scores, threshold) for threshold in thresholds]

        return round(sum(acc_at_iou_values) / len(acc_at_iou_values), 4)

    # Overall acc@iou
    result["acc@iou"]["overall"] = calculate_acc_at_iou(df)

    # Duration acc@iou
    for dur in DURATIONS:
        dur_df = df[df["duration_range"] == dur]
        result["acc@iou"]["duration"][dur] = calculate_acc_at_iou(dur_df)

    # Domain acc@iou
    for domain in DOMAINS:
        domain_df = df[df["domain"] == domain]
        result["acc@iou"]["domain"][domain] = calculate_acc_at_iou(domain_df)

    # Sub-category acc@iou
    for sub_cat in SUB_CATEGORIES:
        sub_cat_df = df[df["sub_category"] == sub_cat]
        result["acc@iou"]["sub_category"][sub_cat] = calculate_acc_at_iou(sub_cat_df)

    # 计算 rec@iou
    def calculate_rec_at_iou_threshold(scores, threshold):
        # 获取所有 miou 类型的数据
        miou_scores = scores[scores["task_mode"] == "miou"]

        # 计算 miou score 大于 threshold 的数量
        miou_positive = miou_scores[miou_scores["score"] > threshold]

        # 计算比例
        rec_at_iou = len(miou_positive) / len(miou_scores) if len(miou_scores) > 0 else 0

        return round(rec_at_iou, 4)

    def calculate_rec_at_iou(scores):
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
        rec_at_iou_values = [calculate_rec_at_iou_threshold(scores, threshold) for threshold in thresholds]

        return round(sum(rec_at_iou_values) / len(rec_at_iou_values), 4)

    # Overall rec@iou
    result["rec@iou"]["overall"] = calculate_rec_at_iou(df)

    # Duration rec@iou
    for dur in DURATIONS:
        dur_df = df[df["duration_range"] == dur]
        result["rec@iou"]["duration"][dur] = calculate_rec_at_iou(dur_df)

    # Domain rec@iou
    for domain in DOMAINS:
        domain_df = df[df["domain"] == domain]
        result["rec@iou"]["domain"][domain] = calculate_rec_at_iou(domain_df)

    # Sub-category rec@iou
    for sub_cat in SUB_CATEGORIES:
        sub_cat_df = df[df["sub_category"] == sub_cat]
        result["rec@iou"]["sub_category"][sub_cat] = calculate_rec_at_iou(sub_cat_df)

    return result


def milliseconds_to_seconds(milliseconds):
    return milliseconds / 1000


def sample_frames_clue_average(clues_time_intervals, frame_num, fps):
    # 计算每个线索区间的时长
    clues_frame_intervals = [(round(interval[0] * fps), round(interval[1] * fps)) for interval in clues_time_intervals]
    clue_durations = [interval[1] - interval[0] for interval in clues_frame_intervals]
    total_duration = sum(clue_durations)
    # 如果 frame_num 的数量大于等于总帧数, 则直接返回全部帧
    if frame_num >= total_duration:
        return [frame for interval in clues_frame_intervals for frame in range(interval[0], interval[1])]
    frames_per_clue = [int(frame_num * (duration / total_duration)) for duration in clue_durations]
    frame_indices = []
    for i, (interval, num_frames) in enumerate(zip(clues_frame_intervals, frames_per_clue)):
        num_frames = max(1, num_frames)
        seg_size = (interval[1] - interval[0]) / num_frames
        clue_frame_indices = [int(interval[0] + seg_size / 2 + seg_size * idx) for idx in range(num_frames)]
        frame_indices.extend(clue_frame_indices)
    return frame_indices


def merge_intervals(intervals):
    """
    Merge overlapping intervals in a list.
    Assumes each interval is a list [start, end].
    """
    if not intervals:
        return []

    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])

    merged = [intervals[0]]

    for current in intervals[1:]:
        last_merged = merged[-1]

        # Check if there is an overlap
        if current[0] <= last_merged[1]:
            # Merge the current interval with the last one
            last_merged[1] = max(last_merged[1], current[1])
        else:
            # No overlap, add current interval
            merged.append(current)

    return merged


def calculate_intervals_iou(intervals1, intervals2):
    """
    Calculate the IoU of two lists of intervals.
    Each list contains intervals represented as [start, end].
    """
    # Merge overlapping intervals in both lists
    merged1 = merge_intervals(intervals1)
    merged2 = merge_intervals(intervals2)

    # Calculate total length of intervals for both lists
    def total_length(merged_intervals):
        return sum(end - start for start, end in merged_intervals)

    length1 = total_length(merged1)
    length2 = total_length(merged2)

    # Calculate intersection length
    intersection_length = 0
    for interval1 in merged1:
        for interval2 in merged2:
            intersection_start = max(interval1[0], interval2[0])
            intersection_end = min(interval1[1], interval2[1])
            intersection_length += max(0, intersection_end - intersection_start)
    # Calculate union length
    union_length = length1 + length2 - intersection_length
    # IoU is intersection divided by union
    iou = intersection_length / union_length if union_length > 0 else 0
    return iou


def post_process(response, right_answer, task_mode, duration):
    result = -1

    if response:
        # 找到 ```json 和 ``` 的位置
        json_start = response.find("```json")
        json_end = response.find("```", json_start + len("```json"))

        # 如果找到了 json 内容
        if json_start != -1 and json_end != -1:
            json_content = response[json_start + len("```json"):json_end].strip()
        else:
            json_content = ""

        if json_content:
            if task_mode in ["long_acc", "clue_acc"]:
                json_content = re.sub(r"(?<=:\s)([A-Za-z_]\w*)", r'"\1"', json_content)

            try:
                model_result = json.loads(json_content)["result"]

                if task_mode in ["long_acc", "clue_acc"]:
                    result = 1 if right_answer == model_result else 0
                elif task_mode == "miou":
                    if not isinstance(model_result, list):
                        return -1
                    if not isinstance(model_result[0], list):
                        model_result = [model_result]

                    need_duration = all(interval[0] <= 1 and interval[1] <= 1 for interval in model_result)

                    if need_duration:
                        model_result = [[interval[0] * duration, interval[1] * duration] for interval in model_result]

                    right_answer = eval(right_answer)

                    result = calculate_intervals_iou(right_answer, model_result)

            except Exception as e:
                print(f"Error in parsing JSON: {e}, {json_content}")

        if result == -1:
            if task_mode in ["long_acc", "clue_acc"]:
                # 检查是否存在大写字母 A-H，认为其为模型答案
                matches = re.findall(r"\b[A-H]\b", response)
                if matches:
                    result = 1 if right_answer in matches else 0
            elif task_mode == "miou":
                # 提取所有实数，进行配对
                numbers = re.findall(r"-?\d+\.?\d*", response)
                if len(numbers) < 2:
                    result = -1
                else:
                    if len(numbers) % 2 != 0:
                        numbers = numbers[:-1]
                    model_result = [[float(numbers[i]), float(numbers[i + 1])] for i in range(0, len(numbers), 2)]

                    if type(right_answer) is str:
                        right_answer = eval(right_answer)

                    result = calculate_intervals_iou(right_answer, model_result)

    return result


def get_timestampes(frame_indices, fps):
    seconds = list(map(lambda x: str(round(x / fps, 4)), frame_indices))
    timestamps = ", ".join(seconds)
    return "A total of {frame_num} frames are sampled. Their corresponding timestamps are:\n\n{timestamps}\n\n".format(
        frame_num=len(frame_indices), timestamps=timestamps
    )


def post_process_open(response):
    model_result = -1

    if response and response != FAIL_MSG:
        json_start = response.find("```json")
        json_end = response.find("```", json_start + len("```json"))

        # 如果找到了 json 内容
        if json_start != -1 and json_end != -1:
            json_content = response[json_start + len("```json"):json_end].strip()
        else:
            json_content = ""

        if json_content:
            try:
                model_result = json.loads(json_content)["result"]
            except Exception as e:
                print(f"Error in parsing JSON: {e}, {json_content}")

        if model_result == -1:
            model_result = response

    return model_result


def post_process_eval_open(response, step):

    model_result = -1

    if response and response != FAIL_MSG:

        json_start = response.find("```json")
        json_end = response.find("```", json_start + len("```json"))

        if json_start != -1 and json_end != -1:
            json_content = response[json_start + len("```json"):json_end].strip()
        else:
            json_content = ""

        if json_content:
            try:
                model_result = json.loads(json_content)["result"]
            except Exception as e:
                print(f"Error in parsing JSON: {e}, {json_content}")
                return -1
        if model_result == -1:
            if step == 1:
                match = re.search(r"[012]", response)
                if match:
                    model_result = int(match.group())
            else:
                match = re.search(r"[01]", response)
                if match:
                    model_result = int(match.group())

    return model_result


def eval_open_first(model, line):

    user_prompt = ""

    user_prompt += f"Question: {line['question']}\n\n"

    user_prompt += f"The ground truth answer is '{line['answer']}'\n\n"

    user_prompt += f"The model's prediction is '{line['model_result']}'\n\n"

    result = model.generate(user_prompt)

    return result


def save_step_1_steps(data, step_1_results):

    # 处理所有结果
    data["step_1_result"] = data["qid"].map(lambda x: post_process_eval_open(step_1_results[x], 1))

    # 条件更新
    mask = data["step_1_result"].isin([-1, 0, 1])
    data.loc[mask, "step_2_result"] = data.loc[mask, "step_1_result"]
    data.loc[mask, "score"] = data.loc[mask, "step_1_result"]

    return data


def eval_open_second(model, line, frame_paths):

    user_prompt = ""

    user_prompt += f"Question: {line['question']}\n\n"

    user_prompt += f"The model's prediction is '{line['model_result']}'\n\n"

    result = model.generate([user_prompt] + frame_paths)

    return result


def save_step_2_steps(data, step_1_results):

    # 处理所有结果
    data["score"] = data["qid"].map(lambda x: post_process_eval_open(step_1_results[x], 2))

    return data


def clue_frame_paths(clue_frame_root, qid, num_frames=8):
    frame_root = osp.join(clue_frame_root, str(qid))
    os.makedirs(frame_root, exist_ok=True)
    return [osp.join(frame_root, frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]


def save_clue_video_frames(data_root, clue_frame_root, video, uid, clue_intervals=None, num_frames=8, fps=-1):

    if type(uid) is str:
        uid = str(uid)

    vid_path = osp.join(data_root, video)
    vid = decord.VideoReader(vid_path)
    vid_fps = vid.get_avg_fps()

    if clue_intervals is not None:
        # 1. 合并重叠区间
        merged_intervals = merge_intervals(clue_intervals)

        if num_frames > 0 and fps < 0:
            # 2. 基于clue_intervals均匀抽帧
            indices = sample_frames_clue_average(merged_intervals, num_frames, vid_fps)
            frame_paths = clue_frame_paths(clue_frame_root, uid, len(indices))

    # 保存帧
    flag = np.all([osp.exists(p) for p in frame_paths])
    if not flag:
        images = [vid[i].asnumpy() for i in indices]
        images = [Image.fromarray(arr) for arr in images]
        for im, pth in zip(images, frame_paths):
            if not osp.exists(pth):
                im.save(pth)

    return frame_paths, indices, vid_fps


def get_chunk_number(filename):
    try:
        num = filename.split("chunk_")[1].split(".zip")[0]
        return int(num)
    except:
        return float('inf')


def unzip_hf_zip(pth):

    import zipfile

    target_dir = pth

    if os.path.exists(f"{target_dir}/cg_videos_720p") and os.path.exists(f"{target_dir}/cg_subtitles")\
            and os.path.exists(f"{target_dir}/cg_clue_videos"):
        print("all exists")
        return

    video_zip_files = [
        os.path.join(target_dir, file)
        for file in os.listdir(target_dir)
        if file.endswith(".zip") and file.startswith("video")
    ]

    video_zip_files = sorted(video_zip_files, key=lambda x: get_chunk_number(os.path.basename(x)))

    videos_temp_zip = os.path.join(target_dir, "videos_merged.zip")

    print("Merging video files ...")

    with open(videos_temp_zip, "wb") as outfile:
        for video_zip_file in tqdm(video_zip_files, desc="Merging videos"):
            with open(video_zip_file, "rb") as infile:
                outfile.write(infile.read())

    print("Extracting video files...")

    try:
        with zipfile.ZipFile(videos_temp_zip, "r") as zip_ref:

            total_files = len(zip_ref.namelist())

            for file in tqdm(zip_ref.namelist(), desc="Extracting", total=total_files):
                zip_ref.extract(file, target_dir)

        print(f"Successfully extracted to {target_dir}")
    except Exception as e:
        print(f"Error during extraction: {e}")
    finally:

        if os.path.exists(videos_temp_zip):
            os.remove(videos_temp_zip)
            print("Cleaned up temporary video file")

    clue_video_zip_files = [
        os.path.join(target_dir, file)
        for file in os.listdir(target_dir)
        if file.endswith(".zip") and file.startswith("clue_video")
    ]

    clue_video_zip_files = sorted(clue_video_zip_files, key=lambda x: get_chunk_number(os.path.basename(x)))

    clue_videos_temp_zip = os.path.join(target_dir, "clue_videos_merged.zip")

    print("Merging clue video files ...")

    with open(clue_videos_temp_zip, "wb") as outfile:
        for clue_video_zip_file in tqdm(clue_video_zip_files, desc="Merging clue_videos"):
            with open(clue_video_zip_file, "rb") as infile:
                outfile.write(infile.read())

    print("Extracting clue video files...")

    try:
        with zipfile.ZipFile(clue_videos_temp_zip, "r") as zip_ref:

            total_files = len(zip_ref.namelist())

            for file in tqdm(zip_ref.namelist(), desc="Extracting", total=total_files):
                zip_ref.extract(file, target_dir)

        print(f"Successfully extracted to {target_dir}")
    except Exception as e:
        print(f"Error during extraction: {e}")
    finally:

        if os.path.exists(clue_videos_temp_zip):
            os.remove(clue_videos_temp_zip)
            print("Cleaned up temporary clue video file")

    print("Extracting subtitle files ...")

    subtitles_zip = os.path.join(target_dir, "subtitles.zip")

    try:
        with zipfile.ZipFile(subtitles_zip, "r") as zip_ref:

            total_files = len(zip_ref.namelist())

            for file in tqdm(zip_ref.namelist(), desc="Extracting", total=total_files):
                zip_ref.extract(file, target_dir)

        print(f"Successfully extracted to {target_dir}")
    except Exception as e:
        print(f"Error during extraction: {e}")
