from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from .utils.cgbench import *
from ..utils import track_progress_rich


class CGBench_MCQ_Grounding_Mini(VideoBaseDataset):

    dataset = "CG-Bench_MCQ_Grounding_Mini"

    TYPE = "Video-MCQ-Grounding"

    MD5 = "54ed3e90a51a6fb375c92b319a715f72"

    SYS = {
        "long_acc": (
            "You will be provided with sampled frames from a video, along with a "
            "multiple-choice question that includes a question and several answer options.\n"
            "Your task is to analyze the provided frames, infer the most plausible "
            "answer based on the visual information.\n"
            "If the video does not provide enough information, infer the answer based "
            "on the options available and still provide a result. "
            "Therefore, In all cases, an answer must be given.\n"
            "Only output the answer in the following format:\n\n"
            '```json\n{"result": "option"}\n```\n\n'
            'The "option" is the uppercase letter corresponding to your answer.\n\n'
        ),
        "clue_acc": (
            "You will be provided with sampled frames from a video, along with a "
            "multiple-choice question that includes a question and several answer options.\n"
            "Your task is to analyze the provided frames, infer the most plausible "
            "answer based on the visual information.\n"
            "If the video does not provide enough information, infer the answer based "
            "on the options available and still provide a result. "
            "Therefore, In all cases, an answer must be given.\n"
            "Only output the answer in the following format:\n\n"
            '```json\n{"result": "option"}\n```\n\n'
            "The 'option' is the uppercase letter corresponding to your answer.\n\n"
        ),
        "miou": (
            "You will be provided with uniformly sampled frames from a video and their "
            "timestamps, along with a multiple-choice question that includes a question "
            "and several answer options.\n"
            "Your task is to determine in which intervals the 'clue intervals' exist "
            "that contain visual information needed to answer the question.\n"
            "Only output the answer in the following format:\n\n"
            '```json\n{"result": [[start1, end1], [start2, end2], ...]}\n```\n\n'
            "In this output format, each 'start' and 'end' represents the beginning and "
            "end of an interval in seconds where relevant clues can be found.\n"
            "You must provide at least one interval and at most five intervals. "
            "Intervals exceeding five will NOT be considered valid.\n"
        ),
        "miou_wo_frame_time": (
            "You will be provided with uniformly sampled frames from a video, along "
            "with a multiple-choice question that includes a question and several "
            "answer options.\n"
            "Your task is to determine in which intervals the 'clue intervals' exist "
            "that contain visual information needed to answer the question.\n"
            "Only output the answer in the following format:\n\n"
            '```json\n{"result": [[start1, end1], [start2, end2], ...]}\n```\n\n'
            'In this output format, each "start" and "end" represents the start and '
            "end of the video where the relevant clue can be found in the form of a "
            "floating point number between 0 and 1, where 0 represents the start time "
            "of the video and 1 represents the end time of the video.\n"
            "You must provide at least one interval and at most five intervals. "
            "Intervals exceeding five will NOT be considered valid.\n"
        ),
    }

    def __init__(
        self,
        dataset="CG-Bench_MCQ_Grounding_Mini",
        use_subtitle=False,
        use_subtitle_time=False,
        use_frame_time=False,
        nframe=0,
        fps=-1,
    ):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.use_subtitle = use_subtitle
        self.use_subtitle_time = use_subtitle_time
        self.use_frame_time = use_frame_time
        self.dataset_name = dataset
        lmu_root = LMUDataRoot()
        self.clue_frame_root = osp.join(lmu_root, "clue_images", dataset)

    @classmethod
    def supported_datasets(cls):
        return ["CG-Bench_MCQ_Grounding_Mini"]

    def clue_frame_paths(self, qid, num_frames=8):
        frame_root = osp.join(self.clue_frame_root, qid)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def clue_frame_paths_fps(self, qid, num_frames=8, fps=-1):
        frame_root = osp.join(self.clue_frame_root, qid)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl_fps.format(i, num_frames, fps)) for i in range(1, num_frames + 1)]

    def get_subtitles(self, subtitle_path, frame_indices=None, fps=None, sub_time=False):

        subtitles = []

        srt_path = osp.join(self.data_root, subtitle_path)
        assert osp.exists(srt_path)
        import pysubs2

        subs = pysubs2.load(srt_path, encoding="utf-8")
        if not frame_indices:
            for sub in subs:
                sub_text = sub.text.replace("\\N", " ")
                if sub_time:
                    start_time = milliseconds_to_seconds(sub.start)
                    end_time = milliseconds_to_seconds(sub.end)
                    sub_text = f"[{start_time}, {end_time}] {sub_text}"
                if sub_text.strip() and sub_text not in subtitles:
                    subtitles.append(sub_text)
        else:
            for selected_frame_id in frame_indices:
                cur_time = pysubs2.make_time(fps=fps, frames=selected_frame_id)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace("\\N", " ")
                        if sub_time:
                            start_time = milliseconds_to_seconds(sub.start)
                            end_time = milliseconds_to_seconds(sub.end)
                            sub_text = f"[{start_time}, {end_time}] {sub_text}"
                        if sub_text.strip() and sub_text not in subtitles:
                            subtitles.append(sub_text)

        if subtitles:
            subtitles_str = '\n'.join(subtitles)
            return f"The subtitles of the video are as follows:\n\n{subtitles_str}\n\n"
        else:
            return ""

    def prepare_dataset(self, dataset_name="CG-Bench_MCQ_Grounding_Mini", repo_id="CG-Bench/CG-Bench"):

        def check_integrity(pth):
            data_file = osp.join(pth, f"{dataset_name}.tsv")

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data["video"]:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False

            return True

        cache_path = get_cache_path(repo_id)

        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def generate_tsv(pth):

                tsv_file = osp.join(pth, f"{dataset_name}.tsv")

                task_modes = ["long_acc", "clue_acc", "miou"]
                all_data = []
                for task_mode in task_modes:
                    with open(osp.join(pth, "cgbench_mini.json"), "r") as f:
                        data_file = pd.DataFrame(json.load(f))

                    data_file = data_file.assign(index=range(len(data_file)))
                    data_file["video"] = data_file["video_uid"].apply(lambda x: f"cg_videos_720p/{x}.mp4")
                    data_file["subtitle_path"] = data_file["video_uid"].apply(
                        lambda x: (
                            f"cg_subtitles/{x}.srt"
                            if osp.exists(osp.join(dataset_path, f"cg_subtitles/{x}.srt"))
                            else ""
                        )
                    )

                    data_file["clue_video_path"] = ""

                    if task_mode in ["clue_acc"]:
                        data_file["clue_video_path"] = data_file["clue_video_path"] = data_file.apply(
                            lambda row: f"cg_clue_videos/{row['qid']}.mp4", axis=1
                        )

                    data_file["task_mode"] = task_mode

                    if task_mode in ["clue_acc", "long_acc"]:
                        data_file["answer"] = data_file["right_answer"]

                    if task_mode == "miou":
                        data_file["answer"] = data_file["clue_intervals"]

                    if task_mode in ["long_acc", "miou"]:
                        data_file["clue_intervals"] = ""

                    data_file = data_file[
                        [
                            "index",
                            "video_uid",
                            "video",
                            "duration",
                            "domain",
                            "choices",
                            "sub_category",
                            "subtitle_path",
                            "question",
                            "answer",
                            "task_mode",
                            "clue_intervals",
                            "qid",
                            "clue_video_path",
                        ]
                    ]

                    all_data.append(data_file)

                final_data = pd.concat(all_data, ignore_index=True)
                final_data["index"] = range(len(final_data))
                final_data.to_csv(tsv_file, sep="\t", index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download

                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type="dataset")

            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        tsv_file = osp.join(dataset_path, f"{dataset_name}.tsv")

        return dict(data_file=tsv_file, root=dataset_path)

    def build_prompt(self, line, video_llm):

        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        task_mode = line["task_mode"]

        message = []

        origin_use_subtitle_time = self.use_subtitle_time

        try:
            if task_mode in ["long_acc", "clue_acc"]:
                system_prompt = self.SYS[task_mode]
            elif task_mode == "miou":
                if self.use_frame_time and not video_llm:
                    system_prompt = self.SYS[task_mode]
                else:
                    system_prompt = self.SYS["miou_wo_frame_time"]
                    if self.use_subtitle_time is True:
                        self.use_subtitle_time = False

            user_prompt = ""

            if task_mode in ["long_acc", "miou"]:
                video_path = line["video"]

                if video_llm:
                    message.append(dict(type="video", value=osp.join(self.data_root, video_path)))

                    if self.use_subtitle and line["subtitle_path"] and not pd.isna(line["subtitle_path"]):
                        if self.nframe:
                            image_paths, frame_indices, vid_fps = self.save_video_frames(
                                video_path, uid=line["video_uid"], num_frames=self.nframe, fps=self.fps
                            )
                            user_prompt += self.get_subtitles(line["subtitle_path"], frame_indices=frame_indices,
                                                              fps=vid_fps, sub_time=self.use_subtitle_time)
                        else:
                            user_prompt += self.get_subtitles(line["subtitle_path"], sub_time=self.use_subtitle_time)
                else:
                    image_paths, frame_indices, vid_fps = self.save_video_frames(
                        video_path, uid=line["video_uid"], num_frames=self.nframe, fps=self.fps
                    )
                    message.extend(dict(type="image", value=im) for im in image_paths)

                    if self.use_frame_time:
                        user_prompt += get_timestampes(frame_indices, vid_fps)

                    if self.use_subtitle and line["subtitle_path"] and not pd.isna(line["subtitle_path"]):
                        user_prompt += self.get_subtitles(
                            line["subtitle_path"], frame_indices=frame_indices, fps=vid_fps,
                            sub_time=self.use_subtitle_time
                        )

            elif task_mode == "clue_acc":
                clue_video_path = line["clue_video_path"]
                video_path = line["video"]

                if video_llm:
                    message.append(dict(type="video", value=osp.join(self.data_root, clue_video_path)))
                    print(message)

                    if self.use_subtitle and line["subtitle_path"] and not pd.isna(line["subtitle_path"]):
                        if self.nframe:
                            image_paths, frame_indices, vid_fps = self.save_video_frames(
                                video_path, uid=line["video_uid"], num_frames=self.nframe, fps=self.fps
                            )
                            user_prompt += self.get_subtitles(line["subtitle_path"], frame_indices=frame_indices,
                                                              fps=vid_fps, sub_time=self.use_subtitle_time)
                        else:
                            user_prompt += self.get_subtitles(line["subtitle_path"], sub_time=self.use_subtitle_time)
                else:
                    if self.nframe > 32:
                        self.nframe = 32
                        print("The maximum number of frames is 32 when evaluating clue-based mcq in CG-Bench !")

                    clue_intervals = eval(line["clue_intervals"])

                    image_paths, frame_indices, vid_fps = self.save_video_frames(
                        video_path, uid=line["qid"], clue_intervals=clue_intervals, num_frames=self.nframe, fps=self.fps
                    )

                    message.extend(dict(type="image", value=im) for im in image_paths)

                    if self.use_frame_time:
                        user_prompt += get_timestampes(frame_indices, vid_fps)

                    if self.use_subtitle and line["subtitle_path"] and not pd.isna(line["subtitle_path"]):
                        user_prompt += self.get_subtitles(
                            line["subtitle_path"], frame_indices=frame_indices, fps=vid_fps,
                            sub_time=self.use_subtitle_time
                        )

            question = line["question"]
            user_prompt += f"Question: {question}\n\n"

            choices = eval(line["choices"])
            labels = [chr(ord("A") + i) for i in range(len(choices))]
            user_prompt += "\n".join([f"{label}:{value}" for label, value in zip(labels, choices)]) + "\n\n"

            message.append(dict(type="text", value=system_prompt + user_prompt))

            return message

        finally:
            # Ensure that `use_subtitle_time` is always restored to its original value
            self.use_subtitle_time = origin_use_subtitle_time

    def save_video_frames(self, video, uid, clue_intervals=None, num_frames=8, fps=-1):

        if type(uid) is not str:
            uid = str(uid)

        vid_path = osp.join(self.data_root, video)
        vid = decord.VideoReader(vid_path)
        vid_fps = vid.get_avg_fps()
        n_frames = len(vid)

        if clue_intervals is not None:
            merged_intervals = merge_intervals(clue_intervals)

            if num_frames > 0 and fps < 0:
                indices = sample_frames_clue_average(merged_intervals, num_frames, vid_fps)
                frame_paths = self.clue_frame_paths(uid, len(indices))

            elif fps > 0:
                frame_indices = []
                for start, end in merged_intervals:
                    start_frame = int(start * vid_fps)
                    end_frame = int(end * vid_fps)
                    step = vid_fps / fps
                    interval_indices = [
                        int(start_frame + i * step) for i in range(int((end_frame - start_frame) / step))
                    ]
                    frame_indices.extend(interval_indices)

                if len(frame_indices) < 32:
                    indices = sample_frames_clue_average(merged_intervals, 32, vid_fps)
                else:
                    indices = frame_indices
                frame_paths = self.clue_frame_paths_fps(uid, len(indices), fps)

        else:
            if num_frames > 0 and fps < 0:
                step_size = len(vid) / (num_frames + 1)
                indices = [int(i * step_size) for i in range(1, num_frames + 1)]

                frame_paths = self.frame_paths(uid)
            elif fps > 0:
                total_duration = n_frames / vid_fps
                required_frames = int(total_duration * fps)
                step_size = vid_fps / fps
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self.frame_paths_fps(uid, len(indices))

        # Save and validate frames
        valid_paths = []
        valid_indices = []

        if not np.all([osp.exists(p) for p in frame_paths]):
            images = [vid[i].asnumpy() for i in indices]
            for i, (img_array, path) in enumerate(zip(images, frame_paths)):
                if osp.exists(path):
                    try:
                        with Image.open(path) as img:
                            img.verify()
                        valid_paths.append(path)
                        valid_indices.append(indices[i])
                    except Exception:
                        continue
                else:
                    try:
                        img = Image.fromarray(img_array)
                        img.save(path)
                        img.verify()
                        valid_paths.append(path)
                        valid_indices.append(indices[i])
                    except Exception:
                        continue
        else:
            for i, path in enumerate(frame_paths):
                try:
                    with Image.open(path) as img:
                        img.verify()
                    valid_paths.append(path)
                    valid_indices.append(indices[i])
                except Exception:
                    continue

        return valid_paths, valid_indices, vid_fps

    def evaluate(self, eval_file, **judge_kwargs):

        assert eval_file.endswith(".xlsx"), "data file should be an xlsx file"

        tgt_file = eval_file.replace(".xlsx", "_rating.json")
        score_file = eval_file.replace(".xlsx", "_score.xlsx")

        data = load(eval_file)

        data_un = data[~pd.isna(data["prediction"])]
        data_pred_na = data[pd.isna(data["prediction"])]

        data_pred_na["score"] = -1

        data_un["score"] = data_un.apply(
            lambda row: post_process(
                response=row["prediction"],
                right_answer=row["answer"],
                task_mode=row["task_mode"],
                duration=row["duration"],
            ),
            axis=1,
        )

        data = pd.concat([data_pred_na, data_un])

        rejected_count = (data["score"] == -1).sum()

        print(
            f"Among {len(data)} questions, "
            f"failed to obtain prediction for {len(data_pred_na)} questions, "
            f"failed to obtain the score for {rejected_count - len(data_pred_na)} questions. "
            f"Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating."
        )

        dump(data, score_file)

        rating = get_dimention_rating_mcq_grouding(score_file)

        dump(rating, tgt_file)

        return rating


# 评估时，step_2 评估时，给出 [prompt] + image_paths 就行
class CGBench_OpenEnded_Mini(VideoBaseDataset):

    TYPE = "Video-OpenEnded"

    dataset = "CG-Bench_OpenEnded_Mini"

    MD5 = "9175791b11afdfa305fdb3e525b7a4ee"

    SYS = (
        "You will be provided with sampled frames from a video, along with a "
        "question.\n"
        "Your task is to analyze the provided frames and infer the most plausible "
        "answer based on the visual information.\n"
        "If the visual information is ambiguous or insufficient, use the available "
        "context to reason your answer.\n"
        "Only output the answer in the following format:\n\n"
        '```json\n{"result": "answer"}\n```\n\n'
        'The "answer" can be a word, phrase, or sentence that directly responds to '
        "the question.\n\n"
    )

    def __init__(
        self,
        dataset="CG-Bench_OpenEnded_Mini",
        use_subtitle=False,
        use_subtitle_time=False,
        use_frame_time=False,
        nframe=0,
        fps=-1,
    ):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.use_subtitle = use_subtitle
        self.use_subtitle_time = use_subtitle_time
        self.use_frame_time = use_frame_time
        self.dataset_name = dataset
        lmu_root = LMUDataRoot()
        self.clue_frame_root = osp.join(lmu_root, "clue_images", dataset)

    @classmethod
    def supported_datasets(cls):
        return ["CG-Bench_OpenEnded_Mini"]

    def get_subtitles(self, subtitle_path, frame_indices=None, fps=None, sub_time=False):

        subtitles = []

        srt_path = osp.join(self.data_root, subtitle_path)
        assert osp.exists(srt_path)
        import pysubs2

        subs = pysubs2.load(srt_path, encoding="utf-8")
        if not frame_indices:
            for sub in subs:
                sub_text = sub.text.replace("\\N", " ")
                if sub_time:
                    start_time = milliseconds_to_seconds(sub.start)
                    end_time = milliseconds_to_seconds(sub.end)
                    sub_text = f"[{start_time}, {end_time}] {sub_text}"
                if sub_text.strip() and sub_text not in subtitles:
                    subtitles.append(sub_text)
        else:
            for selected_frame_id in frame_indices:
                cur_time = pysubs2.make_time(fps=fps, frames=selected_frame_id)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace("\\N", " ")
                        if sub_time:
                            start_time = milliseconds_to_seconds(sub.start)
                            end_time = milliseconds_to_seconds(sub.end)
                            sub_text = f"[{start_time}, {end_time}] {sub_text}"
                            if sub_text.strip() and sub_text not in subtitles:
                                subtitles.append(sub_text)

        if subtitles:
            subtitles_str = '\n'.join(subtitles)
            return f"The subtitles of the video are as follows:\n\n{subtitles_str}\n\n"
        else:
            return ""

    def prepare_dataset(self, dataset_name="CG-Bench_OpenEnded_Mini", repo_id="CG-Bench/CG-Bench"):

        def check_integrity(pth):
            data_file = osp.join(pth, f"{dataset_name}.tsv")

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data["video"]:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False

            return True

        cache_path = get_cache_path(repo_id)

        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def generate_tsv(pth):

                tsv_file = osp.join(pth, f"{dataset_name}.tsv")

                with open(osp.join(pth, "cgbench_mini.json"), "r") as f:
                    data_file = pd.DataFrame(json.load(f))

                data_file = data_file.assign(index=range(len(data_file)))
                data_file["video"] = data_file["video_uid"].apply(lambda x: f"cg_videos_720p/{x}.mp4")
                data_file["subtitle_path"] = data_file["video_uid"].apply(
                    lambda x: f"cg_subtitles/{x}.srt" if osp.exists(osp.join(pth, f"cg_subtitles/{x}.srt")) else ""
                )

                data_file = data_file[
                    [
                        "index",
                        "video_uid",
                        "video",
                        "duration",
                        "domain",
                        "sub_category",
                        "subtitle_path",
                        "question",
                        "answer",
                        "clue_intervals",
                        "qid",
                    ]
                ]

                data_file.to_csv(tsv_file, sep="\t", index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download

                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type="dataset")

            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        tsv_file = osp.join(dataset_path, f"{dataset_name}.tsv")

        return dict(data_file=tsv_file, root=dataset_path)

    def build_prompt(self, line, video_llm):

        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        message = []

        sys_prompt = self.SYS

        user_prompt = ""

        video_path = line["video"]

        if video_llm:
            message.append(dict(type="video", value=osp.join(self.data_root, video_path)))
            if self.use_subtitle and line["subtitle_path"] and not pd.isna(line["subtitle_path"]):
                if self.nframe:
                    image_paths, frame_indices, vid_fps = self.save_video_frames(
                        video_path, uid=line["video_uid"], num_frames=self.nframe, fps=self.fps
                    )
                    user_prompt += self.get_subtitles(line["subtitle_path"], frame_indices=frame_indices,
                                                      fps=vid_fps, sub_time=self.use_subtitle_time)
                else:
                    user_prompt += self.get_subtitles(line["subtitle_path"], sub_time=self.use_subtitle_time)
        else:
            image_paths, frame_indices, vid_fps = self.save_video_frames(
                video_path, uid=line["video_uid"], num_frames=self.nframe, fps=self.fps
            )
            message.extend(dict(type="image", value=im) for im in image_paths)

            if self.use_frame_time:
                user_prompt += get_timestampes(frame_indices, vid_fps)

            if self.use_subtitle and line["subtitle_path"] and not pd.isna(line["subtitle_path"]):
                user_prompt += self.get_subtitles(
                    line["subtitle_path"], frame_indices=frame_indices, fps=vid_fps,
                    sub_time=self.use_subtitle_time
                )

        question = line["question"]
        user_prompt += f"Question: {question}\n\n"

        message.append(dict(type="text", value=sys_prompt + user_prompt))

        return message

    def clue_frame_paths(self, qid, num_frames=8):
        frame_root = osp.join(self.clue_frame_root, qid)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def save_video_frames(self, video, uid, clue_intervals=None, num_frames=8, fps=-1):

        if type(uid) is not str:
            uid = str(uid)

        vid_path = osp.join(self.data_root, video)
        vid = decord.VideoReader(vid_path)
        vid_fps = vid.get_avg_fps()
        n_frames = len(vid)

        if clue_intervals is not None:
            merged_intervals = merge_intervals(clue_intervals)

            if num_frames > 0 and fps < 0:
                indices = sample_frames_clue_average(merged_intervals, num_frames, vid_fps)
                frame_paths = self.clue_frame_paths(uid, len(indices))

            elif fps > 0:
                frame_indices = []
                for start, end in merged_intervals:
                    start_frame = int(start * vid_fps)
                    end_frame = int(end * vid_fps)
                    step = vid_fps / fps
                    interval_indices = [
                        int(start_frame + i * step) for i in range(int((end_frame - start_frame) / step))
                    ]
                    frame_indices.extend(interval_indices)

                if len(frame_indices) < 32:
                    indices = sample_frames_clue_average(merged_intervals, 32, vid_fps)
                else:
                    indices = frame_indices
                frame_paths = self.clue_frame_paths_fps(uid, len(indices), fps)

        else:
            if num_frames > 0 and fps < 0:
                step_size = len(vid) / (num_frames + 1)
                indices = [int(i * step_size) for i in range(1, num_frames + 1)]
                frame_paths = self.frame_paths(uid)
            elif fps > 0:
                total_duration = n_frames / vid_fps
                required_frames = int(total_duration * fps)
                step_size = vid_fps / fps
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self.frame_paths_fps(uid, len(indices))

        valid_paths = []
        valid_indices = []

        if not np.all([osp.exists(p) for p in frame_paths]):
            images = [vid[i].asnumpy() for i in indices]
            for i, (img_array, path) in enumerate(zip(images, frame_paths)):
                if osp.exists(path):
                    try:
                        with Image.open(path) as img:
                            img.verify()
                        valid_paths.append(path)
                        valid_indices.append(indices[i])
                    except Exception:
                        continue
                else:
                    try:
                        img = Image.fromarray(img_array)
                        img.save(path)
                        img.verify()
                        valid_paths.append(path)
                        valid_indices.append(indices[i])
                    except Exception:
                        continue
        else:
            for i, path in enumerate(frame_paths):
                try:
                    with Image.open(path) as img:
                        img.verify()
                    valid_paths.append(path)
                    valid_indices.append(indices[i])
                except Exception:
                    continue

        return valid_paths, valid_indices, vid_fps

    def evaluate(self, eval_file, **judge_kwargs):

        from .utils.cgbench import get_dimention_rating_open_ended, post_process_open

        assert eval_file.endswith(".xlsx"), "data file should be an xlsx file"

        tgt_file = eval_file.replace(".xlsx", "_rating.json")
        score_file = eval_file.replace(".xlsx", "_score.xlsx")
        step_1_tmp_file = eval_file.replace(".xlsx", "_step_1.pkl")
        step_2_tmp_file = eval_file.replace(".xlsx", "_step_2.pkl")

        data = load(eval_file)

        data_pred_no_na = data[~pd.isna(data["prediction"])]
        data_pred_na = data[pd.isna(data["prediction"])]

        data_pred_na["model_result"] = -1
        data_pred_na["step_1_result"] = -1
        data_pred_na["step_2_result"] = -1
        data_pred_na["score"] = -1

        data_pred_no_na["model_result"] = data_pred_no_na.apply(
            lambda row: post_process_open(
                response=row["prediction"],
            ),
            axis=1,
        )

        data_no_model_result = data_pred_no_na[data_pred_no_na["model_result"] == -1]
        data_step_1 = data_pred_no_na[data_pred_no_na["model_result"] != -1]

        if judge_kwargs.get("model", None) != "gpt-4o-0806":
            judge_kwargs["model"] = "gpt-4o-0806"
            print("The judge model in cg-bench is gpt-4o-0806!")

        model_step_1 = build_judge(system_prompt=sys_prompt_open_eval_step_1, **judge_kwargs)
        nproc = judge_kwargs.pop("nproc", 32)

        lines_step_1 = data_step_1.to_dict("records")
        tups_step_1 = [(model_step_1, line) for line in lines_step_1]

        keys_step_1 = {line["qid"] for line in lines_step_1}

        ans = {}
        if osp.exists(step_1_tmp_file):
            ans = load(step_1_tmp_file)
        tups_step_1 = [x for x, i in zip(tups_step_1, keys_step_1) if i not in ans]
        keys_step_1 = [i for i in keys_step_1 if i not in ans]

        _ = track_progress_rich(
            eval_open_first,
            tups_step_1,
            nproc=nproc,
            keys=keys_step_1,
            save=step_1_tmp_file,
        )

        step_1_results = load(step_1_tmp_file)
        data_step_1 = save_step_1_steps(data_step_1, step_1_results)  # -1, 0, 1, 2

        data_no_step_1_results = data_step_1[data_step_1["step_1_result"] == -1]
        data_step_1_over = data_step_1[data_step_1["step_1_result"].isin([0, 1])]
        data_step_2 = data_step_1[data_step_1["step_1_result"] == 2]

        print(judge_kwargs)

        model_step_2 = build_judge(system_prompt=sys_prompt_open_eval_step_2, **judge_kwargs)

        lines_step_2 = data_step_2.to_dict("records")

        tups_step_2 = []

        for line in tqdm(lines_step_2):
            clue_intervals = eval(line["clue_intervals"])
            lmu_root = LMUDataRoot()
            clue_frame_root = osp.join(lmu_root, "clue_images", self.dataset)
            data_root = self.data_root
            frame_paths, _, _ = save_clue_video_frames(
                data_root,
                clue_frame_root,
                video=line["video"],
                uid=line["qid"],
                clue_intervals=clue_intervals,
                num_frames=32,
            )
            tups_step_2.append((model_step_2, line, frame_paths))

        keys_step_2 = {line["qid"] for line in lines_step_2}

        ans = {}
        if osp.exists(step_2_tmp_file):
            ans = load(step_2_tmp_file)
        tups_step_2 = [x for x, i in zip(tups_step_2, keys_step_2) if i not in ans]
        keys_step_2 = [i for i in keys_step_2 if i not in ans]

        _ = track_progress_rich(
            eval_open_second,
            tups_step_2,
            nproc=nproc,
            keys=keys_step_2,
            save=step_2_tmp_file,
        )

        step_2_results = load(step_2_tmp_file)
        data_step_2 = save_step_2_steps(data_step_2, step_2_results)

        data_no_step_2_results = data_step_2[data_step_2["score"] == -1]
        data_step_2_over = data_step_2[data_step_2["score"].isin([0, 1])]

        data = pd.concat(
            [
                data_pred_na,
                data_no_model_result,
                data_no_step_1_results,
                data_step_1_over,
                data_no_step_2_results,
                data_step_2_over,
            ]
        )

        dump(data, score_file)

        rating = get_dimention_rating_open_ended(score_file)

        dump(rating, tgt_file)

        return rating


class CGBench_MCQ_Grounding(VideoBaseDataset):

    TYPE = "Video-MCQ-Grounding"

    MD5 = "eaead3d978a689269fefce4ae29c86df"

    SYS = {
        "long_acc": (
            "You will be provided with sampled frames from a video, along with a "
            "multiple-choice question that includes a question and several answer options.\n"
            "Your task is to analyze the provided frames, infer the most plausible "
            "answer based on the visual information.\n"
            "If the video does not provide enough information, infer the answer based "
            "on the options available and still provide a result. "
            "Therefore, In all cases, an answer must be given.\n"
            "Only output the answer in the following format:\n\n"
            '```json\n{"result": "option"}\n```\n\n'
            'The "option" is the uppercase letter corresponding to your answer.\n\n'
        ),
        "clue_acc": (
            "You will be provided with sampled frames from a video, along with a "
            "multiple-choice question that includes a question and several answer options.\n"
            "Your task is to analyze the provided frames, infer the most plausible "
            "answer based on the visual information.\n"
            "If the video does not provide enough information, infer the answer based "
            "on the options available and still provide a result. "
            "Therefore, In all cases, an answer must be given.\n"
            "Only output the answer in the following format:\n\n"
            '```json\n{"result": "option"}\n```\n\n'
            "The 'option' is the uppercase letter corresponding to your answer.\n\n"
        ),
        "miou": (
            "You will be provided with uniformly sampled frames from a video and their "
            "timestamps, along with a multiple-choice question that includes a question "
            "and several answer options.\n"
            "Your task is to determine in which intervals the 'clue intervals' exist "
            "that contain visual information needed to answer the question.\n"
            "Only output the answer in the following format:\n\n"
            '```json\n{"result": [[start1, end1], [start2, end2], ...]}\n```\n\n'
            "In this output format, each 'start' and 'end' represents the beginning and "
            "end of an interval in seconds where relevant clues can be found.\n"
            "You must provide at least one interval and at most five intervals. "
            "Intervals exceeding five will NOT be considered valid.\n"
        ),
        "miou_wo_frame_time": (
            "You will be provided with uniformly sampled frames from a video, along "
            "with a multiple-choice question that includes a question and several "
            "answer options.\n"
            "Your task is to determine in which intervals the 'clue intervals' exist "
            "that contain visual information needed to answer the question.\n"
            "Only output the answer in the following format:\n\n"
            '```json\n{"result": [[start1, end1], [start2, end2], ...]}\n```\n\n'
            'In this output format, each "start" and "end" represents the start and '
            "end of the video where the relevant clue can be found in the form of a "
            "floating point number between 0 and 1, where 0 represents the start time "
            "of the video and 1 represents the end time of the video.\n"
            "You must provide at least one interval and at most five intervals. "
            "Intervals exceeding five will NOT be considered valid.\n"
        ),
    }

    def __init__(
        self,
        dataset="CG-Bench_MCQ_Grounding",
        use_subtitle=False,
        use_subtitle_time=False,
        use_frame_time=False,
        nframe=0,
        fps=-1,
    ):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.use_subtitle = use_subtitle
        self.use_subtitle_time = use_subtitle_time
        self.use_frame_time = use_frame_time
        self.dataset_name = dataset
        lmu_root = LMUDataRoot()
        self.clue_frame_root = osp.join(lmu_root, "clue_images", dataset)

    @classmethod
    def supported_datasets(cls):
        return ["CG-Bench_MCQ_Grounding"]

    def clue_frame_paths(self, qid, num_frames=8):
        frame_root = osp.join(self.clue_frame_root, qid)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def clue_frame_paths_fps(self, qid, num_frames=8, fps=-1):
        frame_root = osp.join(self.clue_frame_root, qid)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl_fps.format(i, num_frames, fps)) for i in range(1, num_frames + 1)]

    def get_subtitles(self, subtitle_path, frame_indices=None, fps=None, sub_time=False):

        subtitles = []

        srt_path = osp.join(self.data_root, subtitle_path)
        assert osp.exists(srt_path)
        import pysubs2

        subs = pysubs2.load(srt_path, encoding="utf-8")
        if not frame_indices:
            for sub in subs:
                sub_text = sub.text.replace("\\N", " ")
                if sub_time:
                    start_time = milliseconds_to_seconds(sub.start)
                    end_time = milliseconds_to_seconds(sub.end)
                    sub_text = f"[{start_time}, {end_time}] {sub_text}"
                if sub_text.strip() and sub_text not in subtitles:
                    subtitles.append(sub_text)
        else:
            for selected_frame_id in frame_indices:
                cur_time = pysubs2.make_time(fps=fps, frames=selected_frame_id)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace("\\N", " ")
                        if sub_time:
                            start_time = milliseconds_to_seconds(sub.start)
                            end_time = milliseconds_to_seconds(sub.end)
                            sub_text = f"[{start_time}, {end_time}] {sub_text}"
                        if sub_text.strip() and sub_text not in subtitles:
                            subtitles.append(sub_text)

        if subtitles:
            subtitles_str = '\n'.join(subtitles)
            return f"The subtitles of the video are as follows:\n\n{subtitles_str}\n\n"
        else:
            return ""

    def prepare_dataset(self, dataset_name="CG-Bench_MCQ_Grounding", repo_id="CG-Bench/CG-Bench"):

        def check_integrity(pth):
            data_file = osp.join(pth, f"{dataset_name}.tsv")

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data["video"]:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False

            for clue_video_pth in data["clue_video_path"]:
                if clue_video_pth and not (isinstance(clue_video_pth, float) and np.isnan(clue_video_pth)):
                    if not osp.exists(osp.join(pth, clue_video_pth)):
                        return False

            return True

        cache_path = get_cache_path(repo_id)

        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def generate_tsv(pth):

                tsv_file = osp.join(pth, f"{dataset_name}.tsv")

                task_modes = ["long_acc", "clue_acc", "miou"]
                all_data = []
                for task_mode in task_modes:
                    with open(osp.join(pth, "cgbench.json"), "r") as f:
                        data_file = pd.DataFrame(json.load(f))

                    data_file = data_file.assign(index=range(len(data_file)))
                    data_file["video"] = data_file["video_uid"].apply(lambda x: f"cg_videos_720p/{x}.mp4")
                    data_file["subtitle_path"] = data_file["video_uid"].apply(
                        lambda x: (
                            f"cg_subtitles/{x}.srt"
                            if osp.exists(osp.join(dataset_path, f"cg_subtitles/{x}.srt"))
                            else ""
                        )
                    )

                    data_file["clue_video_path"] = ""

                    if task_mode in ["clue_acc"]:
                        data_file["clue_video_path"] = data_file["clue_video_path"] = data_file.apply(
                            lambda row: f"cg_clue_videos/{row['qid']}.mp4", axis=1
                        )

                    data_file["task_mode"] = task_mode

                    if task_mode in ["clue_acc", "long_acc"]:
                        data_file["answer"] = data_file["right_answer"]

                    if task_mode == "miou":
                        data_file["answer"] = data_file["clue_intervals"]

                    if task_mode in ["long_acc", "miou"]:
                        data_file["clue_intervals"] = ""

                    data_file = data_file[
                        [
                            "index",
                            "video_uid",
                            "video",
                            "duration",
                            "domain",
                            "choices",
                            "sub_category",
                            "subtitle_path",
                            "question",
                            "answer",
                            "task_mode",
                            "clue_intervals",
                            "qid",
                            "clue_video_path",
                        ]
                    ]

                    all_data.append(data_file)

                final_data = pd.concat(all_data, ignore_index=True)
                final_data["index"] = range(len(final_data))
                final_data.to_csv(tsv_file, sep="\t", index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download

                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type="dataset")

            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        tsv_file = osp.join(dataset_path, f"{dataset_name}.tsv")

        return dict(data_file=tsv_file, root=dataset_path)

    def build_prompt(self, line, video_llm):

        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        task_mode = line["task_mode"]

        message = []

        origin_use_subtitle_time = self.use_subtitle_time

        try:
            if task_mode in ["long_acc", "clue_acc"]:
                system_prompt = self.SYS[task_mode]
            elif task_mode == "miou":
                if self.use_frame_time and not video_llm:
                    system_prompt = self.SYS[task_mode]
                else:
                    system_prompt = self.SYS["miou_wo_frame_time"]
                    if self.use_subtitle_time is True:
                        self.use_subtitle_time = False

            user_prompt = ""

            if task_mode in ["long_acc", "miou"]:
                video_path = line["video"]

                if video_llm:
                    message.append(dict(type="video", value=osp.join(self.data_root, video_path)))

                    if self.use_subtitle and line["subtitle_path"] and not pd.isna(line["subtitle_path"]):
                        if self.nframe:
                            image_paths, frame_indices, vid_fps = self.save_video_frames(
                                video_path, uid=line["video_uid"], num_frames=self.nframe, fps=self.fps
                            )
                            user_prompt += self.get_subtitles(line["subtitle_path"], frame_indices=frame_indices,
                                                              fps=vid_fps, sub_time=self.use_subtitle_time)
                        else:
                            user_prompt += self.get_subtitles(line["subtitle_path"], sub_time=self.use_subtitle_time)
                else:
                    image_paths, frame_indices, vid_fps = self.save_video_frames(
                        video_path, uid=line["video_uid"], num_frames=self.nframe, fps=self.fps
                    )
                    message.extend(dict(type="image", value=im) for im in image_paths)

                    if self.use_frame_time:
                        user_prompt += get_timestampes(frame_indices, vid_fps)

                    if self.use_subtitle and line["subtitle_path"] and not pd.isna(line["subtitle_path"]):
                        user_prompt += self.get_subtitles(
                            line["subtitle_path"], frame_indices=frame_indices, fps=vid_fps,
                            sub_time=self.use_subtitle_time
                        )

            elif task_mode == "clue_acc":
                clue_video_path = line["clue_video_path"]
                video_path = line["video"]

                if video_llm:
                    message.append(dict(type="video", value=osp.join(self.data_root, clue_video_path)))
                    print(message)

                    if self.use_subtitle and line["subtitle_path"] and not pd.isna(line["subtitle_path"]):
                        if self.nframe:
                            image_paths, frame_indices, vid_fps = self.save_video_frames(
                                video_path, uid=line["video_uid"], num_frames=self.nframe, fps=self.fps
                            )
                            user_prompt += self.get_subtitles(line["subtitle_path"], frame_indices=frame_indices,
                                                              fps=vid_fps, sub_time=self.use_subtitle_time)
                        else:
                            user_prompt += self.get_subtitles(line["subtitle_path"], sub_time=self.use_subtitle_time)
                else:
                    if self.nframe > 32:
                        self.nframe = 32
                        print("The maximum number of frames is 32 when evaluating clue-based mcq in CG-Bench !")

                    clue_intervals = eval(line["clue_intervals"])

                    image_paths, frame_indices, vid_fps = self.save_video_frames(
                        video_path, uid=line["qid"], clue_intervals=clue_intervals, num_frames=self.nframe, fps=self.fps
                    )

                    message.extend(dict(type="image", value=im) for im in image_paths)

                    if self.use_frame_time:
                        user_prompt += get_timestampes(frame_indices, vid_fps)

                    if self.use_subtitle and line["subtitle_path"] and not pd.isna(line["subtitle_path"]):
                        user_prompt += self.get_subtitles(
                            line["subtitle_path"], frame_indices=frame_indices, fps=vid_fps,
                            sub_time=self.use_subtitle_time
                        )

            question = line["question"]
            user_prompt += f"Question: {question}\n\n"

            choices = eval(line["choices"])
            labels = [chr(ord("A") + i) for i in range(len(choices))]
            user_prompt += "\n".join([f"{label}:{value}" for label, value in zip(labels, choices)]) + "\n\n"

            message.append(dict(type="text", value=system_prompt + user_prompt))

            return message

        finally:
            # Ensure that `use_subtitle_time` is always restored to its original value
            self.use_subtitle_time = origin_use_subtitle_time

    def save_video_frames(self, video, uid, clue_intervals=None, num_frames=8, fps=-1):

        if type(uid) is not str:
            uid = str(uid)

        vid_path = osp.join(self.data_root, video)
        vid = decord.VideoReader(vid_path)
        vid_fps = vid.get_avg_fps()
        n_frames = len(vid)

        if clue_intervals is not None:
            merged_intervals = merge_intervals(clue_intervals)

            if num_frames > 0 and fps < 0:
                indices = sample_frames_clue_average(merged_intervals, num_frames, vid_fps)
                frame_paths = self.clue_frame_paths(uid, len(indices))

            elif fps > 0:
                frame_indices = []
                for start, end in merged_intervals:
                    start_frame = int(start * vid_fps)
                    end_frame = int(end * vid_fps)
                    step = vid_fps / fps
                    interval_indices = [
                        int(start_frame + i * step) for i in range(int((end_frame - start_frame) / step))
                    ]
                    frame_indices.extend(interval_indices)

                if len(frame_indices) < 32:
                    indices = sample_frames_clue_average(merged_intervals, 32, vid_fps)
                else:
                    indices = frame_indices
                frame_paths = self.clue_frame_paths_fps(uid, len(indices), fps)

        else:
            if num_frames > 0 and fps < 0:
                step_size = len(vid) / (num_frames + 1)
                indices = [int(i * step_size) for i in range(1, num_frames + 1)]

                frame_paths = self.frame_paths(uid)
            elif fps > 0:
                total_duration = n_frames / vid_fps
                required_frames = int(total_duration * fps)
                step_size = vid_fps / fps
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self.frame_paths_fps(uid, len(indices))

        # Save and validate frames
        valid_paths = []
        valid_indices = []

        if not np.all([osp.exists(p) for p in frame_paths]):
            images = [vid[i].asnumpy() for i in indices]
            for i, (img_array, path) in enumerate(zip(images, frame_paths)):
                if osp.exists(path):
                    try:
                        with Image.open(path) as img:
                            img.verify()
                        valid_paths.append(path)
                        valid_indices.append(indices[i])
                    except Exception:
                        continue
                else:
                    try:
                        img = Image.fromarray(img_array)
                        img.save(path)
                        img.verify()
                        valid_paths.append(path)
                        valid_indices.append(indices[i])
                    except Exception:
                        continue
        else:
            for i, path in enumerate(frame_paths):
                try:
                    with Image.open(path) as img:
                        img.verify()
                    valid_paths.append(path)
                    valid_indices.append(indices[i])
                except Exception:
                    continue

        return valid_paths, valid_indices, vid_fps

    def evaluate(self, eval_file, **judge_kwargs):

        assert eval_file.endswith(".xlsx"), "data file should be an xlsx file"

        tgt_file = eval_file.replace(".xlsx", "_rating.json")
        score_file = eval_file.replace(".xlsx", "_score.xlsx")

        data = load(eval_file)

        data_un = data[~pd.isna(data["prediction"])]
        data_pred_na = data[pd.isna(data["prediction"])]

        data_pred_na["score"] = -1

        data_un["score"] = data_un.apply(
            lambda row: post_process(
                response=row["prediction"],
                right_answer=row["answer"],
                task_mode=row["task_mode"],
                duration=row["duration"],
            ),
            axis=1,
        )

        data = pd.concat([data_pred_na, data_un])

        rejected_count = (data["score"] == -1).sum()

        print(
            f"Among {len(data)} questions, "
            f"failed to obtain prediction for {len(data_pred_na)} questions, "
            f"failed to obtain the score for {rejected_count - len(data_pred_na)} questions. "
            f"Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating."
        )

        dump(data, score_file)

        rating = get_dimention_rating_mcq_grouding(score_file)

        dump(rating, tgt_file)

        return rating


# 评估时，step_2 评估时，给出 [prompt] + image_paths 就行
class CGBench_OpenEnded(VideoBaseDataset):

    TYPE = "Video-OpenEnded"

    dataset = "CG-Bench_OpenEnded"

    MD5 = "796035eda0b1e916c517cdc1bc145cfc"

    SYS = (
        "You will be provided with sampled frames from a video, along with a "
        "question.\n"
        "Your task is to analyze the provided frames and infer the most plausible "
        "answer based on the visual information.\n"
        "If the visual information is ambiguous or insufficient, use the available "
        "context to reason your answer.\n"
        "Only output the answer in the following format:\n\n"
        '```json\n{"result": "answer"}\n```\n\n'
        'The "answer" can be a word, phrase, or sentence that directly responds to '
        "the question.\n\n"
    )

    def __init__(
        self,
        dataset="CG-Bench_OpenEnded",
        use_subtitle=False,
        use_subtitle_time=False,
        use_frame_time=False,
        nframe=0,
        fps=-1,
    ):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.use_subtitle = use_subtitle
        self.use_subtitle_time = use_subtitle_time
        self.use_frame_time = use_frame_time
        self.dataset_name = dataset
        lmu_root = LMUDataRoot()
        self.clue_frame_root = osp.join(lmu_root, "clue_images", dataset)

    @classmethod
    def supported_datasets(cls):
        return ["CG-Bench_OpenEnded"]

    def get_subtitles(self, subtitle_path, frame_indices=None, fps=None, sub_time=False):

        subtitles = []

        srt_path = osp.join(self.data_root, subtitle_path)
        assert osp.exists(srt_path)
        import pysubs2

        subs = pysubs2.load(srt_path, encoding="utf-8")
        if not frame_indices:
            for sub in subs:
                sub_text = sub.text.replace("\\N", " ")
                if sub_time:
                    start_time = milliseconds_to_seconds(sub.start)
                    end_time = milliseconds_to_seconds(sub.end)
                    sub_text = f"[{start_time}, {end_time}] {sub_text}"
                if sub_text.strip() and sub_text not in subtitles:
                    subtitles.append(sub_text)
        else:
            for selected_frame_id in frame_indices:
                cur_time = pysubs2.make_time(fps=fps, frames=selected_frame_id)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace("\\N", " ")
                        if sub_time:
                            start_time = milliseconds_to_seconds(sub.start)
                            end_time = milliseconds_to_seconds(sub.end)
                            sub_text = f"[{start_time}, {end_time}] {sub_text}"
                            if sub_text.strip() and sub_text not in subtitles:
                                subtitles.append(sub_text)

        if subtitles:
            subtitles_str = '\n'.join(subtitles)
            return f"The subtitles of the video are as follows:\n\n{subtitles_str}\n\n"
        else:
            return ""

    def prepare_dataset(self, dataset_name="CG-Bench_OpenEnded", repo_id="CG-Bench/CG-Bench"):

        def check_integrity(pth):
            data_file = osp.join(pth, f"{dataset_name}.tsv")

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data["video"]:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False

            return True

        cache_path = get_cache_path(repo_id)

        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def generate_tsv(pth):

                tsv_file = osp.join(pth, f"{dataset_name}.tsv")

                with open(osp.join(pth, "cgbench.json"), "r") as f:
                    data_file = pd.DataFrame(json.load(f))

                data_file = data_file.assign(index=range(len(data_file)))
                data_file["video"] = data_file["video_uid"].apply(lambda x: f"cg_videos_720p/{x}.mp4")
                data_file["subtitle_path"] = data_file["video_uid"].apply(
                    lambda x: f"cg_subtitles/{x}.srt" if osp.exists(osp.join(pth, f"cg_subtitles/{x}.srt")) else ""
                )

                data_file = data_file[
                    [
                        "index",
                        "video_uid",
                        "video",
                        "duration",
                        "domain",
                        "sub_category",
                        "subtitle_path",
                        "question",
                        "answer",
                        "clue_intervals",
                        "qid",
                    ]
                ]

                data_file.to_csv(tsv_file, sep="\t", index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type="dataset")

            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        tsv_file = osp.join(dataset_path, f"{dataset_name}.tsv")

        return dict(data_file=tsv_file, root=dataset_path)

    def build_prompt(self, line, video_llm):

        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        message = []

        sys_prompt = self.SYS

        user_prompt = ""

        video_path = line["video"]

        if video_llm:
            message.append(dict(type="video", value=osp.join(self.data_root, video_path)))
            if self.use_subtitle and line["subtitle_path"] and not pd.isna(line["subtitle_path"]):
                if self.nframe:
                    image_paths, frame_indices, vid_fps = self.save_video_frames(
                        video_path, uid=line["video_uid"], num_frames=self.nframe, fps=self.fps
                    )
                    user_prompt += self.get_subtitles(line["subtitle_path"], frame_indices=frame_indices,
                                                      fps=vid_fps, sub_time=self.use_subtitle_time)
                else:
                    user_prompt += self.get_subtitles(line["subtitle_path"], sub_time=self.use_subtitle_time)
        else:
            image_paths, frame_indices, vid_fps = self.save_video_frames(
                video_path, uid=line["video_uid"], num_frames=self.nframe, fps=self.fps
            )
            message.extend(dict(type="image", value=im) for im in image_paths)

            if self.use_frame_time:
                user_prompt += get_timestampes(frame_indices, vid_fps)

            if self.use_subtitle and line["subtitle_path"] and not pd.isna(line["subtitle_path"]):
                user_prompt += self.get_subtitles(
                    line["subtitle_path"], frame_indices=frame_indices, fps=vid_fps,
                    sub_time=self.use_subtitle_time
                )

        question = line["question"]
        user_prompt += f"Question: {question}\n\n"

        message.append(dict(type="text", value=sys_prompt + user_prompt))

        return message

    def clue_frame_paths(self, qid, num_frames=8):
        frame_root = osp.join(self.clue_frame_root, qid)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, num_frames)) for i in range(1, num_frames + 1)]

    def save_video_frames(self, video, uid, clue_intervals=None, num_frames=8, fps=-1):

        if type(uid) is not str:
            uid = str(uid)

        vid_path = osp.join(self.data_root, video)
        vid = decord.VideoReader(vid_path)
        vid_fps = vid.get_avg_fps()
        n_frames = len(vid)

        if clue_intervals is not None:
            merged_intervals = merge_intervals(clue_intervals)

            if num_frames > 0 and fps < 0:
                indices = sample_frames_clue_average(merged_intervals, num_frames, vid_fps)
                frame_paths = self.clue_frame_paths(uid, len(indices))

            elif fps > 0:
                frame_indices = []
                for start, end in merged_intervals:
                    start_frame = int(start * vid_fps)
                    end_frame = int(end * vid_fps)
                    step = vid_fps / fps
                    interval_indices = [
                        int(start_frame + i * step) for i in range(int((end_frame - start_frame) / step))
                    ]
                    frame_indices.extend(interval_indices)

                if len(frame_indices) < 32:
                    indices = sample_frames_clue_average(merged_intervals, 32, vid_fps)
                else:
                    indices = frame_indices
                frame_paths = self.clue_frame_paths_fps(uid, len(indices), fps)

        else:
            if num_frames > 0 and fps < 0:
                step_size = len(vid) / (num_frames + 1)
                indices = [int(i * step_size) for i in range(1, num_frames + 1)]
                frame_paths = self.frame_paths(uid)
            elif fps > 0:
                total_duration = n_frames / vid_fps
                required_frames = int(total_duration * fps)
                step_size = vid_fps / fps
                indices = [int(i * step_size) for i in range(required_frames)]
                frame_paths = self.frame_paths_fps(uid, len(indices))

        valid_paths = []
        valid_indices = []

        if not np.all([osp.exists(p) for p in frame_paths]):
            images = [vid[i].asnumpy() for i in indices]
            for i, (img_array, path) in enumerate(zip(images, frame_paths)):
                if osp.exists(path):
                    try:
                        with Image.open(path) as img:
                            img.verify()
                        valid_paths.append(path)
                        valid_indices.append(indices[i])
                    except Exception:
                        continue
                else:
                    try:
                        img = Image.fromarray(img_array)
                        img.save(path)
                        img.verify()
                        valid_paths.append(path)
                        valid_indices.append(indices[i])
                    except Exception:
                        continue
        else:
            for i, path in enumerate(frame_paths):
                try:
                    with Image.open(path) as img:
                        img.verify()
                    valid_paths.append(path)
                    valid_indices.append(indices[i])
                except Exception:
                    continue

        return valid_paths, valid_indices, vid_fps

    def evaluate(self, eval_file, **judge_kwargs):

        from .utils.cgbench import get_dimention_rating_open_ended, post_process_open

        assert eval_file.endswith(".xlsx"), "data file should be an xlsx file"

        tgt_file = eval_file.replace(".xlsx", "_rating.json")
        score_file = eval_file.replace(".xlsx", "_score.xlsx")
        step_1_tmp_file = eval_file.replace(".xlsx", "_step_1.pkl")
        step_2_tmp_file = eval_file.replace(".xlsx", "_step_2.pkl")

        data = load(eval_file)

        data_pred_no_na = data[~pd.isna(data["prediction"])]
        data_pred_na = data[pd.isna(data["prediction"])]

        data_pred_na["model_result"] = -1
        data_pred_na["step_1_result"] = -1
        data_pred_na["step_2_result"] = -1
        data_pred_na["score"] = -1

        data_pred_no_na["model_result"] = data_pred_no_na.apply(
            lambda row: post_process_open(
                response=row["prediction"],
            ),
            axis=1,
        )

        if judge_kwargs.get("model", None) != "gpt-4o-0806":
            judge_kwargs["model"] = "gpt-4o-0806"
            print("The judge model in cg-bench is gpt-4o-0806!")

        data_no_model_result = data_pred_no_na[data_pred_no_na["model_result"] == -1]
        data_step_1 = data_pred_no_na[data_pred_no_na["model_result"] != -1]

        model_step_1 = build_judge(system_prompt=sys_prompt_open_eval_step_1, **judge_kwargs)
        nproc = judge_kwargs.pop('nproc', 32)

        lines_step_1 = data_step_1.to_dict("records")
        tups_step_1 = [(model_step_1, line) for line in lines_step_1]

        keys_step_1 = {line["qid"] for line in lines_step_1}

        ans = {}
        if osp.exists(step_1_tmp_file):
            ans = load(step_1_tmp_file)
        tups_step_1 = [x for x, i in zip(tups_step_1, keys_step_1) if i not in ans]
        keys_step_1 = [i for i in keys_step_1 if i not in ans]

        _ = track_progress_rich(
            eval_open_first,
            tups_step_1,
            nproc=nproc,
            keys=keys_step_1,
            save=step_1_tmp_file,
        )

        step_1_results = load(step_1_tmp_file)
        data_step_1 = save_step_1_steps(data_step_1, step_1_results)  # -1, 0, 1, 2

        data_no_step_1_results = data_step_1[data_step_1["step_1_result"] == -1]
        data_step_1_over = data_step_1[data_step_1["step_1_result"].isin([0, 1])]
        data_step_2 = data_step_1[data_step_1["step_1_result"] == 2]

        model_step_2 = build_judge(system_prompt=sys_prompt_open_eval_step_2, **judge_kwargs)

        lines_step_2 = data_step_2.to_dict("records")

        tups_step_2 = []

        for line in tqdm(lines_step_2):
            clue_intervals = eval(line["clue_intervals"])
            lmu_root = LMUDataRoot()
            clue_frame_root = osp.join(lmu_root, "clue_images", self.dataset)
            data_root = self.data_root
            frame_paths, _, _ = save_clue_video_frames(
                data_root,
                clue_frame_root,
                video=line["video"],
                uid=line["qid"],
                clue_intervals=clue_intervals,
                num_frames=32,
            )
            tups_step_2.append((model_step_2, line, frame_paths))

        keys_step_2 = {line["qid"] for line in lines_step_2}

        ans = {}
        if osp.exists(step_2_tmp_file):
            ans = load(step_2_tmp_file)
        tups_step_2 = [x for x, i in zip(tups_step_2, keys_step_2) if i not in ans]
        keys_step_2 = [i for i in keys_step_2 if i not in ans]

        _ = track_progress_rich(
            eval_open_second,
            tups_step_2,
            nproc=nproc,
            keys=keys_step_2,
            save=step_2_tmp_file,
        )

        step_2_results = load(step_2_tmp_file)
        data_step_2 = save_step_2_steps(data_step_2, step_2_results)

        data_no_step_2_results = data_step_2[data_step_2["score"] == -1]
        data_step_2_over = data_step_2[data_step_2["score"].isin([0, 1])]

        data = pd.concat(
            [
                data_pred_na,
                data_no_model_result,
                data_no_step_1_results,
                data_step_1_over,
                data_no_step_2_results,
                data_step_2_over,
            ]
        )

        dump(data, score_file)

        rating = get_dimention_rating_open_ended(score_file)

        dump(rating, tgt_file)

        return rating
