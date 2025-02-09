from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from glob import glob

FAIL_MSG = 'Failed to obtain answer via API.'


def timestamp_to_seconds(timestamp):
    # Split the timestamp into hours, minutes, and seconds
    h, m, s = timestamp.split(":")
    # Convert hours, minutes, and total seconds (including fractions) to float and compute total seconds
    total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
    return total_seconds


def uniformly_subsample(lst, K):
    n = len(lst)
    if K >= n:
        return lst
    step = n / K
    return [lst[int(i * step)] for i in range(K)]


def insert_subtitles_into_frames(
    frames,
    frame_timestamps,
    subtitles,
    starting_timestamp_for_subtitles,
    duration,
):
    interleaved_list = []
    cur_i = 0

    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]

            if not isinstance(end, float):
                end = duration

            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles

            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]

        for i, (frame, frame_timestamp) in enumerate(
            zip(frames[cur_i:], frame_timestamps[cur_i:])
        ):
            if frame_timestamp <= subtitle_timestamp:
                # print("frame:", frame_timestamp)
                interleaved_list.append({"type": "image", "value": frame})
                cur_i += 1
            else:
                break

        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5

        covering_frames = False
        for frame, frame_timestamp in zip(frames, frame_timestamps):
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break

        if covering_frames:
            interleaved_list.append({"type": "text", "value": subtitle_text + "\n"})
        else:
            pass

    for i, (frame, frame_timestamp) in enumerate(
        zip(frames[cur_i:], frame_timestamps[cur_i:])
    ):
        interleaved_list.append({"type": "image", "value": frame})
    return interleaved_list


class LongVideoBench(VideoBaseDataset):

    MD5 = '82905eae3a5ae7383c5a8ee9655e1ab9'
    SYS = ''

    TYPE = 'Video-MCQ'

    def __init__(self, dataset='LongVideoBench', use_subtitle=False, nframe=0, fps=-1):
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)
        self.use_subtitle = use_subtitle
        self.dataset_name = dataset

    @classmethod
    def supported_datasets(cls):
        return ['LongVideoBench']

    def prepare_dataset(self, dataset_name='LongVideoBench', repo_id='longvideobench/LongVideoBench'):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not osp.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                print("md5 mismatch", md5(data_file), self.MD5)
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    print(video_pth, "is not found")
                    return False
            return True

        if modelscope_flag_set():
            repo_id = "AI-ModelScope/LongVideoBench"

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if osp.exists(data_file) and md5(data_file) == self.MD5:
                    return

                data_file = pd.read_json(osp.join(pth, 'lvb_val.json'))
                data_file = data_file.assign(index=range(len(data_file)))
                data_file['video'] = data_file['video_id']
                data_file['video_path'] = data_file['video_path'].apply(lambda x: f'./videos/{x}')

                data_file.to_csv(osp.join(pth, f'{dataset_name}.tsv'), sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_snapshot_download(dataset_id=repo_id)
            else:
                snapshot_download(repo_id=repo_id, repo_type='dataset')
            print("All videos are downloaded for LongVideoBench")

            if not glob(osp.join(cache_path, "videos")):
                tar_files = glob(osp.join(cache_path, "**/*.tar*"), recursive=True)

                def untar_video_data(tar_file, cache_dir):
                    import tarfile
                    with tarfile.open(tar_file, "r") as tar_ref:
                        tar_ref.extractall(cache_dir)
                        print(f"Extracted all files from {tar_file} to {cache_dir}")

                def concat_tar_parts(tar_parts, output_tar):
                    with open(output_tar, "wb") as out_tar:
                        from tqdm import tqdm
                        for part in tqdm(sorted(tar_parts)):
                            with open(part, "rb") as part_file:
                                out_tar.write(part_file.read())
                    print(f"Concatenated parts {tar_parts} into {output_tar}")

                tar_parts_dict = {}

                # Group tar parts together
                for tar_file in tar_files:
                    base_name = tar_file.split(".tar")[0]
                    if base_name not in tar_parts_dict:
                        tar_parts_dict[base_name] = []
                    tar_parts_dict[base_name].append(tar_file)

                # Concatenate and untar split parts
                for base_name, parts in tar_parts_dict.items():
                    print(f"Extracting following tar files: {parts}")
                    output_tar = base_name + ".tar"
                    if not osp.exists(output_tar):
                        print('Start concatenating tar files')

                        concat_tar_parts(parts, output_tar)
                        print('Finish concatenating tar files')

                    if not osp.exists(osp.join(cache_path, osp.basename(base_name))):
                        untar_video_data(output_tar, cache_path)

            print('All videos are extracted for LongVideoBench')

            dataset_path = cache_path
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video_path, video_llm=False):

        vid_path = osp.join(self.data_root, video_path)
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video_path[:-4])
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video_path[:-4], len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth) and not video_llm:
                    im.save(pth)

        return frame_paths, indices, video_info

    # def save_video_into_images(self, line, num_frames=8):
    #     frame_paths, indices, video_info = self.save_video_frames(line['video_path'], num_frames)
    #     return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video_path'], video_llm)
        fps = video_info["fps"]

        message = [dict(type='text', value=self.SYS)]
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, line['video_path'])))
        else:
            if not self.use_subtitle:
                with open(osp.join(self.data_root, "subtitles", line["subtitle_path"])) as f:
                    subtitles = json.load(f)

                frame_message = insert_subtitles_into_frames(
                    frames,
                    [ind_ / fps for ind_ in indices],
                    subtitles,
                    line["starting_timestamp_for_subtitles"],
                    line["duration"]
                )

                message += frame_message
            else:
                for im in frames:
                    message.append(dict(type='image', value=im))

        line['question'] += '\n' + '\n'.join(
            ["{}. {}".format(chr(ord("A") + i), cand) for i, cand in enumerate(eval(line['candidates']))]
        )
        prompt = line["question"] + "\nAnswer with the option's letter from the given choices directly."
        message.append(dict(type='text', value=prompt))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.longvideobench import get_dimension_rating, extract_characters_regex, extract_option

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):
            model = judge_kwargs.get('model', 'exact_matching')
            assert model in ['chatgpt-0125', 'exact_matching', 'gpt-4-0125']

            if model == 'exact_matching':
                model = None
            elif gpt_key_set():
                model = build_judge(**judge_kwargs)
                if not model.working():
                    warnings.warn('OPENAI API is not working properly, will use exact matching for evaluation')
                    warnings.warn(DEBUG_MESSAGE)
                    model = None
            else:
                warnings.warn('OPENAI_API_KEY is not set properly, will use exact matching for evaluation')
                model = None
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if FAIL_MSG not in v}

            data = load(eval_file)
            data_un = data[~pd.isna(data['prediction'])]

            for idx in data['index']:
                ans = data.loc[data['index'] == idx, 'correct_choice'].values[0]
                ans = chr(ord("A") + ans)
                pred = str(data.loc[data['index'] == idx, 'prediction'].values[0])

                if extract_characters_regex(pred) == '':
                    extract_pred = extract_option(
                        model,
                        data.loc[data['index'] == idx].to_dict(orient='records')[0],
                        'LongVideoBench'
                    )
                    data.loc[idx, 'score'] = int(extract_pred == ans)
                else:
                    data.loc[idx, 'score'] = int(extract_characters_regex(pred) == ans)

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
