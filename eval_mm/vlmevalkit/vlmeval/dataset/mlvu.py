import huggingface_hub
from huggingface_hub import snapshot_download
from ..smp import *
from .video_concat_dataset import ConcatVideoDataset
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..utils import track_progress_rich
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
import pandas as pd
import imageio
import cv2
import zipfile
import os
import glob
from .utils.mlvu import *

FAIL_MSG = 'Failed to obtain answer via API.'


class MLVU(ConcatVideoDataset):
    def __init__(self, dataset='MLVU', nframe=0, fps=-1):
        self.DATASET_SETS[dataset] = ['MLVU_MCQ', 'MLVU_OpenEnded']
        self.type_data_dict = {
            'M-Avg':['plotQA', 'needle', 'ego', 'count', 'anomaly_reco', 'topic_reasoning'],
            'G-Avg':['sub_scene', 'summary']
        }
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['MLVU']

    def evaluate(self, eval_file, **judge_kwargs):
        result = super().evaluate(eval_file=eval_file, **judge_kwargs)
        suffix = eval_file.split('.')[-1]
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        for key in self.type_data_dict:
            result.loc[key] = 0.0
            for name, item in result.iterrows():
                if name in self.type_data_dict[key]:
                    result.loc[key, 'success'] += item['success']
                    result.loc[key, 'overall'] += item['overall']
            if key == 'G-Avg':
                result.loc[key, 'acc'] = round(
                    result.loc[key, 'success'] / result.loc[key, 'overall'], 2
                )
            else:
                result.loc[key, 'acc'] = round(
                    result.loc[key, 'success'] / result.loc[key, 'overall'] * 100, 1
                )
        result = result.reset_index().rename(columns={'index': 'task'})
        dump(result, score_file)
        return result


class MLVU_MCQ(VideoBaseDataset):

    MD5 = 'bb5c37e7cf8d43fc9a25c23d2b4633f5'
    BASE_SYS = 'Carefully watch this video and pay attention to every detail. '
    SYS = BASE_SYS + 'Based on your observations, select the best option that accurately addresses the question.'
    TYPE = 'Video-MCQ'

    def __init__(self, dataset='MLVU_MCQ', nframe=0, fps=-1):
        self.type_data_list = {
            'plotQA': ('1_plotQA.json', './MLVU/video/1_plotQA', 'MCQ'),
            'needle': ('2_needle.json', './MLVU/video/2_needle', 'MCQ'),
            'ego': ('3_ego.json', './MLVU/video/3_ego', 'MCQ'),
            'count': ('4_count.json', './MLVU/video/4_count', 'MCQ'),
            'order': ('5_order.json', './MLVU/video/5_order', 'MCQ'),
            'anomaly_reco': ('6_anomaly_reco.json', './MLVU/video/6_anomaly_reco', 'MCQ'),
            'topic_reasoning': ('7_topic_reasoning.json', './MLVU/video/7_topic_reasoning', 'MCQ'),
        }
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['MLVU_MCQ']

    def prepare_dataset(self, dataset_name='MLVU_MCQ', repo_id='MLVU/MVLU'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['prefix'], item['video'])):
                    return False
            return True

        if modelscope_flag_set():
            repo_id = "AI-ModelScope/MLVU"

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return
                json_data_dir = os.path.join(dataset_path, 'MLVU', 'json')
                self.data_list = []
                for k, v in self.type_data_list.items():
                    with open(os.path.join(json_data_dir, v[0]), 'r') as f:
                        json_data = json.load(f)
                    for data in json_data:
                        self.data_list.append({
                            'task_type': k,
                            'prefix': v[1],
                            'duration': data['duration'],
                            'video': data['video'],
                            'question': data['question'],
                            'answer': data['answer'],
                            'candidates': data['candidates'],
                        })

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                hf_token = os.environ.get('HUGGINGFACE_TOKEN')
                huggingface_hub.login(hf_token)
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def qa_template(self, data):
        question = f"Question: {data['question']}\n"
        question += 'Options:\n'
        answer = data['answer']
        answer_idx = -1
        for idx, c in enumerate(eval(data['candidates'])):
            question += f"({chr(ord('A') + idx)}) {c}\n"
            if c == answer:
                answer_idx = idx
        question = question.rstrip()
        answer = f"({chr(ord('A') + answer_idx)}) {answer}"
        return question, answer

    def save_video_frames(self, line):
        suffix = line['video'].split('.')[-1]
        video = line['video'].replace(f'.{suffix}','')
        vid_path = osp.join(self.data_root, line['prefix'], line['video'])
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths

    def save_video_into_images(self, line):
        frame_paths = self.save_video_frames(line)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question, answer = self.qa_template(line)
        message = [dict(type='text', value=self.SYS, role='system')]
        message.append(dict(type='text', value=question))
        video_path = os.path.join(self.data_root, line['prefix'], line['video'])
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(line)
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value='\nOnly give the best option.'))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):
            model = judge_kwargs.setdefault('model', 'chatgpt-0125')
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
                ans = data.loc[data['index'] == idx, 'answer'].values[0]
                pred = data.loc[data['index'] == idx, 'prediction'].values[0]
                options = eval(data.loc[data['index'] == idx, 'candidates'].values[0])
                answer_idx = -1
                for id, c in enumerate(options):
                    if c == ans:
                        answer_idx = id
                ans = f"({chr(ord('A') + answer_idx)}) {ans}"
                input_item = data.loc[data['index'] == idx].to_dict(orient='records')[0]
                for id, option_content in enumerate(eval(input_item['candidates'])):
                    input_item[chr(ord('A') + id)] = option_content
                    if option_content == input_item['answer']:
                        input_item['answer'] = chr(ord('A') + id)

                if FAIL_MSG in pred:
                    data.loc[idx, 'score'] = -1
                else:
                    data.loc[idx, 'score'] = int(check_ans_with_model(
                        pred, ans, model,
                        input_item,
                        'MLVU_MCQ'
                    ))

            rejected = [x for x in data['score'] if x == -1]

            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(data_un)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as -1 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating


class MLVU_OpenEnded(VideoBaseDataset):

    MD5 = 'cee573a3627c6ac434ded704c60511ba'
    BASE_SYS = 'Carefully watch this video and pay attention to every detail. '
    SYS = BASE_SYS + 'Based on your observations, answer the given questions.'
    TYPE = 'Video-VQA'

    def __init__(self, dataset='MLVU_OpenEnded', nframe=0, fps=-1):
        self.type_data_list = {
            'sub_scene': ('8_sub_scene.json', './MLVU/video/8_sub_scene', 'VQA'),
            'summary': ('9_summary.json', './MLVU/video/9_summary', 'VQA')
        }
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['MLVU_OpenEnded']

    def prepare_dataset(self, dataset_name='MLVU_OpenEnded', repo_id='MLVU/MVLU'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['prefix'], item['video'])):
                    return False
            return True

        if modelscope_flag_set():
            repo_id = "AI-ModelScope/MLVU"

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return
                json_data_dir = os.path.join(dataset_path, 'MLVU', 'json')
                self.data_list = []
                for k, v in self.type_data_list.items():
                    with open(os.path.join(json_data_dir, v[0]), 'r') as f:
                        json_data = json.load(f)
                    for data in json_data:
                        self.data_list.append({
                            'task_type': k,
                            'prefix': v[1],
                            'duration': data['duration'],
                            'video': data['video'],
                            'question': data['question'],
                            'answer': data['answer'],
                            'scoring_points': data['scoring_points'] if 'scoring_points' in data else ''
                        })

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                hf_token = os.environ.get('HUGGINGFACE_TOKEN')
                huggingface_hub.login(hf_token)
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')

            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def qa_template(self, data):
        question = f"{data['question']}"
        answer = data['answer']
        return question, answer

    def save_video_frames(self, line):
        suffix = line['video'].split('.')[-1]
        video = line['video'].replace(f'.{suffix}','')
        vid_path = osp.join(self.data_root, line['prefix'], line['video'])
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(video)
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(video, len(indices))

        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths

    def save_video_into_images(self, line):
        frame_paths = self.save_video_frames(line)
        return frame_paths

    def build_prompt(self, line, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question, answer = self.qa_template(line)
        message = [dict(type='text', value=self.SYS, role='system')]
        message.append(dict(type='text', value=question))
        video_path = os.path.join(self.data_root, line['prefix'], line['video'])
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(line)
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):

        model = judge_kwargs['model'] if 'model' in judge_kwargs else judge_kwargs.setdefault('model', 'gpt-4-0125')
        if model != 'gpt-4-0125':
            print('MLVU Open Ended default using gpt-4-0125! So judge model is changed to gpt-4-0125')
            judge_kwargs['model'] = 'gpt-4-0125'

        suffix = eval_file.split('.')[-1]
        score_file = eval_file.replace(f'.{suffix}', f'_{model}_score.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(score_file):
            data = load(eval_file)
            model_dict = {
                'sub_scene': build_judge(system_prompt=system_prompt_sub_scene, **judge_kwargs),
                'summary': build_judge(system_prompt=system_prompt_summary, **judge_kwargs)
            }
            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model_dict[line['task_type']], line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    MLVU_OpenEnded_generate,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
            ans = load(tmp_file)
            data = MLVU_OpenEnded_extract(ans, data)
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating
