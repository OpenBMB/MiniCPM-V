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
from .utils.tempcompass import *


FAIL_MSG = 'Failed to obtain answer via API.'


class TempCompass(ConcatVideoDataset):
    def __init__(self, dataset='TempCompass', nframe=0, fps=-1):
        self.DATASET_SETS[dataset] = ['TempCompass_MCQ', 'TempCompass_Captioning', 'TempCompass_YorN']
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['TempCompass']

    def evaluate(self, eval_file, **judge_kwargs):
        result = super().evaluate(eval_file=eval_file, **judge_kwargs)
        suffix = eval_file.split('.')[-1]
        result = result.reset_index().rename(columns={'index': 'dim.task_type'})
        score_file = eval_file.replace(f'.{suffix}', '_acc.csv')
        avg_dict = {}
        for idx, item in result.iterrows():
            dim, task_type = item['dim.task_type'].split('. ')
            if dim not in avg_dict:
                avg_dict[dim] = {'success': 0.0, 'overall': 0.0}
            if task_type not in avg_dict:
                avg_dict[task_type] = {'success': 0.0, 'overall': 0.0}
            if 'overall' not in avg_dict:
                avg_dict['overall'] = {'success': 0.0, 'overall': 0.0}
            avg_dict[dim]['success'] += item['success']
            avg_dict[dim]['overall'] += item['overall']
            avg_dict[task_type]['success'] += item['success']
            avg_dict[task_type]['overall'] += item['overall']
            avg_dict['overall']['success'] += item['success']
            avg_dict['overall']['overall'] += item['overall']
            result.loc[idx, 'acc'] = round(item['success'] / item['overall'] * 100, 2)
        for key, value in avg_dict.items():
            # 使用 loc 方法添加新行
            result.loc[len(result)] = {
                'dim.task_type': key,
                'success': value['success'],
                'overall': value['overall'],
                'acc': round(value['success'] / value['overall'] * 100, 2)
            }
        dump(result, score_file)
        return result


class TempCompass_MCQ(VideoBaseDataset):

    MD5 = '7efbb9e6d9dabacd22daf274852691dd'
    TYPE = 'Video-MCQ'

    def __init__(self, dataset='TempCompass_MCQ', nframe=0, fps=-1):
        self.type_data_list = {
            'multi-choice': ('multi-choice.json', './videos', '.mp4'),
            'caption_matching': ('caption_matching.json', './videos', '.mp4'),
        }
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['TempCompass_MCQ']

    def prepare_dataset(self, dataset_name='TempCompass_MCQ', repo_id='lmms-lab/TempCompass'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not osp.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['prefix'], item['video'] + item['suffix'])):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def read_parquet(pth):
                import pandas as pd
                for task_name in self.type_data_list.keys():
                    if not osp.exists(osp.join(pth, f'{task_name}.json')):
                        data = pd.read_parquet(osp.join(pth, task_name, 'test-00000-of-00001.parquet'))
                        data.to_json(osp.join(pth, f'{task_name}.json'), orient='records', lines=False)

            def unzip_videos(pth):
                import zipfile
                if not osp.exists(osp.join(pth, 'videos')):
                    zip_file = osp.join(pth, 'tempcompass_videos.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if osp.exists(data_file) and md5(data_file) == self.MD5:
                    return
                self.data_list = []
                for k, v in self.type_data_list.items():
                    with open(osp.join(pth, v[0]), 'r') as f:
                        json_data = json.load(f)
                    for data in json_data:
                        self.data_list.append({
                            'task_type': k,
                            'prefix': v[1],
                            'suffix': v[2],
                            'video': data['video_id'],
                            'question': data['question'].split('\n')[0],
                            'answer': data['answer'],
                            'dim': data['dim'],
                            'candidates': data['question'].split('\n')[1:],
                        })

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            read_parquet(dataset_path)
            unzip_videos(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def qa_template(self, data):
        question = data['question'] + '\n' + '\n'.join(eval(data['candidates']))
        answer = data['answer']
        return question, answer

    def save_video_frames(self, line):
        vid_path = osp.join(self.data_root, line['prefix'], line['video'] + line['suffix'])
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(line['video'])
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(line['video'], len(indices))

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
        message = []
        message.append(dict(type='text', value=question))
        video_path = osp.join(self.data_root, line['prefix'], line['video'] + line['suffix'])
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(line)
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value='\nPlease directly give the best option:'))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-1106', 'exact_matching']
        judge_kwargs.update({
            "max_tokens": 128,
            "temperature": 1.0,
            "top_p": 1,
            "presence_penalty": 1,
        })

        suffix = eval_file.split('.')[-1]
        score_file = eval_file.replace(f'.{suffix}', f'_{model}_score.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(score_file):
            data = load(eval_file)
            if model != 'exact_matching':
                model = build_judge(system_prompt=sys_prompt, **judge_kwargs)
            else:
                model = None

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    evaluate_tempcompass_mcq,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
            ans = load(tmp_file)
            for idx, item in data.iterrows():
                data.loc[idx, 'score'] = ans[idx]['rating']
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating


class TempCompass_Captioning(VideoBaseDataset):

    MD5 = '35be9bf2581ea7767f02e9a8f37ae1ab'
    TYPE = 'Video-VQA'

    def __init__(self, dataset='TempCompass_Captioning', nframe=0, fps=-1):
        self.type_data_list = {
            'captioning': ('captioning.json', './videos', '.mp4'),
        }
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['TempCompass_Captioning']

    def prepare_dataset(self, dataset_name='TempCompass_Captioning', repo_id='lmms-lab/TempCompass'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not osp.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['prefix'], item['video'] + item['suffix'])):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def read_parquet(pth):
                import pandas as pd
                for task_name in self.type_data_list.keys():
                    if not osp.exists(osp.join(pth, f'{task_name}.json')):
                        data = pd.read_parquet(osp.join(pth, task_name, 'test-00000-of-00001.parquet'))
                        data.to_json(osp.join(pth, f'{task_name}.json'), orient='records', lines=False)

            def unzip_videos(pth):
                import zipfile
                if not osp.exists(osp.join(pth, 'videos')):
                    zip_file = osp.join(pth, 'tempcompass_videos.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if osp.exists(data_file) and md5(data_file) == self.MD5:
                    return
                self.data_list = []
                for k, v in self.type_data_list.items():
                    with open(osp.join(pth, v[0]), 'r') as f:
                        json_data = json.load(f)
                    for data in json_data:
                        self.data_list.append({
                            'task_type': k,
                            'prefix': v[1],
                            'suffix': v[2],
                            'video': data['video_id'],
                            'question': data['question'],
                            'answer': data['answer'],
                            'dim': data['dim'],
                            'mc_question': data['mc_question'],
                            'mc_answer': data['mc_answer'],
                        })

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            read_parquet(dataset_path)
            unzip_videos(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def qa_template(self, data):
        question = data['question']
        answer = data['answer']
        return question, answer

    def save_video_frames(self, line):
        vid_path = osp.join(self.data_root, line['prefix'], line['video'] + line['suffix'])
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(line['video'])
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(line['video'], len(indices))

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
        message = []
        message.append(dict(type='text', value=question))
        video_path = osp.join(self.data_root, line['prefix'], line['video'] + line['suffix'])
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(line)
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-1106', 'exact_matching']
        judge_kwargs.update({
            "max_tokens": 128,
            "temperature": 1.0,
            "top_p": 1,
            "presence_penalty": 1,
        })

        suffix = eval_file.split('.')[-1]
        score_file = eval_file.replace(f'.{suffix}', f'_{model}_score.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(score_file):
            data = load(eval_file)
            if model != 'exact_matching':
                model = build_judge(system_prompt=sys_prompt, **judge_kwargs)
            else:
                model = None

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    evaluate_tempcompass_captioning,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
            ans = load(tmp_file)
            for idx, item in data.iterrows():
                data.loc[idx, 'score'] = ans[idx]['rating']
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating


class TempCompass_YorN(VideoBaseDataset):

    MD5 = 'c72c046d7fa0e82c8cd7462f2e844ea8'
    TYPE = 'Video-Y/N'

    def __init__(self, dataset='TempCompass_YorN', nframe=0, fps=-1):
        self.type_data_list = {
            'yes_no': ('yes_no.json', './videos', '.mp4'),
        }
        super().__init__(dataset=dataset, nframe=nframe, fps=fps)

    @classmethod
    def supported_datasets(cls):
        return ['TempCompass_YorN']

    def prepare_dataset(self, dataset_name='TempCompass_YorN', repo_id='lmms-lab/TempCompass'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not osp.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['prefix'], item['video'] + item['suffix'])):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def read_parquet(pth):
                import pandas as pd
                for task_name in self.type_data_list.keys():
                    if not osp.exists(osp.join(pth, f'{task_name}.json')):
                        data = pd.read_parquet(osp.join(pth, task_name, 'test-00000-of-00001.parquet'))
                        data.to_json(osp.join(pth, f'{task_name}.json'), orient='records', lines=False)

            def unzip_videos(pth):
                import zipfile
                if not osp.exists(osp.join(pth, 'videos')):
                    zip_file = osp.join(pth, 'tempcompass_videos.zip')
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(pth)

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if osp.exists(data_file) and md5(data_file) == self.MD5:
                    return
                self.data_list = []
                for k, v in self.type_data_list.items():
                    with open(osp.join(pth, v[0]), 'r') as f:
                        json_data = json.load(f)
                    for data in json_data:
                        self.data_list.append({
                            'task_type': k,
                            'prefix': v[1],
                            'suffix': v[2],
                            'video': data['video_id'],
                            'question': data['question'].split('\n')[0],
                            'answer': data['answer'],
                            'dim': data['dim']
                        })

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            if modelscope_flag_set():
                from modelscope import dataset_snapshot_download
                dataset_path = dataset_snapshot_download(dataset_id=repo_id)
            else:
                dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            read_parquet(dataset_path)
            unzip_videos(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')
        return dict(root=dataset_path, data_file=data_file)

    def qa_template(self, data):
        question = data['question']
        answer = data['answer']
        return question, answer

    def save_video_frames(self, line):
        vid_path = osp.join(self.data_root, line['prefix'], line['video'] + line['suffix'])
        vid = decord.VideoReader(vid_path)
        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }
        if self.nframe > 0 and self.fps < 0:
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            frame_paths = self.frame_paths(line['video'])
        elif self.fps > 0:
            # not constrained by num_frames, get frames by fps
            total_duration = video_info['n_frames'] / video_info['fps']
            required_frames = int(total_duration * self.fps)
            step_size = video_info['fps'] / self.fps
            indices = [int(i * step_size) for i in range(required_frames)]
            frame_paths = self.frame_paths_fps(line['video'], len(indices))

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
        message = []
        message.append(dict(type='text', value=question))
        video_path = osp.join(self.data_root, line['prefix'], line['video'] + line['suffix'])
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            img_frame_paths = self.save_video_into_images(line)
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value='\nPlease answer yes or no:'))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        model = judge_kwargs.get('model', 'exact_matching')
        assert model in ['chatgpt-1106', 'exact_matching']
        judge_kwargs.update({
            "max_tokens": 128,
            "temperature": 1.0,
            "top_p": 1,
            "presence_penalty": 1,
        })

        suffix = eval_file.split('.')[-1]
        score_file = eval_file.replace(f'.{suffix}', f'_{model}_score.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        nproc = judge_kwargs.pop('nproc', 4)

        if not osp.exists(score_file):
            data = load(eval_file)
            if model != 'exact_matching':
                model = build_judge(system_prompt=sys_prompt, **judge_kwargs)
            else:
                model = None

            lt = len(data)
            lines = [data.iloc[i] for i in range(lt)]
            tups = [(model, line) for line in lines]
            indices = [line['index'] for line in lines]

            ans = {}
            if osp.exists(tmp_file):
                ans = load(tmp_file)
            tups = [x for x, i in zip(tups, indices) if i not in ans]
            indices = [i for i in indices if i not in ans]

            if len(indices):
                _ = track_progress_rich(
                    evaluate_tempcompass_YorN,
                    tups,
                    nproc=nproc,
                    chunksize=nproc,
                    keys=indices,
                    save=tmp_file,
                )
            ans = load(tmp_file)
            for idx, item in data.iterrows():
                data.loc[idx, 'score'] = ans[idx]['rating']
            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        return rating
