import huggingface_hub
from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..utils import track_progress_rich
import torchvision.transforms as T
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from decord import VideoReader, cpu
import imageio
import cv2
import zipfile
import os
import glob
from moviepy.editor import VideoFileClip, ImageSequenceClip
import moviepy.config_defaults
from .utils.mvbench import *

FAIL_MSG = 'Failed to obtain answer via API.'
moviepy.config_defaults.LOGGER_LEVEL = logging.CRITICAL + 1


class MVBench(VideoBaseDataset):

    MD5 = 'ae2a2607e2f8618155709220c6e927a6'
    SYS = """Carefully watch the video and pay attention to the cause and sequence of events, \
the detail and movement of objects, and the action and pose of persons. \
Based on your observations, select the best option that accurately addresses the question.
"""

    TYPE = 'MCQ'

    def __init__(self, dataset='MVBench', pack=False):
        self.type_data_list = {
            'Action Sequence': ('action_sequence.json',
                                'your_data_path/star/Charades_v1_480/', 'video', True),  # has start & end
            'Action Prediction': ('action_prediction.json',
                                  'your_data_path/star/Charades_v1_480/', 'video', True),  # has start & end
            'Action Antonym': ('action_antonym.json',
                               'your_data_path/ssv2_video/', 'video', False),
            'Fine-grained Action': ('fine_grained_action.json',
                                    'your_data_path/Moments_in_Time_Raw/videos/', 'video', False),
            'Unexpected Action': ('unexpected_action.json',
                                  'your_data_path/FunQA_test/test/', 'video', False),
            'Object Existence': ('object_existence.json',
                                 'your_data_path/clevrer/video_validation/', 'video', False),
            'Object Interaction': ('object_interaction.json',
                                   'your_data_path/star/Charades_v1_480/', 'video', True),  # has start & end
            'Object Shuffle': ('object_shuffle.json',
                               'your_data_path/perception/videos/', 'video', False),
            'Moving Direction': ('moving_direction.json',
                                 'your_data_path/clevrer/video_validation/', 'video', False),
            'Action Localization': ('action_localization.json',
                                    'your_data_path/sta/sta_video/', 'video', True),   # has start & end
            'Scene Transition': ('scene_transition.json',
                                 'your_data_path/scene_qa/video/', 'video', False),
            'Action Count': ('action_count.json',
                             'your_data_path/perception/videos/', 'video', False),
            'Moving Count': ('moving_count.json',
                             'your_data_path/clevrer/video_validation/', 'video', False),
            'Moving Attribute': ('moving_attribute.json',
                                 'your_data_path/clevrer/video_validation/', 'video', False),
            'State Change': ('state_change.json',
                             'your_data_path/perception/videos/', 'video', False),
            'Fine-grained Pose': ('fine_grained_pose.json',
                                  'your_data_path/nturgbd/', 'video', False),
            'Character Order': ('character_order.json',
                                'your_data_path/perception/videos/', 'video', False),
            'Egocentric Navigation': ('egocentric_navigation.json',
                                      'your_data_path/vlnqa/', 'video', False),
            'Episodic Reasoning': ('episodic_reasoning.json',
                                   'your_data_path/tvqa/frames_fps3_hq/', 'frame', True),  # has start & end, read frame
            'Counterfactual Inference': ('counterfactual_inference.json',
                                         'your_data_path/clevrer/video_validation/', 'video', False),
        }
        super().__init__(dataset=dataset, pack=pack)

    @classmethod
    def supported_datasets(cls):
        return ['MVBench']

    def prepare_dataset(self, dataset_name='MVBench', repo_id='OpenGVLab/MVBench'):
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

        cache_path = get_cache_path(repo_id, branch='main')
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def unzip_hf_zip(pth):
                pth = os.path.join(pth, 'video/')
                for filename in os.listdir(pth):
                    if filename.endswith('.zip'):
                        # 构建完整的文件路径
                        zip_path = os.path.join(pth, filename)

                        # 解压 ZIP 文件
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(pth)

            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return
                json_data_dir = os.path.join(dataset_path, 'json')
                self.data_list = []
                for k, v in self.type_data_list.items():
                    with open(os.path.join(json_data_dir, v[0]), 'r') as f:
                        json_data = json.load(f)
                    for data in json_data:
                        self.data_list.append({
                            'task_type': k,
                            'prefix': v[1].replace('your_data_path', os.path.join(dataset_path, 'video')),
                            'data_type': v[2],
                            'bound': v[3],
                            'start': data['start'] if 'start' in data.keys() else None,
                            'end': data['end'] if 'end' in data.keys() else None,
                            'video': data['video'],
                            'question': data['question'],
                            'answer': data['answer'],
                            'candidates': data['candidates']
                        })

                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            def move_files(pth):
                # special for mvbench
                src_folder = os.path.join(pth, 'video/data0613')
                for subdir in os.listdir(src_folder):
                    subdir_path = os.path.join(src_folder, subdir)
                    if os.path.isdir(subdir_path):
                        for subsubdir in os.listdir(subdir_path):
                            subsubdir_path = os.path.join(subdir_path, subsubdir)
                            if os.path.isdir(subsubdir_path):
                                for item in os.listdir(subsubdir_path):
                                    item_path = os.path.join(subsubdir_path, item)
                                    target_folder = os.path.join(pth, 'video', subdir, subsubdir, item)
                                    if not os.path.exists(target_folder):
                                        shutil.move(item_path, os.path.join(target_folder, item))

            hf_token = os.environ.get('HUGGINGFACE_TOKEN')
            huggingface_hub.login(hf_token)
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            move_files(dataset_path)
            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        self.decord_method = {
            'video': self.read_video,
            'gif': self.read_gif,
            'frame': self.read_frame,
        }

        self.nframe = 8
        self.resolution = 224
        self.frame_fps = 3

        # transform
        crop_size = self.resolution
        scale_size = self.resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std)
        ])

        return dict(root=dataset_path, data_file=data_file)

    def get_index(self, bound, fps, max_frame, first_idx=0):
        if bound:
            start, end = bound[0], bound[1]
        else:
            start, end = -100000, 100000
        start_idx = max(first_idx, round(start * fps))
        end_idx = min(round(end * fps), max_frame)
        seg_size = float(end_idx - start_idx) / self.num_segments
        frame_indices = np.array([
            int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1
        fps = float(vr.get_avg_fps())

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def read_gif(self, video_path, bound=None, fps=25):
        gif = imageio.get_reader(video_path)
        max_frame = len(gif) - 1

        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=0)
        for index, frame in enumerate(gif):
            if index in frame_indices:
                img = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                img = Image.fromarray(img)
                images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def read_frame(self, video_path, bound=None, fps=3):
        max_frame = len(os.listdir(video_path))
        images_group = list()
        frame_indices = self.get_index(bound, fps, max_frame, first_idx=1)  # frame_idx starts from 1
        for frame_index in frame_indices:
            img = Image.open(os.path.join(video_path, f'{frame_index:05d}.jpg'))
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def save_video_frames(self, imgs, video_name, frames):

        frame_paths = self.frame_paths(video_name, frames)
        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            block_size = imgs.size(0) // frames
            split_tensors = torch.split(imgs, block_size)
            to_pil = transforms.ToPILImage()
            images = [to_pil(arr) for arr in split_tensors]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths

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

    def load_into_video_and_process(self, line):
        video_path = os.path.join(line['prefix'], line['video'])

        if line['data_type'] in ['gif'] or os.path.splitext(video_path)[1] in ['.webm']:
            processed_video_path = video_path.replace(os.path.splitext(video_path)[1], '.mp4')
            if not os.path.exists(processed_video_path):
                # using MoviePy to transform GIF, webm into mp4 format
                gif_clip = VideoFileClip(video_path)
                gif_clip.write_videofile(processed_video_path, codec='libx264')
                gif_clip.close()
        elif line['data_type'] in ['frame']:
            input_images = os.path.join(video_path, '*.jpg')
            processed_video_path = f'{video_path}.mp4'
            if not os.path.exists(processed_video_path):
                # using MoviePy to transform images into mp4
                image_files = sorted(glob.glob(input_images))
                image_clip = ImageSequenceClip(image_files, fps=self.frame_fps)
                image_clip.write_videofile(processed_video_path, codec='libx264')
                image_clip.close()
        else:
            processed_video_path = video_path

        if line['bound']:
            base_name, suffix = os.path.splitext(processed_video_path)
            output_video_path = f'{base_name}_processed{suffix}'
            if not os.path.exists(output_video_path):
                video_clip = VideoFileClip(processed_video_path)
                clip = video_clip.subclip(line['start'], min(line['end'], video_clip.duration))
                clip.write_videofile(output_video_path)
                clip.close()
        else:
            output_video_path = processed_video_path

        return output_video_path

    def build_prompt(self, line, num_frames, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question, answer = self.qa_template(line)
        message = [dict(type='text', value=self.SYS)]
        message.append(dict(type='text', value=question))
        if video_llm:
            new_video_path = self.load_into_video_and_process(line)
            message.append(dict(type='video', value=new_video_path))
        else:
            bound = None
            if line['bound']:
                bound = (
                    line['start'],
                    line['end'],
                )
            video_path = os.path.join(line['prefix'], line['video'])
            decord_method = self.decord_method[line['data_type']]
            self.num_segments = num_frames if num_frames > 0 else self.nframe
            torch_imgs = decord_method(video_path, bound)
            img_frame_paths = self.save_video_frames(torch_imgs, line['video'], self.num_segments)
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value='\nOnly give the best option.'))
        message.append(dict(type='text', value='Best option:('))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):
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

                if FAIL_MSG in pred:
                    data.loc[idx, 'score'] = -1
                else:
                    data.loc[idx, 'score'] = int(check_ans(pred, ans))

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


class MVBench_MP4(VideoBaseDataset):

    MP4_MD5 = '7b4608045347904c28c153015a7a2b6b'
    SYS = """Carefully watch the video and pay attention to the cause and sequence of events, \
the detail and movement of objects, and the action and pose of persons. \
Based on your observations, select the best option that accurately addresses the question.
"""
    TYPE = 'MCQ'

    def __init__(self, dataset='MVBench_MP4', pack=False):
        super().__init__(dataset=dataset, pack=pack)

    @classmethod
    def supported_datasets(cls):
        return ['MVBench_MP4']

    def prepare_dataset(self, dataset_name='MVBench_MP4', repo_id='OpenGVLab/MVBench'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MP4_MD5:
                return False

            data = load(data_file)
            for idx, item in data.iterrows():
                if not osp.exists(osp.join(pth, item['prefix'], item['video'])):
                    return False
            return True

        cache_path = get_cache_path(repo_id, branch='video')
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:
            def generate_tsv(pth):
                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return
                json_data_path = os.path.join(dataset_path, 'test.json')
                json_data = load(json_data_path)
                root_data_dict = json_data['root']
                self.data_list = []
                for k, v in json_data['meta'].items():
                    for item in v:
                        self.data_list.append({
                            'task_type': k,
                            'prefix': root_data_dict[k],
                            'video': item['video'],
                            'question': item['question'],
                            'answer': item['answer'],
                            'candidates': item['candidates']
                        })
                data_df = pd.DataFrame(self.data_list)
                data_df = data_df.assign(index=range(len(data_df)))
                data_df.to_csv(data_file, sep='\t', index=False)

            hf_token = os.environ.get('HUGGINGFACE_TOKEN')
            huggingface_hub.login(hf_token)
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset', revision='video')
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        self.nframe = 8
        self.resolution = 224

        # transform
        crop_size = self.resolution
        scale_size = self.resolution
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        self.transform = T.Compose([
            GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
            GroupCenterCrop(crop_size),
            Stack(),
            ToTorchFormatTensor(),
            GroupNormalize(input_mean, input_std)
        ])

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

    def get_index(self, max_frame):
        seg_size = float(max_frame) / self.num_segments
        frame_indices = np.array([
            int((seg_size / 2) + np.round(seg_size * idx))
            for idx in range(self.num_segments)
        ])
        return frame_indices

    def read_video(self, video_path, bound=None):
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        max_frame = len(vr) - 1

        images_group = list()
        frame_indices = self.get_index(max_frame)
        for frame_index in frame_indices:
            img = Image.fromarray(vr[frame_index].asnumpy())
            images_group.append(img)
        torch_imgs = self.transform(images_group)
        return torch_imgs

    def save_video_frames(self, imgs, video_name, frames):

        frame_paths = self.frame_paths(video_name, frames)
        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            block_size = imgs.size(0) // frames
            split_tensors = torch.split(imgs, block_size)
            to_pil = transforms.ToPILImage()
            images = [to_pil(arr) for arr in split_tensors]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths

    def build_prompt(self, line, num_frames, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        question, answer = self.qa_template(line)
        message = [dict(type='text', value=self.SYS)]
        message.append(dict(type='text', value=question))
        video_path = os.path.join(self.data_root, line['prefix'], line['video'])
        if video_llm:
            message.append(dict(type='video', value=video_path))
        else:
            video_path = os.path.join(self.data_root, line['prefix'], line['video'])
            self.num_segments = num_frames if num_frames > 0 else self.nframe
            torch_imgs = self.read_video(video_path)
            img_frame_paths = self.save_video_frames(torch_imgs, line['video'], self.num_segments)
            for im in img_frame_paths:
                message.append(dict(type='image', value=im))
        message.append(dict(type='text', value='\nOnly give the best option.'))
        message.append(dict(type='text', value='Best option:('))
        return message

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'

        tmp_file = eval_file.replace('.xlsx', '_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', '_rating.json')
        score_file = eval_file.replace('.xlsx', '_score.xlsx')

        if not osp.exists(score_file):
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

                if FAIL_MSG in pred:
                    data.loc[idx, 'score'] = -1
                else:
                    data.loc[idx, 'score'] = int(check_ans(pred, ans))

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
