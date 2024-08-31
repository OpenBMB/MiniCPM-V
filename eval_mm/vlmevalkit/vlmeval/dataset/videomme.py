from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'


def unwrap_hf_pkl(pth, suffix='.mp4'):
    base_dir = os.path.join(pth, 'video_pkl/')
    target_dir = os.path.join(pth, 'video/')
    pickle_files = [os.path.join(base_dir, file) for file in os.listdir(base_dir)]
    pickle_files.sort()

    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
        for pickle_file in pickle_files:
            with open(pickle_file, 'rb') as file:
                video_data = pickle.load(file)
            # For each video file in the pickle file, write its contents to a new mp4 file
            for video_name, video_content in video_data.items():
                output_path = os.path.join(target_dir, f'{video_name}{suffix}')
                with open(output_path, 'wb') as output_file:
                    output_file.write(video_content)
        print('The video file has been restored and stored from the pickle file.')
    else:
        print('The video file already exists.')


class VideoMME(VideoBaseDataset):

    MD5 = '2f16cd40b1c125b67e661e59da2f6cd0'
    SYS = ''

    FRAMES_TMPL_NOSUB = """
These are the frames of a video. \
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    FRAMES_TMPL_SUB = """
These are the frames of a video. \
This video's subtitles are listed below:
{}
Select the best answer to the following multiple-choice question based on the video. \
Respond with only the letter (A, B, C, or D) of the correct option.
"""

    TYPE = 'MCQ'

    def __init__(self, dataset='Video-MME', use_subtitle=False):
        super().__init__(dataset=dataset)
        self.use_subtitle = use_subtitle

    @classmethod
    def supported_datasets(cls):
        return ['Video-MME']

    def prepare_dataset(self, dataset_name='Video-MME', repo_id='lmms-lab/Video-MME'):

        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')

            if not os.path.exists(data_file):
                return False

            if md5(data_file) != self.MD5:
                return False
            data = load(data_file)
            for video_pth in data['video_path']:
                if not osp.exists(osp.join(pth, video_pth)):
                    return False
            return True

        cache_path = get_cache_path(repo_id)
        if cache_path is not None and check_integrity(cache_path):
            dataset_path = cache_path
        else:

            def unzip_hf_zip(pth):
                import zipfile
                base_dir = pth
                target_dir = os.path.join(pth, 'video/')
                zip_files = [
                    os.path.join(base_dir, file) for file in os.listdir(base_dir)
                    if file.endswith('.zip') and file.startswith('video')
                ]
                zip_files.sort()

                if not os.path.exists(target_dir):
                    os.makedirs(target_dir, exist_ok=True)
                    for zip_file in zip_files:
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            for member in zip_ref.namelist():
                                # Check if the member is a file (not a directory)
                                if not member.endswith('/'):
                                    # Extract the file to the specified directory
                                    source = zip_ref.open(member)
                                    target = open(os.path.join(target_dir, os.path.basename(member)), 'wb')
                                    with source, target:
                                        target.write(source.read())
                    print('The video file has been restored and stored from the zip file.')
                else:
                    print('The video file already exists.')

                subtitle_zip_file = os.path.join(base_dir, 'subtitle.zip')
                subtitle_target_dir = os.path.join(base_dir, 'subtitle')

                if not os.path.exists(subtitle_target_dir):
                    os.makedirs(subtitle_target_dir, exist_ok=True)
                    with zipfile.ZipFile(subtitle_zip_file, 'r') as zip_ref:
                        for member in zip_ref.namelist():
                            # Check if the member is a file (not a directory)
                            if not member.endswith('/'):
                                # Extract the file to the specified directory
                                source = zip_ref.open(member)
                                target = open(os.path.join(subtitle_target_dir, os.path.basename(member)), 'wb')
                                with source, target:
                                    target.write(source.read())
                    print('The subtitle file has been restored and stored from the zip file.')
                else:
                    print('The subtitle file already exists.')

            def generate_tsv(pth):

                data_file = osp.join(pth, f'{dataset_name}.tsv')
                if os.path.exists(data_file) and md5(data_file) == self.MD5:
                    return

                data_file = pd.read_parquet(os.path.join(pth, 'videomme/test-00000-of-00001.parquet'))
                data_file = data_file.assign(index=range(len(data_file)))
                data_file['video'] = data_file['videoID']
                data_file['video_path'] = data_file['videoID'].apply(lambda x: f'./video/{x}.mp4')
                data_file['subtitle_path'] = data_file['videoID'].apply(lambda x: f'./subtitle/{x}.srt')
                data_file['question'] += '\n' + data_file['options'].apply(lambda x: '\n'.join(x))

                data_file = data_file[['index', 'video', 'video_path', 'duration', 'domain',
                                       'sub_category', 'task_type', 'subtitle_path', 'question', 'answer']]

                data_file.to_csv(osp.join(pth, f'{dataset_name}.tsv'), sep='\t', index=False)

            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unzip_hf_zip(dataset_path)
            generate_tsv(dataset_path)

        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=dataset_path)

    def save_video_frames(self, video, num_frames=8):

        vid_path = osp.join(self.data_root, 'video', video + '.mp4')
        vid = decord.VideoReader(vid_path)
        step_size = len(vid) / (num_frames + 1)
        indices = [int(i * step_size) for i in range(1, num_frames + 1)]

        video_info = {
            'fps': vid.get_avg_fps(),
            'n_frames': len(vid),
        }

        frame_paths = self.frame_paths(video, num_frames)
        flag = np.all([osp.exists(p) for p in frame_paths])

        if not flag:
            images = [vid[i].numpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)

        return frame_paths, indices, video_info

    def build_prompt(self, line, num_frames, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]

        frames, indices, video_info = self.save_video_frames(line['video'], num_frames)

        if self.use_subtitle and os.path.exists(osp.join(self.data_root, line['subtitle_path'])):
            import pysubs2
            subs = pysubs2.load(osp.join(self.data_root, line['subtitle_path']), encoding='utf-8')
            subtitles = []

            for seleced_frame_id in indices:
                sub_text = ''
                cur_time = pysubs2.make_time(fps=video_info['fps'], frames=seleced_frame_id)
                for sub in subs:
                    if sub.start < cur_time and sub.end > cur_time:
                        sub_text = sub.text.replace('\\N', ' ')
                        break
                if sub_text.strip():
                    subtitles.append(sub_text)
            subtitles = '\n'.join(subtitles)
        else:
            subtitles = ''

        message = [dict(type='text', value=self.SYS)]
        if video_llm:
            message.append(dict(type='video', value=osp.join(self.data_root, 'video', line['video'] + '.mp4')))
        else:
            for im in frames:
                message.append(dict(type='image', value=im))

        text_prompt = self.FRAMES_TMPL_NOSUB if not self.use_subtitle else self.FRAMES_TMPL_SUB.format(subtitles)
        message.append(dict(type='text', value=text_prompt))
        prompt = 'Question: {}\nAnswer: '.format(line['question'])
        message.append(dict(type='text', value=prompt))
        return message

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.videomme import get_dimension_rating, extract_characters_regex

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

                if extract_characters_regex(pred) == '':
                    data.loc[idx, 'score'] = -1
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
