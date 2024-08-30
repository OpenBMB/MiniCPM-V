from huggingface_hub import snapshot_download
from ..smp import *
from .video_base import VideoBaseDataset
from .utils import build_judge, DEBUG_MESSAGE
from ..utils import track_progress_rich


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


class MMBenchVideo(VideoBaseDataset):

    MD5 = '98f7df3eb1007fc375ea6fe88a98e2ff'
    SYS = 'You are an AI assistant responsible for answering questions about videos.'
    FRAMES_TMPL_PACK = """
You will be provided with {} separate frames uniformly sampled from a video, \
the frames are provided in chronological order of the video.
Please analyze these images and provide the answer / answers to the \
following question / questions about the video content.
If multiple questions are provided (with indices I1, I2, I3, ...), \
you should organize your answers in the following json format:
{{
    'I1': 'Answer to Question I1',
    'I2': 'Answer to Question I2',
    ...
}}
Otherwise, please directly reply with your response to the only question.
Even if the information in these separate frames is not enough to give an answer,
PLEASE GIVE A RESPONSE TO EACH OF THE QUESTIONS IN THE FORMAT DESCRIBED ABOVE.
"""

    FRAMES_TMPL_NOPACK = """
You will be provided with {} separate frames uniformly sampled from a video, \
the frames are provided in chronological order of the video.
Please analyze these images and provide the answer to the question about the video content.
Please directly reply with your response to the only question.
"""

    TYPE = 'VQA'

    def __init__(self, dataset='MMBench-Video', pack=False):
        super().__init__(dataset=dataset, pack=pack)

    @classmethod
    def supported_datasets(cls):
        return ['MMBench-Video']

    def prepare_dataset(self, dataset_name='MMBench-Video', repo_id='nebulae09/MMBench-Video'):
        def check_integrity(pth):
            data_file = osp.join(pth, f'{dataset_name}.tsv')
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
            dataset_path = snapshot_download(repo_id=repo_id, repo_type='dataset')
            unwrap_hf_pkl(dataset_path)
        self.video_path = osp.join(dataset_path, 'video/')
        data_file = osp.join(dataset_path, f'{dataset_name}.tsv')

        return dict(data_file=data_file, root=osp.join(dataset_path, 'video'))

    def build_prompt_pack(self, line, num_frames):
        if isinstance(line, int):
            assert line < len(self)
            video = self.videos[line]
        elif isinstance(line, pd.Series):
            video = line['video']
        elif isinstance(line, str):
            video = line

        frames = self.save_video_frames(video, num_frames)
        sub = self.data[self.data['video'] == video]
        sys_prompt = self.SYS + self.FRAMES_TMPL_PACK.format(num_frames)
        message = [dict(type='text', value=sys_prompt)]
        for im in frames:
            message.append(dict(type='image', value=im))
        nq = len(sub)
        prompt = 'Questions: \n{}\nAnswers: \n'
        qs = {int(sub.iloc[i]['index']): sub.iloc[i]['question'] for i in range(nq)}
        prompt = prompt.format(json.dumps(qs))
        message.append(dict(type='text', value=prompt))
        return message

    def build_prompt_nopack(self, line, num_frames, video_llm):
        if isinstance(line, int):
            assert line < len(self)
            line = self.data.iloc[line]
        if video_llm:
            question = line['question']
            prefix, video_idx_path = os.path.split(line['video_path'])
            message = [dict(type='text', value=question)]
            message.append(dict(type='video', value=os.path.join(self.video_path, video_idx_path)))
            return message
        else:
            frames = self.save_video_frames(line['video'], num_frames)
            sys_prompt = self.FRAMES_TMPL_NOPACK.format(num_frames)
            message = [dict(type='text', value=sys_prompt)]
            for im in frames:
                message.append(dict(type='image', value=im))
            prompt = 'Question: {}\nAnswer: '.format(line['question'])
            message.append(dict(type='text', value=prompt))
        return message

    def build_prompt(self, line, num_frames, video_llm):
        if self.pack and not video_llm:
            return self.build_prompt_pack(line, num_frames)
        else:
            return self.build_prompt_nopack(line, num_frames, video_llm)

    @staticmethod
    def remove_side_quote(s, syms=[',', '"', "'"]):
        if np.all([x in syms for x in s]):
            return ''
        while s[0] in syms:
            s = s[1:]
        while s[-1] in syms:
            s = s[:-1]
        return s

    @staticmethod
    def robust_json_load(s):
        try:
            jsons = list(extract_json_objects(s))
            assert len(jsons) == 1
            return jsons[0]
        except:
            if '{' in s and s.find('{') == s.rfind('{'):
                sub_str = s[s.find('{') + 1:].strip()
                lines = sub_str.split('\n')
                res = {}
                for l in lines:
                    l = l.strip()
                    if ': ' in l:
                        key = l.split(': ')[0].strip()
                        val = l.split(': ')[1].strip()
                        key = MMBenchVideo.remove_side_quote(key)
                        val = MMBenchVideo.remove_side_quote(val)
                        if len(key) and len(val):
                            res[key] = val
                return res
            return None

    def load_pack_answers(self, data_raw):
        vstats = defaultdict(lambda: 0)
        data = defaultdict(lambda: {})

        for k in data_raw:
            ans = data_raw[k].strip()
            if FAIL_MSG in ans:
                vstats['GEN_FAIL'] += 1
                continue
            res = self.robust_json_load(ans)
            if res is not None:
                data[k] = res
                vstats['PARSE_OK'] += 1
            else:
                vstats['PARSE_FAIL'] += 1

        # return data
        meta = cp.deepcopy(self.data)
        lt = len(meta)
        prediction = []
        for i in range(lt):
            line = meta.iloc[i]
            vid = line['video']
            idx = str(line['index'])
            prediction.append(data[vid][idx] if idx in data[vid] else None)
        meta['prediction'] = prediction
        vstats['VALIDQ'] = len([x for x in prediction if x is not None])
        vstats['INVALIDQ'] = len([x for x in prediction if x is None])
        return meta, vstats

    # It returns a dictionary
    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        from .utils.mmbench_video import get_dimension_rating, system_prompt, build_prompt

        assert eval_file.endswith('.xlsx'), 'data file should be an xlsx file'
        judge = judge_kwargs['model']
        nproc = judge_kwargs.pop('nproc', 4)

        tmp_file = eval_file.replace('.xlsx', f'_{judge}_tmp.pkl')
        tgt_file = eval_file.replace('.xlsx', f'_{judge}_rating.json')
        score_file = eval_file.replace('.xlsx', f'_{judge}_score.xlsx')

        model = build_judge(system_prompt=system_prompt, **judge_kwargs)
        assert model.working(), 'MMBench-Video evaluation requires a working OPENAI API\n' + DEBUG_MESSAGE

        if not osp.exists(score_file):
            res = {} if not osp.exists(tmp_file) else load(tmp_file)
            res = {k: v for k, v in res.items() if model.fail_msg not in v}

            data = load(eval_file)
            data_un = data[~data['index'].isin(res)]
            data_un = data_un[~pd.isna(data_un['prediction'])]
            lt = len(data_un)
            prompts = [build_prompt(data_un.iloc[i]) for i in range(lt)]
            indices = [data_un.iloc[i]['index'] for i in range(lt)]

            if len(prompts):
                _ = track_progress_rich(
                    model.generate,
                    prompts,
                    keys=indices,
                    save=tmp_file,
                    nproc=nproc,
                    chunksize=nproc
                )
            score_map = load(tmp_file)
            data['score'] = [score_map[idx] if idx in score_map else -1 for idx in data['index']]
            rejected = [x for x in score_map.values() if FAIL_MSG in x]
            data['score'] = [int(x) if istype(x, int) else -1 for x in data['score']]
            print(
                f'Among {len(data)} questions, failed to obtain prediction for {len(data) - len(score_map)} questions, '
                f'failed to obtain the score for another {len(rejected)} questions. '
                f'Those questions will be counted as 0 score in ALL rating, and will not be counted in VALID rating.'
            )

            dump(data, score_file)

        rating = get_dimension_rating(score_file)
        dump(rating, tgt_file)
        return rating
