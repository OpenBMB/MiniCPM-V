from .image_base import ImageBaseDataset
from .utils.judge_util import build_judge
from ..smp import *
from ..utils import track_progress_rich


class ImageMTDataset(ImageBaseDataset):

    TYPE = 'MT'

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        if self.meta_only:
            tgt_path = toliststr(line['image_path'])
        else:
            tgt_path = self.dump_image(line)

        questions = toliststr(line['question'])
        if 'answer' in line:
            answers = toliststr(line['answer'])
        else:
            answers = [''] * len(questions)
        assert len(questions) == len(answers)

        dlgs, pics_number = [], 0
        for i in range(len(questions)):
            q, a = questions[i], answers[i]
            if '<ImageHere>' in q:
                content = []
                tag_number = q.count('<ImageHere>')
                images = tgt_path[pics_number: pics_number + tag_number]
                pics_number += tag_number
                q_split = q.split('<ImageHere>')
                for i in range(tag_number):
                    qsp, im = q_split[i], images[i]
                    if qsp != '':
                        content.append(dict(type='text', value=qsp))
                    content.append(dict(type='image', value=im))
                if q_split[-1] != '':
                    content.append(dict(type='text', value=q_split[-1]))
            else:
                content = [dict(type='text', value=q)]
            dlgs.append(dict(role='user', content=content))
            assert '<ImageHere>' not in a, 'We currently do not support images in the answer. '
            content = [dict(type='text', value=a)]
            dlgs.append(dict(role='assistant', content=content))
        return dlgs


class MMDUDataset(ImageMTDataset):

    DATASET_URL = {'MMDU': 'https://opencompass.openxlab.space/utils/VLMEval/MMDU.tsv'}
    DATASET_MD5 = {'MMDU': '848b635a88a078f49aebcc6e39792061'}
    DIMS = [
        'Creativity', 'Richness', 'Visual Perception', 'Logical Coherence',
        'Answer Accuracy', 'Image Relationship Understanding', 'Overall Score'
    ]

    def calculat_metric(self, ans):
        all = defaultdict(lambda: 0)
        tot = defaultdict(lambda: 0)
        valid = defaultdict(lambda: 0)
        for k in ans:
            res = ans[k]['res']
            assert isinstance(res, pd.DataFrame)
            lt = len(res)
            for i in range(lt):
                line = res.iloc[i]
                for k in self.DIMS:
                    tot[k] += 1
                    if k in line and line[k] is not None:
                        try:
                            score = int(line[k])
                            score = np.clip(score, 0, 10)
                            all[k] += score
                            valid[k] += 1
                        except Exception as e:
                            print(f'Failed to parse the score: {str(e)}')
        sp1 = {'set': 'all'}
        sp1.update({k: all[k] / tot[k] * 10 for k in self.DIMS})
        sp2 = {'set': 'valid'}
        sp2.update({k: all[k] / valid[k] * 10 for k in self.DIMS})

        return pd.DataFrame([sp1, sp2])

    def evaluate(self, eval_file, **judge_kwargs):
        suffix = eval_file.split('.')[-1]
        model = judge_kwargs['model']

        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')
        score_file = eval_file.replace(f'.{suffix}', f'_{model}_score.csv')
        nproc = judge_kwargs.pop('nproc', 4)

        data = load(eval_file)
        model = judge_kwargs.pop('model', 'gpt-4o')
        judge_model = build_judge(model=model, **judge_kwargs)

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]
        tups = [(judge_model, line) for line in lines]
        indices = [line['index'] for line in lines]

        ans = {}
        if osp.exists(tmp_file):
            ans = load(tmp_file)

        tups = [x for x, i in zip(tups, indices) if i not in ans]
        indices = [i for i in indices if i not in ans]

        from .utils.mmdu import mmdu_score

        if len(indices):
            new_results = track_progress_rich(
                mmdu_score,
                tups,
                nproc=nproc,
                chunksize=nproc,
                keys=indices,
                save=tmp_file,)
            ans = load(tmp_file)
            for k, v in zip(indices, new_results):
                assert k in ans

        metric = self.calculat_metric(ans)
        dump(metric, score_file)
        return metric
