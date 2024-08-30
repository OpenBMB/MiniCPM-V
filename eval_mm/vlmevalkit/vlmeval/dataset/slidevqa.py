import re
import math
from typing import List

from vlmeval.dataset.utils.judge_util import build_judge
from vlmeval.smp import *
from .image_base import ImageBaseDataset
from .mmlongbench import concat_images, MMLongBench_auxeval, anls_compute


FAIL_MSG = 'Failed to obtain answer via API.'


def get_f1(gt, pred):
    gt_bow, pred_bow = gt.strip().split(), pred.strip().split()
    if not gt_bow or not pred_bow:
        return 0.0

    recall = len([pred_e for pred_e in pred_bow if pred_e in gt_bow]) / len(gt_bow)
    precision = len([pred_e for pred_e in pred_bow if pred_e in gt_bow]) / len(pred_bow)
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 1e-4 else 0.0
    return f1


def SlideVQA_acc(result_file):
    data = load(result_file)
    anls_list, em_list, f1_list = list(), list(), list()
    for i in range(len(data)):
        item = data.iloc[i]
        if isinstance(item['answer'], float) and math.isnan(item['answer']):
            item['answer'] = 'Not answerable'

        item['answer'] = re.sub('\n', '', item['answer']).lower()
        item['pred'] = str(item['pred']).lower()
        anls_score = anls_compute(item['answer'], item['pred'])
        em_score = (item['answer'].strip() == item['pred'].strip())
        f1_score = get_f1(item['answer'], item['pred'])
        anls_list.append(anls_score)
        em_list.append(em_score)
        f1_list.append(f1_score)
        print('---------------------')
        print(item['answer'], item['pred'], anls_score, em_score, f1_score)

    data['anls'] = anls_list
    data['em'] = em_list
    data['f1'] = f1_list
    dump(data, result_file)

    res = dict()
    res['category'], res['num'] = ['anls', 'EM', 'F1'], [len(data), len(data), len(data)]
    res['avg'] = [sum(anls_list) / len(data), sum(em_list) / len(data), sum(f1_list) / len(data)]
    res = pd.DataFrame(res)
    return res


class SlideVQA(ImageBaseDataset):

    TYPE = 'VQA'

    DATASET_URL = {
        'SLIDEVQA_MINI': 'https://opencompass.openxlab.space/utils/VLMEval/SLIDEVQA_MINI.tsv',
        'SLIDEVQA': 'https://opencompass.openxlab.space/utils/VLMEval/SLIDEVQA.tsv',
    }
    DATASET_MD5 = {
        'SLIDEVQA_MINI': '6d9a8d8814fa5b7669deb2af3a3208eb',
        'SLIDEVQA': '5e822c2f800e94c1e23badfd478326b6',
    }

    SUPPORTED_MODELS = {
        'GPT4': (1, 1),
        'GPT4V': (1, 1),
        'GPT4V_HIGH': (1, 1),
        'GPT4o': (1, 1),
        'GPT4o_HIGH': (1, 1),
        'GPT4o_MINI': (1, 1),
        'XComposer2d5': (1, -1),
        'XComposer2_4KHD': (1, -1),
        'MiniCPM-Llama3-V-2_5': (1, 5),
        'InternVL-Chat-V1-5': (5, 2),
    }

    def __init__(self, dataset, **kwargs):
        self.model_list = list(self.SUPPORTED_MODELS.keys())
        model_name = kwargs['model']
        if not listinstr(self.model_list, model_name):
            raise AssertionError("{} doesn't support the evaluation on SlideVQA.".format(model_name))
        super(SlideVQA, self).__init__(dataset)

        self.is_api = True if listinstr(['GPT4'], model_name) else False
        self.max_pages = 120
        concat_num, column_num = self.SUPPORTED_MODELS.get(model_name)
        self.concat_num = concat_num
        self.column_num = column_num

    def dump_image(self, origin_line):
        os.makedirs(self.img_root, exist_ok=True)

        line = origin_line.copy()
        if not isinstance(line['image_path'], List):
            line['image_path'] = [line['image_path']]
        line['image_path'] = line['image_path'][:self.max_pages]

        if 'image' in line:
            if isinstance(line['image'], list):
                tgt_path = []
                assert 'image_path' in line
                for img, im_name in zip(line['image'], line['image_path']):
                    path = osp.join(self.img_root, im_name)
                    if not read_ok(path):
                        decode_base64_to_image_file(img, path)
                    tgt_path.append(path)
            else:
                tgt_path = osp.join(self.img_root, f"{line['index']}.jpg")
                if not read_ok(tgt_path):
                    decode_base64_to_image_file(line['image'], tgt_path)
                tgt_path = [tgt_path]
        else:
            assert 'image_path' in line
            tgt_path = toliststr(line['image_path'])

        if self.concat_num > 0 and not self.is_api:
            concatenated_images = concat_images(tgt_path, max_concat=self.concat_num, column_num=self.column_num)

            old_tgt_path = tgt_path
            assert isinstance(old_tgt_path, list)
            if self.column_num != -1:
                tgt_path = [
                    '_'.join(old_tgt_path[0].split('_')[:-1]) + '_concat{}_{}.jpg'.format(self.concat_num, i)
                    for i in range(len(concatenated_images))
                ]
            else:
                tgt_path = ['_'.join(old_tgt_path[0].split('_')[:-1]) + '_concat_all.jpg']

            for path, concatenated_image in zip(tgt_path, concatenated_images):
                if not read_ok(path):
                    decode_base64_to_image_file(encode_image_to_base64(concatenated_image), path)
                    num_images, image_size = len(old_tgt_path), concatenated_image.size
                    print('concat {} images to a new one with size {}. save at {}'.format(num_images, image_size, path))
        return tgt_path

    @classmethod
    def evaluate(self, eval_file, **judge_kwargs):
        logger = get_logger('Evaluation')
        model = judge_kwargs['model']

        suffix = eval_file.split('.')[-1]
        storage = eval_file.replace(f'.{suffix}', f'_{model}.xlsx')
        tmp_file = eval_file.replace(f'.{suffix}', f'_{model}.pkl')

        if osp.exists(storage):
            logger.warning(f'GPT scoring file {storage} already exists, will reuse it in SlideVQA_eval. ')
        else:
            data = load(eval_file)
            model = build_judge(max_tokens=128, **judge_kwargs)
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
                new_results = list()
                for model, line in tqdm(tups):
                    res = MMLongBench_auxeval(model, line)
                    new_results.append(res)

            log_map, res_map, pred_map = {}, {}, {}
            all_inds = [line['index'] for line in lines]
            for k, v in zip(all_inds, new_results):
                log_map[k] = v['log']
                res_map[k] = v['res']
                pred_map[k] = v['pred']
            data['res'] = [res_map[idx] for idx in data['index']]
            data['log'] = [log_map[idx] for idx in data['index']]
            data['pred'] = [pred_map[idx] for idx in data['index']]
            dump(data, storage)

        score = SlideVQA_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')

        dump(score, score_pth)
        logger.info(f'SlideVQA successfully finished evaluating {eval_file}, results saved in {score_pth}')
        logger.info('Score: ')
        logger.info(score)
