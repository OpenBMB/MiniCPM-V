import re
import math
from urllib.request import urlopen
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms

from vlmeval.dataset.utils import build_judge, levenshtein_distance
from vlmeval.smp import *
from .image_base import ImageBaseDataset

FAIL_MSG = 'Failed to obtain answer via API.'


def get_gpt4_ICE():
    example_1 = """
---
Question: List the primary questions asked about the services in this report.
Analysis:  The primary questions asked about the services in the report for The Limes Residential Home are:\n\n
1. Is the service safe?\n
2. Is the service effective?\n
3. Is the service caring?\n
4. Is the service responsive?\n
5. Is the service well-led?
Extracted answer: [
    'Is the servife safe?',
    'Is the service effective',
    'Is the serve caring?',
    'Is the service responsive?',
    'Is the service well-led?'
]
Answer format: List\n
"""

    example_2 = """
---
Question: How many regulations of the HSCA 2008 are breached in all according to this report?
Analysis: According to the report, the provider breached 10 Health and Social Care Act 2008 (Regulated Activities)
Regulations in total. Here are the specifics:\n\n1. Regulation 13: Safeguarding service users from abuse and
improper treatment\n2. Regulation 12: Safe care and treatment\n3. Regulation 18: Staffing\n4. Regulation 11:
Need for consent\n5. Regulation 10: Dignity and respect\n6. Regulation 9: Person-centred care\n7. Regulation 17:
Good governance\n8. Regulation 18 (CQC Registration Regulations 2009): Notification of other incidents\n9.
Regulation 18: Failure to maintain an accurate and up-to-date care plan\n10. Regulation 11: Failure to implement
the Mental Capacity Act 2005 code of practice effectively\n\nThese breaches involve issues concerning staffing,
safeguarding, medicines management, dignity and respect, consent, care planning, governance, and failure to
notify the CQC of incidents.
Extracted answer: 10
Answer format: Integer\n
"""

    example_3 = """
---
Question: According to the survey that is the percentage of Chinese who are paying more or
about the same attention to politics after Trump's election?
Analysis: The survey provided does not specify the percentage of Chinese individuals specifically who are paying
more or about the same attention to politics after Trump's election. The report focuses primarily on American
demographics and does not include specific details about the Chinese population in relation to this question. If
you need information about a different demographic or a summary of the findings from the American demographic,
I can certainly help with that!
Extracted answer: Not answerable
Answer format: String\n
"""

    example_4 = """
---
Question: How many quotations from male respondent over 50 years old are included in this report?
Analysis: The image you've provided appears to be a screenshot of a document with multiple charts. However, the
text is too small and blurry to read accurately. If you can provide a clearer image or more context, I might be
able to help you with your question.
Extracted answer: Fail to answer
Answer format: String\n
"""

    return [example_1, example_2, example_3, example_4]


def build_mmlongbench_gpt4_prompt(line):
    task_description = """
Given the question and analysis, you are tasked to extract answers with required formats from the free-form analysis.
- Your extracted answers should be one of the following formats: (1) Integer, (2) Float, (3) String and (4) List.
If you find the analysis the question can not be answered from the given documents, type "Not answerable".
Exception: If the analysis only tells you that it can not read/understand the images or documents,
type "Fail to answer".
- Please make your response as concise as possible. Also note that your response should be formatted as below:
```
Extracted answer: [answer]
Answer format: [answer format]
```
Please read the following example, then extract the answer from the model response
and type it at the end of the prompt.\n
"""
    question = line['question']
    prediction = str(line['prediction'])
    prompt = task_description
    examples = get_gpt4_ICE()
    for example in examples:
        prompt += example
    prompt += '---\nQuestion:' + question + '\n'
    prompt += 'Analysis: ' + prediction
    return prompt


def anls_compute(groundtruth, prediction, threshold=0.5):
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls <= threshold:
        anls = 0.0
    return anls


def is_float_equal(reference, prediction, include_percentage: bool = False, is_close: float = False) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    reference = float(str(reference).strip().rstrip('%').strip())
    try:
        prediction = float(str(prediction).strip().rstrip('%').strip())
    except:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if math.isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(s):
    s = str(s).lower().strip()
    if s.endswith('mile'):
        s.rstrip('mile').strip()
    if s.endswith('miles'):
        s.rstrip('miles').strip()
    if s.endswith('million'):
        s.rstrip('million').strip()
    # remove parenthesis
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", '', s).strip()
    s = s.strip().lstrip('$').strip()
    s = s.strip().rstrip('%').strip()
    return s


def is_exact_match(s):
    flag = False
    # Website
    if 'https://' in s:
        flag = True
    # code file
    if s.endswith('.py') or s.endswith('ipynb'):
        flag = True
    if s.startswith('page'):
        flag = True
    # telephone number
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        flag = True
    # time
    if 'a.m.' in s or 'p.m.' in s:
        flag = True
    # YYYY-MM-DD
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        flag = True
    # YYYY-MM
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        flag = True
    # Email address
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        flag = True
    return flag


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def get_font():
    try:
        truetype_url = 'http://opencompass.openxlab.space/utils/Fonts/SimHei.ttf'
        ff = urlopen(truetype_url)
        font = ImageFont.truetype(ff, size=40)
    except:
        print('Fail to download the font. Use the default one.')
        font = ImageFont.load_default(size=40)
    return font


def frame2img(img_path_list, font, save_path=None, idx_start=0):
    imgs = [Image.open(img_path) for img_path in img_path_list]

    new_imgs = []
    for img in imgs:
        w, h = img.size
        scale = w / h
        if w > h:
            new_w = 560 * 2
            new_h = int(560 * 2 / scale)
        else:
            new_w = int(560 * 2 * scale)
            new_h = 560 * 2
        img = transforms.functional.resize(img, [new_h, new_w],)
        new_imgs.append(img)
    imgs = new_imgs
    new_w = 0
    new_h = 0
    pad = 40
    if w > h:
        for im in imgs:
            w, h = im.size
            new_w = max(new_w, w)
            new_h += h + 10 + pad
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_h = 0
        for idx, im in enumerate(imgs):
            w, h = im.size
            new_img.paste(im, (0, pad + curr_h))
            draw.text((0, curr_h), f'<IMAGE {idx+idx_start}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(0, pad + curr_h + h + 5), (new_w, pad + curr_h + h + 5)], fill='black', width=2)
            curr_h += h + 10 + pad
    else:
        for im in imgs:
            w, h = im.size
            new_w += w + 10
            new_h = max(new_h, h)
        new_h += pad
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_w = 0
        for idx, im in enumerate(imgs):
            w, h = im.size
            new_img.paste(im, (curr_w, pad))
            draw.text((curr_w, 0), f'<IMAGE {idx+idx_start}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(curr_w + w + 5, 0), (curr_w + w + 5, new_h)], fill='black', width=2)
            curr_w += w + 10

    if save_path is not None:
        new_img.save(save_path)

    return new_img


def concat_images(image_list, max_concat=1, column_num=1):
    concatenated_images = []
    if column_num == -1:
        MAX_COLUMN_NUM = 20
        max_concat = 1
        while len(image_list) / max_concat > MAX_COLUMN_NUM:
            max_concat += 1
        interval = max(math.ceil(len(image_list) / max_concat), 1)
        for i in range(0, len(image_list), interval):
            batch_images = image_list[i:i + interval]
            concatenated_image = frame2img(batch_images, font=get_font(), idx_start=i)
            concatenated_images.append(concatenated_image)
    else:
        interval = max(math.ceil(len(image_list) / max_concat), 1)
        for i in range(0, len(image_list), interval):
            batch_images = [Image.open(filename) for filename in image_list[i:i + interval]]
            if column_num == 1:
                total_height = batch_images[0].height * len(batch_images)
            else:
                total_height = batch_images[0].height * ((len(batch_images) - 1) // column_num + 1)
            concatenated_image = Image.new('RGB', (batch_images[0].width * column_num, total_height), 'white')

            x_offset, y_offset = 0, 0
            for count, image in enumerate(batch_images):
                concatenated_image.paste(image, (x_offset, y_offset))
                x_offset += image.width
                if (count + 1) % column_num == 0:
                    y_offset += image.height
                    x_offset = 0
            concatenated_images.append(concatenated_image)
    return concatenated_images


def eval_score(gt, pred, answer_type):
    if answer_type == 'Int':
        try:
            gt, pred = int(gt), int(float(pred))
        except:
            pred = ''
        score = (gt == pred)
    elif answer_type == 'Float':
        try:
            gt = float(get_clean_string(str(gt)))
            pred = float(get_clean_string(str(pred)))
        except:
            pred = ''
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
    elif answer_type == 'Str':
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = (gt == pred)
        else:
            score = anls_compute(gt, pred)
    else:
        if isinstance(gt, str) and gt.startswith('['):
            gt = eval(gt)
        if not isinstance(gt, list):
            gt = [gt]
        if isinstance(pred, str) and pred.startswith('['):
            pred = eval(pred)
        if not isinstance(pred, list):
            pred = [pred]
        print(len(gt), len(pred))
        if len(gt) != len(pred):
            score = 0.0
        else:
            gt = sorted([get_clean_string(a) for a in gt])
            pred = sorted([get_clean_string(a) for a in pred])
            print(gt, pred)
            if isfloat(gt[0]) or is_exact_match(gt[0]):
                score = ('-'.join(gt) == '-'.join(pred))
            else:
                score = min([anls_compute(gt_v, pred_v) for gt_v, pred_v in zip(gt, pred)])

    return float(score)


def MMLongBench_auxeval(model, line):
    prompt = build_mmlongbench_gpt4_prompt(line)
    log = ''
    retry = 5

    for i in range(retry):
        prediction = line['prediction']
        res = model.generate(prompt, temperature=i * 0.5)

        if FAIL_MSG in res:
            log += f'Try {i}: output is {prediction}, failed to parse.\n'
        else:
            log += 'Succeed'
            try:
                pred = res.split('Answer format:')[0].split('Extracted answer:')[1].strip()
            except:
                pred = ''
            return dict(log=log, res=res, pred=pred)
    log += 'All 5 retries failed.\n'
    return dict(log=log, res='', pred='')


def get_f1(data):
    gt_pos_data = data[data.apply(lambda k: k['answer'] != 'Not answerable', axis=1)]
    pred_pos_data = data[data.apply(lambda k: k['pred'] != 'Not answerable', axis=1)]
    recall = sum(gt_pos_data['score'].tolist()) / len(gt_pos_data)
    precision = sum(pred_pos_data['score'].tolist()) / len(pred_pos_data)
    return 2 * recall * precision / (recall + precision)


def MMLongBench_acc(result_file):
    data = load(result_file)
    overall_score = 0.0
    score_list = list()
    for i in range(len(data)):
        item = data.iloc[i]
        try:
            score = eval_score(item['answer'], item['pred'], item['answer_format'])
        except:
            score = 0.0
        score_list.append(score)
        overall_score += score

    data['score'] = score_list
    dump(data, result_file)

    data_chart = data[data.apply(lambda k: 'Chart' in eval(k['evidence_sources']), axis=1)]
    data_table = data[data.apply(lambda k: 'Table' in eval(k['evidence_sources']), axis=1)]
    data_image = data[data.apply(lambda k: 'Figure' in eval(k['evidence_sources']), axis=1)]
    data_text = data[data.apply(lambda k: 'Pure-text (Plain-text)' in eval(k['evidence_sources']), axis=1)]
    data_layout = data[data.apply(lambda k: 'Generalized-text (Layout)' in eval(k['evidence_sources']), axis=1)]

    data_single = data[data.apply(lambda k: len(eval(k['evidence_pages'])) == 1, axis=1)]
    data_multi = data[data.apply(lambda k: len(eval(k['evidence_pages'])) > 1, axis=1)]
    data_unans = data[data.apply(lambda k: len(eval(k['evidence_pages'])) == 0, axis=1)]

    res = dict()
    res['category'] = [
        'overall_f1', 'overall_acc', 'text', 'layout', 'table', 'chart',
        'image', 'single-page', 'multi-page', 'unanswerable'
    ]
    res['num'] = [
        len(data), len(data), len(data_text), len(data_layout), len(data_table),
        len(data_chart), len(data_image), len(data_single), len(data_multi), len(data_unans)
    ]
    res['avg_score'] = [
        get_f1(data),
        overall_score / len(data),
        sum(data_text['score'].tolist()) / len(data_text) if len(data_text) > 0 else 0.0,
        sum(data_layout['score'].tolist()) / len(data_layout) if len(data_layout) > 0 else 0.0,
        sum(data_table['score'].tolist()) / len(data_table) if len(data_table) > 0 else 0.0,
        sum(data_chart['score'].tolist()) / len(data_chart) if len(data_chart) > 0 else 0.0,
        sum(data_image['score'].tolist()) / len(data_image) if len(data_image) > 0 else 0.0,
        sum(data_single['score'].tolist()) / len(data_single) if len(data_single) > 0 else 0.0,
        sum(data_multi['score'].tolist()) / len(data_multi) if len(data_multi) > 0 else 0.0,
        sum(data_unans['score'].tolist()) / len(data_unans) if len(data_unans) > 0 else 0.0,
    ]
    res = pd.DataFrame(res)
    return res


class MMLongBench(ImageBaseDataset):

    TYPE = 'VQA'

    DATASET_URL = {
        'MMLongBench_DOC': 'https://opencompass.openxlab.space/utils/VLMEval/MMLongBench_DOC.tsv',
    }
    DATASET_MD5 = {
        'MMLongBench_DOC': '9b393e1f4c52718380d50586197eac9b',
    }

    SUPPORTED_MODELS = {
        'GPT4': (1, 1),
        'GPT4V': (1, 1),
        'GPT4V_HIGH': (1, 1),
        'GPT4o': (1, 1),
        'GPT4o_HIGH': (1, 1),
        'GPT4o_MINI': (1, 1),
        'MiniCPM-Llama3-V-2_5': (1, 5),
        'InternVL-Chat-V1-5': (5, 2),
        'XComposer2_4KHD': (1, 5),
        'XComposer2d5': (1, -1),
    }

    def __init__(self, dataset, **kwargs):
        self.model_list = list(self.SUPPORTED_MODELS.keys())
        model_name = kwargs['model']
        if not listinstr(self.model_list, model_name):
            raise AssertionError("{} doesn't support the evaluation on MMLongBench_DOC.".format(model_name))
        super(MMLongBench, self).__init__(dataset)

        self.is_api = True if listinstr(['GPT4'], model_name) else False
        self.max_pages = 120
        concat_num, column_num = self.SUPPORTED_MODELS.get(model_name)
        self.concat_num = concat_num
        self.column_num = column_num

    def dump_image(self, origin_line):
        os.makedirs(self.img_root, exist_ok=True)
        try:
            import fitz
        except:
            warnings.warn('Please use `pip install pymupdf` to parse PDF files.')

        line = origin_line.copy()
        line['image_path'] = line['image_path'][:self.max_pages]
        skip_pdf_parse = True
        for im_name in line['image_path']:
            path = osp.join(self.img_root, im_name)
            if not read_ok(path):
                skip_pdf_parse = False
                break

        # Just for being compatible with the zooped loop: zip(line['image'], line['image_path'])
        if skip_pdf_parse:
            line['image'] = line['image_path']
        else:
            pdf_data = base64.b64decode(line['image'])
            pdf_file = io.BytesIO(pdf_data)
            encoded_images = []
            with fitz.open(stream=pdf_file, filetype='pdf') as doc:
                doc = doc[:self.max_pages]
                for page in doc:
                    image = page.get_pixmap(dpi=144)
                    image_file = io.BytesIO(image.tobytes(output='png'))
                    image = Image.open(image_file)
                    encoded_image = encode_image_to_base64(image)
                    encoded_images.append(encoded_image)
            line['image'] = encoded_images
            print('process {}'.format(line['doc_id']))

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
                tgt_path = [
                    '_'.join(old_tgt_path[0].split('_')[:-1]) + '_concat_all_{}.jpg'.format(i)
                    for i in range(len(concatenated_images))
                ]

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
            logger.warning(f'GPT scoring file {storage} already exists, will reuse it in MMLongBench_eval. ')
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

        score = MMLongBench_acc(storage)
        score_pth = storage.replace('.xlsx', '_score.csv')

        dump(score, score_pth)
        logger.info(f'MMLongBench_eval successfully finished evaluating {eval_file}, results saved in {score_pth}')
        logger.info('Score: ')
        logger.info(score)
