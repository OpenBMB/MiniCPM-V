import uuid
from functools import partial
from .image_base import ImageBaseDataset
from ..smp import *

rouge = None
nlp_en = None
nlp_zh = None
nlp = None


def initialize():
    import evaluate
    import spacy

    global rouge, nlp_en, nlp_zh, nlp

    try:
        rouge = evaluate.load('rouge', experiment_id=str(uuid.uuid4()))
    except:
        warnings.warn('Please first `pip install rouge_score`.')

    try:
        nlp_en = spacy.load('en_core_web_sm')
    except:
        warnings.warn('Will automatically download en_core_web_sm via spacy.')
        spacy.cli.download('en_core_web_sm')
        nlp_en = spacy.load('en_core_web_sm')

    try:
        nlp_zh = spacy.load('zh_core_web_sm')
    except:
        warnings.warn('Will automatically download zh_core_web_sm via spacy.')
        spacy.cli.download('zh_core_web_sm')
        nlp_zh = spacy.load('zh_core_web_sm')

    nlp = {'en': nlp_en, 'zh': nlp_zh}


def rough_filter(answer_text):
    if "I can't" in answer_text:
        return False
    elif 'I cannot' in answer_text:
        return False
    elif 'sorry' in answer_text.lower():
        return False
    if '无法' in answer_text:
        return False
    elif '抱歉' in answer_text:
        return False
    else:
        return True


def zero_template(crossed_text):
    return {
        'crossed_text': crossed_text,
        'max_sim_val': 0,
        'max_sim_string': '',
        'precision': 0,
        'recall': 0,
        'f1': 0,
        'jaccard': 0,
        'rouge1': 0,
        'exact_match': 0,
    }


def tokenize(text, language):
    """
    Tokenize the text and return the tokens.

    Parameters:
    text (str): The text to tokenize.
    language (str): The language of the text.

    Returns:
    list: The list of tokens.
    """
    assert language in ['en', 'zh']
    nlp_language = nlp[language]
    processed_text = nlp_language(text)
    return [token.text for token in processed_text]


def find_best_match(needle, hay, language, rouge):
    """
    Finds the best matching n-gram in the haystack for the given needle.

    Parameters:
    needle (str): The string to find.
    hay (str): The text to search within.

    Returns:
    tuple: The highest similarity value and the best matching string.
    """
    assert language in ['en', 'zh']
    from nltk.util import ngrams
    from difflib import SequenceMatcher as SM

    tokens_hay = tokenize(hay, language)
    tokens_needle = tokenize(needle, language)

    splitter = '' if language == 'zh' else ' '
    ngrams_ = ngrams(tokens_hay, len(tokens_needle))
    max_sim_val = 0
    max_sim_string = ''
    max_sim_ngram = []
    tokens_needle_set = set(tokens_needle)
    ngrams_hasjoint = [
        ngram
        for ngram in ngrams_
        if not set(ngram).isdisjoint(tokens_needle_set)
    ]

    for ngram in ngrams_hasjoint:
        hay_ngram = splitter.join(ngram)
        similarity = SM(None, hay_ngram, needle).ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = hay_ngram
            max_sim_ngram = ngram

    # Evaluate
    if len(max_sim_ngram) == 0:
        return {
            'crossed_text': needle,
            'max_sim_val': 0,
            'max_sim_string': '',
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'jaccard': 0,
            'rouge1': 0,
            'exact_match': 0,
        }
    pred_set = set(max_sim_ngram)
    ref_set = set(tokens_needle)
    correct_tokens = pred_set.intersection(ref_set)
    len_correct_tokens = len(correct_tokens)

    precision = len_correct_tokens / len(pred_set)
    recall = len_correct_tokens / len(ref_set)
    if (precision + recall) == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    union = pred_set.union(ref_set)
    jaccard = len_correct_tokens / len(union) if len(union) > 0 else 0
    rouge_1 = rouge.compute(
        predictions=[max_sim_string],
        references=[needle],
        tokenizer=partial(tokenize, language=language),
        rouge_types=['rouge1'],
    )['rouge1']
    exact_match = float(list(max_sim_ngram) == list(tokens_needle))
    out = {
        'crossed_text': needle,
        'max_sim_string': max_sim_string,
        'max_sim_val': max_sim_val,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'jaccard': jaccard,
        'rouge1': rouge_1,
        'exact_match': exact_match,
    }
    return out


def process_match_single_new(
        image_id, prediction, answer, language, progress
):
    """
    process the inference results for a single image and calculate the metrics

    Parameters:
    image_id (int): The image id (question id).
    prediction (str): The prediction text.
    answer (Union[str, List[str]]): The answer text, or a list of answer texts. The masked n-grams in the image.
    language (str): The language of the text. Can be "en" or "zh".
    rouge (rouge): The rouge metric object.
    progress (multiprocessing.Queue): The progress queue.

    Returns:
    tuple: The image id (question_id, int) and the result per id (dict of dict of dict).
    """
    result_per_id = {image_id: {}}
    if isinstance(answer, str):
        answer = eval(answer)
    assert isinstance(answer, list)
    result = prediction.split('Assistant: ')[-1]
    for i, crossed_text in enumerate(answer):
        if rough_filter(result):
            find_best_match_result = find_best_match(
                crossed_text, result, language, rouge
            )
            if i == 0:
                result_per_id[image_id] = {str(i): find_best_match_result}
            else:
                result_per_id[image_id][str(i)] = find_best_match_result
        else:
            if i == 0:
                result_per_id[image_id] = {str(i): zero_template(crossed_text)}
            else:
                result_per_id[image_id][str(i)] = zero_template(crossed_text)
    progress.put(1)
    return image_id, result_per_id


class VCRDataset(ImageBaseDataset):
    TYPE = 'VQA'

    URL_PREFIX = 'https://huggingface.co/datasets/vcr-org'

    DATASET_URL = {
        'VCR_EN_EASY_500': f'{URL_PREFIX}/VCR-wiki-en-easy-test-500/resolve/main/VCR-wiki-en-easy-test-500.tsv',
        'VCR_EN_EASY_100': f'{URL_PREFIX}/VCR-wiki-en-easy-test-100/resolve/main/VCR-wiki-en-easy-test-100.tsv',
        'VCR_EN_EASY_ALL': f'{URL_PREFIX}/VCR-wiki-en-easy-test/resolve/main/VCR-wiki-en-easy-test.tsv',
        'VCR_EN_HARD_500': f'{URL_PREFIX}/VCR-wiki-en-hard-test-500/resolve/main/VCR-wiki-en-hard-test-500.tsv',
        'VCR_EN_HARD_100': f'{URL_PREFIX}/VCR-wiki-en-hard-test-100/resolve/main/VCR-wiki-en-hard-test-100.tsv',
        'VCR_EN_HARD_ALL': f'{URL_PREFIX}/VCR-wiki-en-hard-test/resolve/main/VCR-wiki-en-hard-test.tsv',
        'VCR_ZH_EASY_500': f'{URL_PREFIX}/VCR-wiki-zh-easy-test-500/resolve/main/VCR-wiki-zh-easy-test-500.tsv',
        'VCR_ZH_EASY_100': f'{URL_PREFIX}/VCR-wiki-zh-easy-test-100/resolve/main/VCR-wiki-zh-easy-test-100.tsv',
        'VCR_ZH_EASY_ALL': f'{URL_PREFIX}/VCR-wiki-zh-easy-test/resolve/main/VCR-wiki-zh-easy-test.tsv',
        'VCR_ZH_HARD_500': f'{URL_PREFIX}/VCR-wiki-zh-hard-test-500/resolve/main/VCR-wiki-zh-hard-test-500.tsv',
        'VCR_ZH_HARD_100': f'{URL_PREFIX}/VCR-wiki-zh-hard-test-100/resolve/main/VCR-wiki-zh-hard-test-100.tsv',
        'VCR_ZH_HARD_ALL': f'{URL_PREFIX}/VCR-wiki-zh-hard-test/resolve/main/VCR-wiki-zh-hard-test.tsv',
    }

    DATASET_MD5 = {
        'VCR_EN_EASY_500': 'fd9258db52f8685dc710619a0ea0a261',
        'VCR_EN_EASY_100': '9df5d7266683458621ecbe122beb72f0',
        'VCR_EN_EASY_ALL': '8a9b96885f251d1c85f42f84073327f1',
        'VCR_EN_HARD_500': '0a22a85080b6a1f52b1f95e302d43df4',
        'VCR_EN_HARD_100': '1b20f5cbcbeae0b0bec77f7a36143958',
        'VCR_EN_HARD_ALL': '2d8b8b1ee0eba0e0b618fd3aa7d9710e',
        'VCR_ZH_EASY_500': 'beca5fd54176adf44cf94bd9b50cf048',
        'VCR_ZH_EASY_100': '4a86a5678a79844d6d22ab0629c51cd5',
        'VCR_ZH_EASY_ALL': '5050fe7f0027ad2068fd4c7f220edaea',
        'VCR_ZH_HARD_500': '617e3360f75c54455625cb0a8da5c1e7',
        'VCR_ZH_HARD_100': 'b0e38c85f5d5e63894a3b881c372a62b',
        'VCR_ZH_HARD_ALL': '54bbfef448206518b03127ef8b61404c',
    }

    def __init__(self, dataset='VCR_EN_EASY_500', skip_noimg=True):
        super().__init__(dataset, skip_noimg)

        initialize()
        self.language = 'en' if 'EN' in dataset else 'zh'
        self.difficulty = 'easy' if 'EASY' in dataset else 'hard'

    # def build_prompt(self, line):
    #     msgs = super().build_prompt(line)
    #     assert msgs[-1]['type'] == 'text'
    #     if self.language == 'zh':
    #         msgs[-1]['value'] += '图像中被覆盖的文本是什么？请在不输出解释的情况下还原被覆盖的文本。'
    #     else:
    #         msgs[-1]['value'] += ('What is the covered texts in the image? '
    #                               'Please restore the covered texts without outputting the explanations.')
    #     return msgs

    def evaluate(self, eval_file, **judge_kwargs):
        import multiprocessing

        vcr_score_list = {'Exact_Match': [], 'Jaccard': []}
        vcr_score = {'Exact_Match': 0, 'Jaccard': 0}
        logger = get_logger('Evaluation')
        data = load(eval_file)

        lt = len(data)
        lines = [data.iloc[i] for i in range(lt)]

        pool = multiprocessing.Pool()
        manager = multiprocessing.Manager()
        progress_queue = manager.Queue()
        results = []

        overall_results = {str(image_id): {} for image_id in range(len(lines))}

        for instance_id, instance in enumerate(lines):
            results.append(
                pool.apply_async(
                    process_match_single_new,
                    args=(
                        str(instance_id),
                        instance['prediction'],
                        instance['answer'],
                        self.language,
                        progress_queue,
                    ),
                )
            )
        pool.close()

        # Display progress bar
        for _ in tqdm(range(len(results))):
            progress_queue.get()

        pool.join()

        # Merging results into overall_result
        for result in results:
            image_id, result_per_id = result.get()
            overall_results[str(image_id)].update(result_per_id[image_id])
            for blank_id_str in result_per_id[image_id].keys():
                vcr_score_list['Exact_Match'].append(
                    result_per_id[image_id][blank_id_str]['exact_match']
                )
                vcr_score_list['Jaccard'].append(
                    result_per_id[image_id][blank_id_str]['jaccard']
                )
            vcr_score['Exact_Match'] = np.mean(vcr_score_list['Exact_Match'])
            vcr_score['Jaccard'] = np.mean(vcr_score_list['Jaccard'])
        results_out = {
            k: v for i in range(len(results)) for k, v in results[i].get()[1].items()
        }
        results_with_metrics = {
            'Exact_Match': vcr_score['Exact_Match'],
            'Jaccard': vcr_score['Jaccard'],
            'Predictions': results_out,
        }
        score_pth = eval_file.replace(
            '.xlsx', f'{self.language}_{self.difficulty}_score.json'
        )
        dump(results_with_metrics, score_pth)
        logger.info(
            f'VCR successfully finished evaluating {eval_file}, results saved in {score_pth}'
        )
        logger.info('Score: ')
        for key, value in vcr_score.items():
            logger.info('{}:{}'.format(key, value))
