from ...smp import *


def AMBER_rating(data_file):
    data = load(data_file)
    stats = defaultdict(dict)
    lt = len(data)
    category_mapping = {
        'discriminative-attribute-state': 'Attribute',
        'discriminative-attribute-number': 'Attribute',
        'discriminative-attribute-action': 'Attribute',
        'discriminative-hallucination': 'Existence',
        'discriminative-relation': 'Relation',
        'relation': 'Relation'
    }

    for i in range(lt):
        item = data.iloc[i]
        category = item['category']
        image_path = item['image_path']
        score = item['score']

        new_category = category_mapping.get(category, category)

        if image_path not in stats[new_category]:
            stats[new_category][image_path] = []
        stats[new_category][image_path].append(score)

    def acc(key):
        res = stats[key]
        values = []
        for val in res.values():
            values.extend(val)
        return np.mean(values) * 100

    scores = {}
    for k in stats:
        scores[k] = acc(k)

    scores['Avg ACC'] = np.mean(list(scores.values()))
    ret = d2df(scores)
    return ret


def MME_rating(data_file):
    data = load(data_file)
    stats = defaultdict(dict)
    lt = len(data)
    for i in range(lt):
        item = data.iloc[i]
        category = item['category']
        image_path = item['image_path']
        score = item['score']
        if image_path not in stats[category]:
            stats[category][image_path] = []
        stats[category][image_path].append(score)

    def acc(key, mode='normal'):
        res = stats[key]
        values = []
        for val in res.values():
            if mode == 'normal':
                values.extend(val)
            elif mode == 'plus':
                values.append(val[0] * val[1])
        return np.mean(values) * 100

    scores = {}
    for k in stats:
        scores[k] = acc(k) + acc(k, 'plus')

    super_cates = dict(
        perception=[
            'OCR', 'artwork', 'celebrity', 'color', 'count', 'existence',
            'landmark', 'position', 'posters', 'scene'
        ],
        reasoning=['code_reasoning', 'commonsense_reasoning', 'numerical_calculation', 'text_translation']
    )

    ret = {}
    for sc, cate_list in super_cates.items():
        base = 0
        for c in cate_list:
            base += scores[c]
        ret[sc] = base
    ret.update(scores)
    ret = d2df(ret)
    return ret


def Hallusion_rating(data_file):
    def calc_fAcc(data):
        res = defaultdict(list)
        lt = len(data)
        for i in range(lt):
            line = data.iloc[i]
            res[f"{line['l2-category']}_{line['set_id']}_{line['figure_id']}"].append(line['score'])
        return np.mean([np.all(x) for x in res.values()]) * 100

    def calc_qAcc(data):
        res = defaultdict(list)
        lt = len(data)
        for i in range(lt):
            line = data.iloc[i]
            res[f"{line['l2-category']}_{line['set_id']}_{line['question_id']}"].append(line['score'])
        return np.mean([np.all(x) for x in res.values()]) * 100

    def calc_aAcc(data):
        return np.mean(data['score']) * 100

    data = load(data_file)
    data['set_id'] = [x.split('_')[3] for x in data['index']]
    data['figure_id'] = [x.split('_')[4] for x in data['index']]
    data['question_id'] = [x.split('_')[5] for x in data['index']]

    res = dict(split=[], aAcc=[], fAcc=[], qAcc=[])
    res['split'].append('Overall')
    res['aAcc'].append(calc_aAcc(data))
    res['fAcc'].append(calc_fAcc(data))
    res['qAcc'].append(calc_qAcc(data))

    if 'category' in data:
        cates = list(set(data['category']))
        for c in cates:
            sub = data[data['category'] == c]
            res['split'].append(c)
            res['aAcc'].append(calc_aAcc(sub))
            res['fAcc'].append(calc_fAcc(sub))
            res['qAcc'].append(calc_qAcc(sub))

    if 'l2-category' in data:
        cates = list(set(data['l2-category']))
        for c in cates:
            sub = data[data['l2-category'] == c]
            res['split'].append(c)
            res['aAcc'].append(calc_aAcc(sub))
            res['fAcc'].append(calc_fAcc(sub))
            res['qAcc'].append(calc_qAcc(sub))
    ret = pd.DataFrame(res)
    return ret


def POPE_rating(data_file):
    def cal_f1_score(y_true, y_pred):
        tp = sum((y_true == 1) & (y_pred == 1))
        fp = sum((y_true == 0) & (y_pred == 1))
        fn = sum((y_true == 1) & (y_pred == 0))

        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        return f1_score, precision, recall

    data = load(data_file)
    data = data.assign(category=data['category'].str.split(',')).explode('category')
    data['index'] = range(len(data))
    res = dict(split=[], Overall=[], acc=[], precision=[], recall=[])
    y_true = np.array([1 if i == 'Yes' else 0 for i in data['answer']])
    y_pred = np.array([1 if i == 'Yes' else 0 for i in data['extracted']])
    f1_score, precision, recall = cal_f1_score(y_true, y_pred)
    res['split'].append('Overall')
    res['Overall'].append(f1_score * 100)
    res['acc'].append(np.mean(data['score']) * 100)
    res['precision'].append(precision * 100)
    res['recall'].append(recall * 100)

    if 'category' in data:
        cates = list(set(data['category']))
        cates = [c for c in cates if not pd.isna(c)]
        for c in cates:
            sub = data[data['category'] == c]
            y_true = np.array([1 if i == 'Yes' else 0 for i in sub['answer']])
            y_pred = np.array([1 if i == 'Yes' else 0 for i in sub['extracted']])
            f1_score, precision, recall = cal_f1_score(y_true, y_pred)
            res['split'].append(c)
            res['Overall'].append(f1_score * 100)
            res['acc'].append(np.mean(sub['score']) * 100)
            res['precision'].append(precision * 100)
            res['recall'].append(recall * 100)

    ret = pd.DataFrame(res)
    return ret


def default_rating(data_file):
    data = load(data_file)
    res = {}
    res['Overall'] = np.mean(data['score']) * 100
    if 'category' in data:
        cates = list(set(data['category']))
        cates = [c for c in cates if not pd.isna(c)]
        cates.sort()
        for c in cates:
            sub = data[data['category'] == c]
            res[c] = np.mean(sub['score']) * 100
    if 'l2-category' in data:
        cates = list(set(data['l2-category']))
        cates = [c for c in cates if not pd.isna(c)]
        cates.sort()
        for c in cates:
            sub = data[data['l2-category'] == c]
            res[c] = np.mean(sub['score']) * 100
    ret = d2df(res)
    return ret


def YOrN_match_prompt(line):
    tmpl = (
        'You are an AI assistant who will help me to match an answer with two options of a question. '
        'The options are only Yes / No. '
        'You are provided with a question and an answer, '
        'and you need to find which option (Yes / No) is most similar to the answer. '
        'If the meaning of all options are significantly different from the answer, output Unknown. '
        'Your should output a single word among the following 3 choices: Yes, No, Unknown.\n'
        'Example 1: \n'
        "Question: Is the word in this image 'Hello'?\nAnswer: The word in this image is 'Hello'.\nYour output: Yes\n"
        'Example 2: \n'
        "Question: Is the word in this image 'Hello'?\n"
        "Answer: The word in this image is not 'Hello'.\nYour output: No\n"
        'Example 3: \n'
        'Question: {}?\nAnswer: {}\nYour output: '
    )
    return tmpl.format(line['question'], line['prediction'])


def YOrN_Extraction(output):
    s = output.lower()
    words = process_punctuation(s).split()
    if 'yes' in words and 'no' not in words:
        return 'Yes'
    if 'yes' not in words and 'no' in words:
        return 'No'
    return 'Unknown'


def YOrN_auxeval(model, line):
    prompt = YOrN_match_prompt(line)
    retry = 5
    for i in range(retry):
        output = model.generate(prompt, temperature=0.5 * i)
        ans = YOrN_Extraction(output)
        if ans != 'Unknown':
            return ans
    return 'Unknown'
