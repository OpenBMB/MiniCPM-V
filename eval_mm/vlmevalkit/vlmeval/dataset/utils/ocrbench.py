from ...smp import *


def OCRBench_eval(eval_file):
    OCRBench_score = {
        'Regular Text Recognition': 0,
        'Irregular Text Recognition': 0,
        'Artistic Text Recognition': 0,
        'Handwriting Recognition': 0,
        'Digit String Recognition': 0,
        'Non-Semantic Text Recognition': 0,
        'Scene Text-centric VQA': 0,
        'Doc-oriented VQA': 0,
        'Key Information Extraction': 0,
        'Handwritten Mathematical Expression Recognition': 0
    }

    logger = get_logger('Evaluation')

    data = load(eval_file)
    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]
    for i in tqdm(range(len(lines))):
        line = lines[i]
        predict = str(line['prediction'])
        answers = eval(line['answer'])
        category = line['category']
        if category == 'Handwritten Mathematical Expression Recognition':
            for j in range(len(answers)):
                answer = answers[j].strip().replace('\n', ' ').replace(' ', '')
                predict = predict.strip().replace('\n', ' ').replace(' ', '')
                if answer in predict:
                    OCRBench_score[category] += 1
                    break
        else:
            for j in range(len(answers)):
                answer = answers[j].lower().strip().replace('\n', ' ')
                predict = predict.lower().strip().replace('\n', ' ')
                if answer in predict:
                    OCRBench_score[category] += 1
                    break

    final_score_dict = {}
    final_score_dict['Text Recognition'] = (
        OCRBench_score['Regular Text Recognition'] + OCRBench_score['Irregular Text Recognition']
        + OCRBench_score['Artistic Text Recognition'] + OCRBench_score['Handwriting Recognition']
        + OCRBench_score['Digit String Recognition'] + OCRBench_score['Non-Semantic Text Recognition']
    )
    final_score_dict['Scene Text-centric VQA'] = OCRBench_score['Scene Text-centric VQA']
    final_score_dict['Doc-oriented VQA'] = OCRBench_score['Doc-oriented VQA']
    final_score_dict['Key Information Extraction'] = OCRBench_score['Key Information Extraction']
    final_score_dict['Handwritten Mathematical Expression Recognition'] = \
        OCRBench_score['Handwritten Mathematical Expression Recognition']
    final_score_dict['Final Score'] = (
        final_score_dict['Text Recognition'] + final_score_dict['Scene Text-centric VQA']
        + final_score_dict['Doc-oriented VQA'] + final_score_dict['Key Information Extraction']
        + final_score_dict['Handwritten Mathematical Expression Recognition']
    )
    final_score_dict['Final Score Norm'] = float(final_score_dict['Final Score']) / 10
    score_pth = eval_file.replace('.xlsx', '_score.json')
    dump(final_score_dict, score_pth)
    logger.info(f'OCRBench_eval successfully finished evaluating {eval_file}, results saved in {score_pth}')
    logger.info('Score: ')
    for key, value in final_score_dict.items():
        logger.info('{}:{}'.format(key, value))
