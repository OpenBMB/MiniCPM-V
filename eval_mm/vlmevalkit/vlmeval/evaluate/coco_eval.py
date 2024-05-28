from vlmeval.smp import *
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


class COCO_Caption_Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
            # (Meteor(), "METEOR"), # need java version 11.0.16+
            (Rouge(), 'ROUGE_L'),
            (Cider(), 'CIDEr'),
            # (Spice(), "SPICE"), # need java version 11.0.16+
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print('%s: %0.3f' % (m, sc * 100))
                total_scores['Bleu'] = [x * 100 for x in score]
            else:
                print('%s: %0.3f' % (method, score * 100))
                total_scores[method] = score * 100

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores


def COCO_eval(eval_file, nproc=4, verbose=False):
    logger = get_logger('Evaluation')

    data = load(eval_file)

    lt = len(data)
    lines = [data.iloc[i] for i in range(lt)]
    ref = {}
    gt = {}
    for i, line in enumerate(lines):
        ref[str(i)] = [str(line['prediction'])]
        gt[str(i)] = eval(line['answer'])

    scorer = COCO_Caption_Scorer(ref, gt)
    coco_caption_score_dict = scorer.compute_scores()

    score_pth = eval_file.replace('.xlsx', '_score.json')
    dump(coco_caption_score_dict, score_pth)
    logger.info(f'COCO_eval successfully finished evaluating {eval_file}, results saved in {score_pth}')
    logger.info('Score: ')
    for key, value in coco_caption_score_dict.items():
        logger.info('{}:{}'.format(key, value))


def parse_args():
    parser = argparse.ArgumentParser(description='Inference LLM Answers. ')
    parser.add_argument('--data', type=str, help='The question set for inference, in excel / tsv / json format. ')
    parser.add_argument('--nproc', type=int, default=4)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    COCO_eval(eval_file=args.data, nproc=args.nproc, verbose=args.verbose)
