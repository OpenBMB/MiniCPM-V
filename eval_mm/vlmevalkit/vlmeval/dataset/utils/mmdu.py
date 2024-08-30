from ...smp import *

meta_prompt = """
You are an assistant skilled at evaluating the quality of creative text.
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to \
the user question displayed below. You'll need to assess the response on the following dimensions: \
Creativity, Richness, Visual Perception, Logical Coherence, Answer Accuracy and Image Relationship Understanding. \
We will provide you with a creative question and the AI model's response and a reference answer for your evaluation. \
As you begin your assessment, follow this process:
1. Evaluate the AI model's answers on different dimensions, pointing out its strengths or weaknesses \
in each dimension and assigning a score of 1 to 10 for each.
2. Finally, based on the assessments across dimensions, \
provide an overall score of 1 to 10 for the AI model's response.
3. Your scoring should be as stringent as possible and follow the scoring rules below:
In general, the higher the quality of the model's response and its strict adherence to user needs, \
the higher the score. Responses that do not meet user needs will receive lower scores.
Scoring rules:
Creativity:
Scores 1-2 when there is no innovation or uniqueness in the content.
Scores 3-4 when providing partially original content but with low creative quality.
Scores 5-6 when mostly creative but lacks significant novelty, with moderate quality.
Scores 7-8 when having novelty and high-quality content.
Scores 9-10 when highly novel and of exceptional quality compared to the reference answer.
Richness:
Scores 1-2 when lacking depth and breadth, with very limited information.
Scores 3-4 when limited in depth and breadth, with fewer explanations and examples, showing low diversity.
Scores 5-6 when limited in depth and breadth but provides basic necessary information.
Scores 7-8 when providing depth and useful additional information.
Scores 9-10 when providing exceptional depth, breadth, and high diversity compared to the reference answer.
Visual Perception:
Scores 1-2 when the description of the visual information in the image contains errors or \
is significantly inconsistent with the content of the image.
Scores 3-4 When the description of the visual information in the image reflects only a small amount \
of the image's information and contains some errors.
Scores 5-6 when the description of the visual information in the image includes the basic information \
of the image but contains minimal information.
Scores 7-8 when the description of the visual information in the image matches the image well and is rich in content, \
providing a substantial amount of information about the image.
Scores 9-10 when the description of the visual information in the image not only matches the image \
but also is more detailed and informative compared to the reference answer, providing more information about the image.
Logical Coherence:
Scores 1-2 when entirely incoherent, lacking any logic, and not matching the question or known information.
Scores 3-4 when somewhat coherent but with many logical errors or inconsistencies.
Scores 5-6 when mostly coherent, with few errors, but may struggle to maintain complete coherence in complex situations.
Scores 7-8 when excellent logical handling, very few errors.
Scores 9-10 when flawless logic, impeccable in handling complexity, \
and significantly higher logical coherence compared to the reference answer.
Answer Accuracy:
Scores 1-2 when the answer is significantly inconsistent with the question or contains obvious errors.
Scores 3-4 when the answer is partially correct but contains some errors or is incomplete.
Scores 5-6 when the answer is basically correct but lacks details or is not sufficiently detailed.
Scores 7-8 when the answer is accurate and detailed, fully corresponding to the question.
Scores 9-10 when the answer is not only accurate and detailed but also provides additional useful information, \
exceeding expectations.
Image Relationship Understanding:
Scores 1-2 when there are significant errors or confusion in distinguishing and describing different images, \
unable to correctly identify and relate the content of the images.
Scores 3-4 when the description of different images reflects only minimal distinguishing information, \
contains some errors and confusion, and fails to clearly differentiate and relate the images.
Scores 5-6 when the description of different images includes basic distinguishing information, \
is able to correctly identify and relate the images in a basic manner, \
but the information provided is minimal and lacks detail.
Scores 7-8 when the description of different images is accurate and detailed, \
clearly distinguishing and relating the images, \
with rich content that points out the main commonalities and differences between the images.
Scores 9-10 when the description of different images is not only accurate and detailed but also \
provides richer information and analysis, clearly distinguishing and relating the images, \
more comprehensively pointing out the commonalities and differences \
between the images compared to the reference answer.
Overall Score:
Scores 1-2 when irrelevant to the question, factually incorrect, or generates harmful content.
Scores 3-4 when no serious errors, mostly harmless, but of low quality and does not meet requirements.
Scores 5-6 when basically meeting requirements but performing poorly in some dimensions, with moderate quality.
Scores 7-8 when performing well in all dimensions.
Scores 9-10 when fully addressing user questions and all requirements, significantly surpassing the reference answer.
Please remember, you must evaluate and explain before scoring. After your explanation for each dimension, \
add the score for that dimension. Finally, at the end of your response, \
in the format of the dictionary (including brackets), return all your scoring results, \
ensuring your scores are integers:
{'Dimension One': Score, 'Dimension Two': Score, ..., 'Overall Score': Score}, \
for example: {'Creativity': 9, 'Richness': 6, ..., 'Overall Score': 7}.\n
"""
question_begin_prompt = '[Question]'
reference_begin_prompt = '[The Start of Reference Answer]'
reference_end_prompt = '[The End of Reference Answer]'
answers_begin_prompt = '[The Start of Assistant’s Answer]'
answers_end_prompt = '[The End of Assistant’s Answer]'


def mmdu_score(model, line):
    question = eval(line['question'])
    gt = eval(line['answer'])
    prediction = eval(line['prediction'])

    DIMS = [
        'Creativity', 'Richness', 'Visual Perception', 'Logical Coherence',
        'Answer Accuracy', 'Image Relationship Understanding', 'Overall Score'
    ]

    all_result_dict = []
    logs = []
    for j in range(len(question)):
        try:
            prompt = meta_prompt + question_begin_prompt + '\n' + question[j] + '\n\n' + \
                reference_begin_prompt + '\n' + gt[j] + '\n' + reference_end_prompt + '\n\n' + \
                answers_begin_prompt + '\n' + prediction[j] + '\n' + answers_end_prompt
            response = model.generate(prompt)
            start_index = response.find('{')
            end_index = response.rfind('}') + 1
            dictionary_str = response[start_index: end_index]
            result_dict = eval(dictionary_str)
            all_result_dict.append(result_dict)
            if all([x in result_dict for x in DIMS]):
                logs.append('Succeed')
            else:
                logs.append(
                    f'Following Dims are not in results of turn {j}: '
                    f'{",".join([x for x in DIMS if x not in result_dict])}'
                )
        except Exception as e:
            print({e})
            all_result_dict.append({d: None for d in DIMS})
            logs.append(str(e))

    df = pd.DataFrame(all_result_dict)
    return dict(res=df, log='\n'.join(logs))
