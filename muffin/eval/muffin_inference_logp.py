import os
import json
import tqdm
import torch
import base64
import torch.utils.data as torch_data

from typing import List
from functools import partial

from muffin.eval.muffin_vqa import init_muffin
from muffin.train.train_utils import encode_multimodal_preference_sample, SFT_collator_fn
from muffin.data.datasets import SingleDataSourceDataset
from muffin.data.tsv_file_op import multimodal_img_tsv_writer_prev
from muffin.data.tsv_file import TSVFile


def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    # print(per_token_logps.shape, labels.shape)
    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob


class PreferenceInferenceDataset(torch_data.Dataset):
    def __init__(self,
                 data_dir,
                 tokenizer,
                 tsv_filenames: List[str],
                 image_token_len,
                 img_processor,
                 use_im_start_end):
        if 'DPO_preference_llava' in data_dir or 'llavarlhf' in tsv_filenames[0]:
            self.data = SingleDataSourceDataset(
                'dpo_preference_llava_7b_v1_preference_hallonly', data_dir, tsv_filenames)
        else:
            self.data = SingleDataSourceDataset(
                'dpo_1005', data_dir, tsv_filenames)

        self.mm_cfg = {
            'image_processor': img_processor,
            'is_multimodal': True,
            'image_token_len': image_token_len,
            'use_im_start_end': use_im_start_end
        }
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sample = self.data[index]
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(
            sample, self.tokenizer, self.mm_cfg)
        return rej_data_dict, win_data_dict

    def __len__(self):
        return len(self.data)


def pretty_print(data_dict, tokenizer):
    input_ids = data_dict['input_ids']
    input_str = tokenizer.decode(input_ids)
    print(f'input_ids.shape={input_ids.shape}\ninput_str is {input_str}')

    label_ids = data_dict['labels']
    print(f'label_ids.shape={input_ids.shape}')
    for i, o in zip(input_ids, label_ids):
        i_tok = tokenizer.convert_ids_to_tokens(i.item())
        o_tok = tokenizer.convert_ids_to_tokens(
            o.item()) if o.item() != -100 else '[SKIP]'
        print(f'{i_tok:10s} => {o_tok:10s}')


def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out


def preference_collator_fn(instances, pad_token_id):
    rej_instances, win_instances = list(zip(*instances))
    rej_batch = SFT_collator_fn(rej_instances, pad_token_id)
    win_batch = SFT_collator_fn(win_instances, pad_token_id)

    concatenated_input_ids = concate_pad(
        win_batch['input_ids'], rej_batch['input_ids'], pad_token_id)
    concatenated_labels = concate_pad(
        win_batch['labels'], rej_batch['labels'], -100)
    # concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)

    batch = dict(
        concatenated_input_ids=concatenated_input_ids,
        concatenated_labels=concatenated_labels,
        # concatenated_attention_mask=concatenated_attention_mask,
        win_input_ids=win_batch['input_ids'],
        rej_input_ids=rej_batch['input_ids'],
        win_labels=win_batch['labels'],
        rej_labels=rej_batch['labels'],
        # win_attention_mask=win_batch['attention_mask'],
        # rej_attention_mask=rej_batch['attention_mask'],
        images=win_batch['images'],
    )
    return batch


def get_multimodal_sample_logps(model, tokenizer, data_dir, tsv_files, image_token_len, img_processor, use_im_start_end, ds_class=None):
    if ds_class is None:
        ds_class = PreferenceInferenceDataset
    dataset = ds_class(data_dir=data_dir,
                       tokenizer=tokenizer,
                       tsv_filenames=tsv_files,
                       image_token_len=image_token_len,
                       img_processor=img_processor,
                       use_im_start_end=use_im_start_end)
    collate_fn = partial(preference_collator_fn,
                         pad_token_id=tokenizer.pad_token_id)
    dataloader = torch_data.DataLoader(dataset, batch_size=1, collate_fn=collate_fn,
                                       num_workers=5, shuffle=False)

    win_logp_list = []
    rej_logp_list = []

    win_avg_logp_list = []
    rej_avg_logp_list = []

    win_per_token_logp_list = []
    rej_per_token_logp_list = []

    with torch.inference_mode():
        for batch in tqdm.tqdm(dataloader):
            for key in ['win', 'rej']:
                input_ids = batch[f'{key}_input_ids'].cuda()
                labels = batch[f'{key}_labels'].cuda()
                # attention_mask = batch[f'{key}_attention_mask'].cuda()

                output = model(
                    input_ids=input_ids,
                    labels=labels,
                    # attention_mask=attention_mask,
                    images=batch['images'].half().cuda()
                )
                per_token_logp, log_prob, average_log_prob = get_batch_logps(
                    output.logits, labels, return_all=True)

                # print(per_token_logp.shape, input_ids.shape, labels.shape, flush=True)
                assert per_token_logp.size(1) >= input_ids.size(1) - 1
                per_token_logp = per_token_logp.tolist()
                # per_token_logp = [x[:input_ids[i].ne(tokenizer.pad_token_id).sum().item()] for i, x in enumerate(per_token_logp)]
                log_prob = log_prob.tolist()
                average_log_prob = average_log_prob.tolist()

                if key == 'win':
                    win_logp_list += log_prob
                    win_avg_logp_list += average_log_prob
                    win_per_token_logp_list += per_token_logp
                else:
                    rej_logp_list += log_prob
                    rej_avg_logp_list += average_log_prob
                    rej_per_token_logp_list += per_token_logp
                # print(f'{key} logits in {output.logits.shape}, logp in {log_prob.shape} avg_logp in {average_log_prob.shape}')

    return win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list


def write_logp_to_preference_tsv(tsv_filename, out_tsv_filename, logps, overwrite_logps=False):
    origin_data = TSVFile(tsv_filename)

    out_data = []
    for line, logp_data in tqdm.tqdm(zip(origin_data, logps)):
        text_b64 = line[2]
        text = base64.b64decode(text_b64).decode('utf-8')
        preference_data = json.loads(text)
        if len(preference_data) == 4:
            assert overwrite_logps, 'Found existing logp data, pass overwrite_logps=True to force overwritting'
            preference_data[3] = logp_data
        else:
            assert len(
                preference_data) == 3, f'Undefined data structure, expecting [Q, Win, Rej], got {text}'
            preference_data.append(logp_data)

        line[2] = base64.b64encode(json.dumps(
            preference_data).encode('utf-8')).decode('utf-8')
        out_data.append(line)

    multimodal_img_tsv_writer_prev(out_data, out_tsv_filename)


if __name__ == '__main__':
    # model, img_processor, image_token_len, tokenizer = init_muffin('../Muffin_checkpoints/310m_pretrain_100k_SFT_M3IT_2800_then_M3IT-LVA-UniMM-SYNTHEDOG_2800/')
    # model, img_processor, image_token_len, tokenizer = init_muffin('../Muffin_checkpoints/vqa_exp/muffin_13b_SFT-no_crop_0.99-vqav2-train#unimm-chat-50#50-beit3_large_patch16_448/checkpionts/checkpoint-2800/')
    # model, img_processor, image_token_len, tokenizer = init_muffin('/home/yutianyu/Muffin_checkpoints/vqa_exp/muffin_13b_SFT-vqa_with_unimmchat_run2-vqav2-train#unimm-chat-50#50-beit3_large_patch16_448/checkpionts/checkpoint-1200/')
    # model, img_processor, image_token_len, tokenizer = init_muffin('../Muffin_checkpoints/SFT_exp/muffin_13b_SFT-Muffin_QA_GPT-4v_1102_SFT_bsz512-gpt4v_detailed_1102-1-beit3_large_patch16_448/checkpionts/checkpoint-50/')
    # model, img_processor, image_token_len, tokenizer = init_muffin('../Muffin_checkpoints/SFT_exp/muffin_13b_SFT-Muffin_QA_GPT-4v_1102_SFT_bsz512-gpt4v_detailed_1102-1-beit3_large_patch16_448/checkpionts/checkpoint-100/')
    # model, img_processor, image_token_len, tokenizer = init_muffin('../Muffin_checkpoints/SFT_exp/muffin_13b_SFT-Muffin_QA_GPT-4v_1102_SFT_bsz512-vqav2-train#gpt4v_detailed_1102-1#1-beit3_large_patch16_448/checkpionts/checkpoint-40/')
    # model, img_processor, image_token_len, tokenizer = init_muffin('../Muffin_checkpoints/SFT_exp/muffin_13b_SFT-Muffin_QA_GPT-4v_1102_SFT_bsz512-gpt4v_detailed_1102-1-beit3_large_patch16_448/checkpionts/checkpoint-20/')

    model, img_processor, image_token_len, tokenizer = init_muffin(
        '/home/yutianyu/Muffin_checkpoints/SFT_exp/muffin_13b_SFT-Muffin_QA_win_SFT_combine-vqav2-train#dpo_sftwin_checked_1005-1026#dpo_sftwin_checked_1103-1106-1#1#1-beit3_large_patch16_448/checkpionts/checkpoint-20/')

    use_im_start_end = True

    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231012'
    # tsv_files = ['dpo_preference_1012_0-686.tsv']
    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231005'
    # tsv_files = ['dpo_preference_0-225.tsv']

    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_llava_7b_v1_preference_hallonly'
    # tsv_files = ['dpo_preference_llava_7b_v1_preference_hallonly_0-2122.tsv']

    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_llava_7b_v1_preference_nothall'
    # tsv_files = ['dpo_preference_llava_7b_v1_preference_nothall_0-7300.tsv']

    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_1005-1026_good'
    # tsv_files = ['dpo_1005_1016_1019-23-24-25-26_merge-799.tsv']

    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main'
    tsv_files = ['dpo_cvpr-1401.tsv', 'dpo_cvpr-100.tsv',
                 'dpo_cvpr-200.tsv', 'dpo_cvpr-400.tsv', 'dpo_cvpr-800.tsv']

    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_human_inspected_1005-1026'
    tsv_files = ['DPO_preference_checked_1005-1026-545.tsv']

    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_human_inspected_1103-1106'
    tsv_files = ['DPO_preference_checked_1103-1106-521.tsv']

    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_llavarlhf-onlyhall'
    # tsv_files = ['dpo_llavarlhf-2122.tsv', 'dpo_llavarlhf-800.tsv', 'dpo_llavarlhf-400.tsv', 'dpo_llavarlhf-200.tsv', 'dpo_llavarlhf-100.tsv']

    # win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list = get_multimodal_sample_logps(model, tokenizer, data_dir, tsv_files, image_token_len, img_processor, use_im_start_end)
    # logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list))

    for tsv_filename in tsv_files:
        win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list = get_multimodal_sample_logps(
            model, tokenizer, data_dir, [tsv_filename], image_token_len, img_processor, use_im_start_end)
        logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list,
                     rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list))

        tsv_filepath = os.path.join(data_dir, tsv_filename)
        # write_logp_to_preference_tsv(tsv_filepath, f'{data_dir}/dpo_with_per_token_logp_train', logps)
        # write_logp_to_preference_tsv(tsv_filepath, f'{data_dir}/dpo_with_per_token_vqa_logp_train', logps)
        # write_logp_to_preference_tsv(tsv_filepath, f'{data_dir}/dpo_with_per_token_docrop_vqa_logp_train', logps)
        # write_logp_to_preference_tsv(tsv_filepath, f'{data_dir}/dpo_with_per_token_gpt4v1107wvqa_logp_train', logps)
        # write_logp_to_preference_tsv(tsv_filepath, f'{data_dir}/dpo_with_per_token_gpt4v1107_logp_train', logps)

        write_logp_to_preference_tsv(
            tsv_filepath, f'{data_dir}/dpo_with_per_token_after_SFT_with_vqa_logp_train', logps)
