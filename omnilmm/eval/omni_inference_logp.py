import os
import json
import tqdm
import torch
import base64
import torch.utils.data as torch_data

from typing import List
from functools import partial

from omnilmm.eval.omnilmm_vqa import init_omnilmm
from omnilmm.train.train_utils import encode_multimodal_preference_sample
from omnilmm.data.datasets import SingleDataSourceDataset
from omnilmm.eval.omnilmm_inference_logp import get_batch_logps, pretty_print, get_multimodal_sample_logps, write_logp_to_preference_tsv
from omnilmm.train.train_utils import omni_preprocess
from omnilmm.eval.omni_lmm_chat import init_omni_lmm


class ZephyrPreferenceInferenceDataset(torch_data.Dataset):
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
            sample, self.tokenizer, self.mm_cfg, preprocess_func=omni_preprocess)
        # print(f'Rej:')
        # pretty_print(rej_data_dict, self.tokenizer)
        # print(f'Win:')
        # pretty_print(win_data_dict, self.tokenizer)
        return rej_data_dict, win_data_dict

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    # model, img_processor, image_token_len, tokenizer = init_omni_lmm(
    #     '/home/yutianyu/Zephyr_checkpoints/SFT_exp/omni_lmm_12b_SFT-6node_5kPT_SFT_stage3-caterpillar-stage2_mix#caterpillar-stage3_lvis#caterpillar-stage3_svit#caterpillar-stage3_sharegpt4v#llava#unimm-chat-134#222#600#101#157#117/checkpionts/checkpoint-4000', tune_clip=True)
    model, img_processor, image_token_len, tokenizer = init_omni_lmm(
        '/home/yutianyu/Zephyr_checkpoints/DPO_exp/omni_lmm_12b_DPO-all_data_SFT-dpo_omni_2102-trs#dpo_omni_1065-trs#dpo_omni_2566-trs-2102#1065#2566/checkpoints/checkpoint-150', tune_clip=True)
    
    use_im_start_end = True

    data_dir = '/data/public/multimodal/multimodal_data/dpo/refined_test/'
    # tsv_files = ['omnilmm_rewrite_cvpr_dpo_with_per_token_vqa_logp_train-1401.tsv']
    tsv_files = ['omnilmm_rewrite-by_translate_cvpr_dpo_with_per_token_vqa_logp_train-1401.tsv']
    
    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_rewrite_cvpr_1020_1027-1124_good/'
    # tsv_files = ['omnilmm_rewrite-by_mianbi_cvpr_zj1020_zj1027-1124_dpo_with_per_token_vqa_logp_train-2102.tsv']
    tsv_files = ['omnilmm_rewrite-by_trans_cvpr_zj1020_zj1027-1124_dpo_with_per_token_vqa_logp_train-2102.tsv']

    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_rewrite_llava_1122-1123_1128/'
    # tsv_files = ['llava_rewrite-by_trans_zj1122-1123_1128_good_dpo_with_per_token_vqa_logp_train-1065.tsv']
    
    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_diverse_20231218-1229_all/'
    # tsv_files = ['diverse_1218_1229_dpo_with_per_token_vqa_logp_train-2566.tsv']
    
    # data_dir = '/data/public/multimodal/multimodal_data/dpo/eval/reward_bench/tsvs_clean'
    # tsv_files = ['RM_bench_clean_diff1_dpo_with_per_token_vqa_logp_train-893.tsv', 'RM_bench_clean_diff2_dpo_with_per_token_vqa_logp_train-262.tsv', 'RM_bench_clean_diff3_dpo_with_per_token_vqa_logp_train-90.tsv']

    for tsv_filename in tsv_files:
        win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list = get_multimodal_sample_logps(
            model, tokenizer, data_dir, [tsv_filename], image_token_len, img_processor, use_im_start_end, ds_class=ZephyrPreferenceInferenceDataset)
        logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list,
                     rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list))

        tsv_filepath = os.path.join(data_dir, tsv_filename)
        # write_logp_to_preference_tsv(
        #     tsv_filepath, f'{data_dir}/omni_stage3_4k_logp', logps, overwrite_logps=True)
        # write_logp_to_preference_tsv(
        #     tsv_filepath, f'{data_dir}/omni_stage3_4k_logp_by_trans', logps, overwrite_logps=True)
        # write_logp_to_preference_tsv(
        #     tsv_filepath, f'{data_dir}/omni_stage3_4k_logp_SFT', logps, overwrite_logps=True)
        write_logp_to_preference_tsv(
            tsv_filepath, f'{data_dir}/omni_stage3_4k_logp_SFT_by_trans', logps, overwrite_logps=True)
