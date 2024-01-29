import os

from omnilmm.eval.llava_vqa import init_llava
from omnilmm.eval.muffin_inference_logp import write_logp_to_preference_tsv, get_multimodal_sample_logps


if __name__ == '__main__':
    model, img_processor, image_token_len, tokenizer = init_llava(
        '/home/yutianyu/llava_checkpoint')
    # model, img_processor, image_token_len, tokenizer = init_llava('../Muffin_checkpoints/llava_exp/llava_13b_dpo-only_SFT_our-dpo_cvpr_llava-1/checkpionts/checkpoint-60/')

    use_im_start_end = True

    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231012'
    # tsv_files = ['dpo_preference_1012_0-686.tsv']
    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231005'
    # tsv_files = ['dpo_preference_0-225.tsv']

    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_llava_7b_v1_preference_hallonly'
    # tsv_files = ['dpo_preference_llava_7b_v1_preference_hallonly_0-2122.tsv']
    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_llava_7b_v1_preference_nothall'
    # tsv_files = ['dpo_preference_llava_7b_v1_preference_nothall_0-7300.tsv']
    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_main'
    # tsv_files = ['dpo_cvpr-1401.tsv', 'dpo_cvpr-100.tsv', 'dpo_cvpr-200.tsv', 'dpo_cvpr-400.tsv', 'dpo_cvpr-800.tsv']

    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_CVPR24_llavarlhf-onlyhall'
    # tsv_files = ['dpo_llavarlhf-2122.tsv', 'dpo_llavarlhf-800.tsv', 'dpo_llavarlhf-400.tsv', 'dpo_llavarlhf-200.tsv', 'dpo_llavarlhf-100.tsv']

    # data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231122to1123-1'
    # tsv_files = ['dpo_preference_1122to1123-1_0-845.tsv']

    data_dir = '/data/public/multimodal/multimodal_data/dpo/DPO_preference_20231128to1128'
    tsv_files = ['dpo_preference_1128to1128_0-220.tsv']

    # win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list = get_multimodal_sample_logps(model, tokenizer, data_dir, tsv_files, image_token_len, img_processor, use_im_start_end)
    # logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list))

    for tsv_filename in tsv_files:
        win_logp_list, win_avg_logp_list, win_per_token_logp_list, rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list = get_multimodal_sample_logps(
            model, tokenizer, data_dir, [tsv_filename], image_token_len, img_processor, use_im_start_end)
        logps = list(zip(win_logp_list, win_avg_logp_list, win_per_token_logp_list,
                     rej_logp_list, rej_avg_logp_list, rej_per_token_logp_list))

        tsv_filepath = os.path.join(data_dir, tsv_filename)
        # write_logp_to_preference_tsv(tsv_filepath, f'{data_dir}/dpo_with_per_token_llava_after_SFT_logp_train', logps)
        write_logp_to_preference_tsv(
            tsv_filepath, f'{data_dir}/dpo_with_per_token_llava_logp_train', logps)
