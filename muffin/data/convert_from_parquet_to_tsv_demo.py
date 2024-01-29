import os
import json
import tqdm
import base64
import pandas

from muffin.data.tsv_file_op import multimodal_img_tsv_writer

datasetname = 'unimm-chat'
output_name = datasetname
file_list = ['filtered_sft_coco_vqa_gpt3.5_0.parquet',
             'filtered_sft_coco_vqa_gpt3.5_25000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_60000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_100000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_30000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_65000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_10000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_35000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_70000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_105000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_40000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_75000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_110000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_45000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_80000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_115000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_50000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_85000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_15000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_5000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_90000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_20000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_55000.parquet',
             'filtered_sft_coco_vqa_gpt3.5_95000.parquet']
file_list = file_list[0:2]
parquet_data_folder = '/data/public/multimodal/multimodal_data/sft_data/coco_based/vqa_chat_20230628/'


# TODO: You have to re-implement regarding your source data format
def parquet_data_stream():
    for file in tqdm.tqdm(file_list):
        list_data = list(pandas.read_parquet(
            os.path.join(parquet_data_folder, file)).iterrows())
        for idx, value in list_data:
            img_buffer = base64.b64encode(value['BUFFER']).decode('utf-8')
            text = base64.b64encode(json.dumps(
                list(value['TEXT'])).encode('utf-8')).decode('utf-8')
            img_path = value['IMAGE_ID']

            dataset_name = datasetname
            origin_dataset = ''
            origin_split = ''
            origin_split_inner_idx = f'{idx}'

            yield dataset_name, img_buffer, text, origin_dataset, origin_split, origin_split_inner_idx, img_path


# TODO: output data with be ./DATA_NAME-DATA_SIZE.tsv and ./DATA_NAME-DATA_SIZE.tsv.lineidx
multimodal_img_tsv_writer(parquet_data_stream(),
                          f'{parquet_data_folder}/{output_name}')
