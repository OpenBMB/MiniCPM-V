import io
import gc
import json
import random
import numpy
import base64

import os.path as op
import torch.utils.data as torch_data

from PIL import Image
from typing import List, Iterator
from muffin.data.tsv_file import TSVFile
from muffin.data.data_processors import register_data_processor


class MultimodalQADataset(torch_data.Dataset):
    def __init__(self, qa_file, question_process):
        '''
        qa_file: jsonl file that each line is a dict like {
            'image': b64img,
            'question': question_text
        }
        '''
        super().__init__()

        self.qa_file = qa_file
        self.qa_data = [json.loads(line) for line in open(self.qa_file)]
        if isinstance(self.qa_data[0], list):
            # unwrap one-line json question file
            self.qa_data = self.qa_data[0]

        self.question_process = question_process

    def __getitem__(self, index):
        item = self.qa_data[index]

        img_b64 = item['image']
        image = Image.open(io.BytesIO(
            base64.b64decode(img_b64))).convert('RGB')

        raw_question = item['question']
        question_text = self.question_process(raw_question)
        return {
            'image': image,
            'raw_question': raw_question,
            'question': question_text
        }

    def __len__(self):
        return len(self.qa_data)
class MultimodalQADataset_web(torch_data.Dataset):
    def __init__(self, qa_file, question_process):
        '''
        qa_file: jsonl file that each line is a dict like {
            'image': b64img,
            'question': question_text
        }
        '''
        super().__init__()

        self.question = qa_file["question"]
        self.qa_image = qa_file["image"]
        self.question_process = question_process

    def __getitem__(self, index):

        img_b64 = self.qa_image
        image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert('RGB')

        raw_question = self.question
        question_text = self.question_process(raw_question)
        return {
            'image': image,
            'raw_question': raw_question,
            'question': question_text
        }

    def __len__(self):
        length = 1
        return length

class SingleDataSourceDataset(torch_data.Dataset):
    def __init__(self, ds_name, data_dir, tsv_filenames: List[str], intent='sft') -> None:
        super().__init__()

        self.data_dir = data_dir
        self.filenames = tsv_filenames
        self.ds_name = ds_name

        self.sizes = []
        for filename in self.filenames:
            try:
                size = int(filename[:-4].split('-')[-1])
            except:
                raise ValueError(
                    f'TSV Data File {filename} is not valid, last component separated by `-` must be the number of sample in this file')
            self.sizes.append(size)

        self.file_border_index = []
        self.prepare_border_index()

        self.files = self.filenames[:]
        self.intent = intent

        self.fetch_count = 0
        self.clear_at_n_fetch = 1000 + random.randint(100, 1000)

    def prepare_border_index(self):
        self.file_border_index = [0]

        temp_sum = 0
        for size in self.sizes:
            temp_sum += size
            self.file_border_index.append(temp_sum)

    def get_file_idx_and_row_idx(self, index):
        found = False
        file_idx = -1

        for border_idx, border in enumerate(self.file_border_index):
            if index < border:
                file_idx = border_idx - 1
                found = True
                break
        if not found:
            raise ValueError(
                f'Index {index} out of range for {self.size_sum} border markers')

        offset = self.file_border_index[file_idx]
        row_idx = index - offset
        return file_idx, row_idx

    def __len__(self):
        return self.file_border_index[-1]

    def __getitem__(self, index, error_count=0):
        self.fetch_count += 1
        if self.fetch_count >= self.clear_at_n_fetch:
            self.fetch_count = 0

            # only apply to super large datasets in fact
            if len(self.filenames) >= 50:
                self.files = self.filenames[:]
            gc.collect()

        file_idx, row_idx = self.get_file_idx_and_row_idx(index)

        try:
            data_record = self.fetch_sample(file_idx, row_idx)
            return data_record
        except Exception as e:
            print(
                f'Encounter error while reading line-{row_idx} from {self.filenames[file_idx]}')
            print(e, flush=True)
            if error_count >= 3:
                raise e
            else:
                return self.__getitem__(index + 1, error_count + 1)

    def fetch_sample(self, file_idx, row_idx):
        file = self.files[file_idx]
        if isinstance(file, str):
            self.prepare_file(file_idx)
            file = self.files[file_idx]

        assert isinstance(
            file, TSVFile), f'Expecting TSVFile but get {file} as {type(file)}'

        # tsv line as tuple
        sample = file[row_idx]
        ds_name, *values = sample

        # data dict
        sample = register_data_processor[self.ds_name](
            *values, intent=self.intent)

        if row_idx + 1 == len(file):
            del file
            # TODO: might have to update to clean memory usage?
            self.files[file_idx] = self.filenames[file_idx]

        return sample

    def prepare_file(self, idx):
        filename = self.filenames[idx]
        file = TSVFile(op.join(self.data_dir, filename))
        self.files[idx] = file

# TODO: use datapile?


class IterableSingleDataSourceDataset(torch_data.IterableDataset):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplemented


class MultiDataSourceDataset(torch_data.Dataset):
    def __init__(self, data_sources: List[SingleDataSourceDataset], data_source_weights: List[int]):
        super().__init__()

        self.ds_list = data_sources

        self.sum_weight = sum(data_source_weights)
        self.ds_weights = data_source_weights
        for weight in self.ds_weights:
            assert isinstance(weight, int), 'weight must be integer'

        self.offset2ds = {}
        self.offset2wt = {}
        self.offset2pd = {}
        self.prepare_offset2ds()

        ds_loops = []
        for ds, wt in zip(self.ds_list, self.ds_weights):
            ds_loop = len(ds) // wt
            ds_loops.append(ds_loop)
        max_loop = max(ds_loops)
        self.size = max_loop * self.sum_weight

    def prepare_offset2ds(self):
        offset = 0
        for ds, weight in zip(self.ds_list, self.ds_weights):
            pd = offset
            for _ in range(weight):
                self.offset2ds[offset] = ds
                self.offset2wt[offset] = weight
                self.offset2pd[offset] = pd
                offset += 1

    def __getitem__(self, index):
        n_loop = index // self.sum_weight
        offset = index % self.sum_weight

        ds = self.offset2ds[offset]
        ds_inner_idx = n_loop * \
            self.offset2wt[offset] + offset - self.offset2pd[offset]
        ds_inner_idx = ds_inner_idx % len(ds)

        return ds[ds_inner_idx]

    def __len__(self):
        return self.size


class IterableMultiDataSourceDataset(torch_data.IterableDataset):
    def __init__(self, data_sources, data_source_weights):
        super().__init__()

        self.ds_list = data_sources

        sum_weight = sum(data_source_weights)
        self.ds_weights = [x / sum_weight for x in data_source_weights]

        self.ds_consumption = []
        self.ds_sizes = [len(ds) for ds in self.ds_list]

    def __next__(self):
        ds_idx = numpy.random.choice(
            range(len(self.ds_list)), 1, p=self.ds_weights)[0]
        data_source = self.ds_list[ds_idx]

        self.ds_consumption[ds_idx] += 1
        if self.ds_consumption[ds_idx] % self.ds_sizes[ds_idx] == 0:
            self.report_consumption()

        sample = next(data_source)
        return sample

    def __iter__(self) -> Iterator:
        return self

    def __len__(self):
        return sum(self.ds_sizes)

    def report_consumption(self):
        for ds, consumption, size in zip(self.ds_list, self.ds_consumption, self.ds_sizes):
            print(
                f'Data {ds} consumption: {consumption / size:.2f} epoch', flush=True)
