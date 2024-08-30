from abc import abstractmethod
from ..smp import *


class TextBaseDataset:
    MODALITY = 'TEXT'
    DATASET_URL = {}
    DATASET_MD5 = {}

    def __init__(self, dataset='MMBench', **kwargs):
        self.dataset_name = dataset

        data = self.load_data(dataset)

        data['index'] = [str(x) for x in data['index']]

        if np.all([istype(x, int) for x in data['index']]):
            data['index'] = [int(x) for x in data['index']]

        self.data = data
        self.post_build(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return dict(self.data.iloc[idx])

    def prepare_tsv(self, url, file_md5=None):
        data_root = LMUDataRoot()
        os.makedirs(data_root, exist_ok=True)
        update_flag = False
        file_name = url.split('/')[-1]
        data_path = osp.join(data_root, file_name)
        if osp.exists(data_path) and (file_md5 is None or md5(data_path) == file_md5):
            pass
        else:
            warnings.warn('The dataset tsv is not downloaded')
            download_file(url, data_path)
            update_flag = True

        if file_size(data_path, 'GB') > 1:
            local_path = data_path.replace('.tsv', '_local.tsv')
            if not osp.exists(local_path) or os.environ.get('FORCE_LOCAL', None) or update_flag:
                from ..tools import LOCALIZE
                LOCALIZE(data_path, local_path)
            data_path = local_path
        return load(data_path)

    def dump_image(self, line):
        return []

    def display(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]
        assert isinstance(line, pd.Series) or isinstance(line, dict)
        mmqa_display(line)

    # Return a list of dataset names that are supported by this class, can override
    @classmethod
    def supported_datasets(cls):
        return list(cls.DATASET_URL)

    # Given the dataset name, return the dataset as a pandas dataframe, can override
    def load_data(self, dataset):
        url = self.DATASET_URL[dataset]
        file_md5 = self.DATASET_MD5[dataset]
        return self.prepare_tsv(url, file_md5)

    # Post built hook, will be called after the dataset is built, can override
    def post_build(self, dataset):
        pass

    # Given one data record, return the built prompt (a multi-modal message), can override
    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        question = line['question']

        msgs = []
        msgs.append(dict(type='text', value=question))
        return msgs

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass
