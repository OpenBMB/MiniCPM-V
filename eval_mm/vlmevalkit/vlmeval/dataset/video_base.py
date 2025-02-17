from abc import abstractmethod
from ..smp import *


class VideoBaseDataset:

    MODALITY = 'VIDEO'

    def __init__(self,
                 dataset='MMBench-Video',
                 pack=False,
                 nframe=0,
                 fps=-1):
        try:
            import decord
        except Exception as e:
            logging.critical(f'{type(e)}: {e}')
            logging.critical('Please install decord via `pip install decord`.')

        self.dataset_name = dataset
        ret = self.prepare_dataset(dataset)
        assert ret is not None
        lmu_root = LMUDataRoot()
        self.frame_root = osp.join(lmu_root, 'images', dataset)
        os.makedirs(self.frame_root, exist_ok=True)
        self.frame_tmpl = 'frame-{}-of-{}.jpg'
        self.frame_tmpl_fps = 'frame-{}-of-{}-{}fps.jpg'

        self.data_root = ret['root']
        self.data_file = ret['data_file']
        self.data = load(self.data_file)

        assert 'question' in self.data and 'video' in self.data
        videos = list(set(self.data['video']))
        videos.sort()
        self.videos = videos
        self.pack = pack
        self.nframe = nframe
        self.fps = fps
        if self.fps > 0 and self.nframe > 0:
            raise ValueError('fps and nframe should not be set at the same time')
        if self.fps <= 0 and self.nframe <= 0:
            raise ValueError('fps and nframe should be set at least one valid value')

    def __len__(self):
        return len(self.videos) if self.pack else len(self.data)

    def __getitem__(self, idx):
        if self.pack:
            assert idx < len(self.videos)
            sub_data = self.data[self.data['video'] == self.videos[idx]]
            return sub_data
        else:
            assert idx < len(self.data)
            return dict(self.data.iloc[idx])

    def frame_paths(self, video):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root, self.frame_tmpl.format(i, self.nframe)) for i in range(1, self.nframe + 1)]

    def frame_paths_fps(self, video, num_frames):
        frame_root = osp.join(self.frame_root, video)
        os.makedirs(frame_root, exist_ok=True)
        return [osp.join(frame_root,
                         self.frame_tmpl_fps.format(i, num_frames, self.fps)) for i in range(1, num_frames + 1)]

    def save_video_frames(self, video):
        if self.fps > 0:
            vid_path = osp.join(self.data_root, video + '.mp4')
            vid = decord.VideoReader(vid_path)

            # 计算视频的总帧数和总时长
            total_frames = len(vid)
            video_fps = vid.get_avg_fps()
            total_duration = total_frames / video_fps

            # 计算需要提取的总帧数
            required_frames = int(total_duration * self.fps)

            # 计算提取帧的间隔
            step_size = video_fps / self.fps

            # 计算提取帧的索引
            indices = [int(i * step_size) for i in range(required_frames)]

            # 提取帧并保存
            frame_paths = self.frame_paths_fps(video, len(indices))
            flag = np.all([osp.exists(p) for p in frame_paths])
            if flag:
                return frame_paths

            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)
            return frame_paths

        else:
            frame_paths = self.frame_paths(video)
            flag = np.all([osp.exists(p) for p in frame_paths])
            if flag:
                return frame_paths
            vid_path = osp.join(self.data_root, video + '.mp4')
            vid = decord.VideoReader(vid_path)
            step_size = len(vid) / (self.nframe + 1)
            indices = [int(i * step_size) for i in range(1, self.nframe + 1)]
            images = [vid[i].asnumpy() for i in indices]
            images = [Image.fromarray(arr) for arr in images]
            for im, pth in zip(images, frame_paths):
                if not osp.exists(pth):
                    im.save(pth)
            return frame_paths

    # Return a list of dataset names that are supported by this class, can override
    @classmethod
    def supported_datasets(cls):
        return ['MMBench-Video', 'Video-MME', 'MVBench', 'MVBench_MP4', 'LongVideoBench']

    # Given the prediction file, return the evaluation results in the format of a dictionary or pandas dataframe
    @abstractmethod
    def evaluate(self, eval_file, **judge_kwargs):
        pass

    @abstractmethod
    def build_prompt(self, idx):
        pass

    @abstractmethod
    def prepare_dataset(self, dataset):
        # The prepare_dataset function should return a dictionary containing:
        # `root` (directory that containing video files)
        # `data_file` (the TSV dataset file)
        pass
