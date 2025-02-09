from vlmeval.dataset import *
from functools import partial

mmbench_video_dataset = {
    'MMBench_Video_8frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=False),
    'MMBench_Video_8frame_pack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=8, pack=True),
    'MMBench_Video_16frame_nopack': partial(MMBenchVideo, dataset='MMBench-Video', nframe=16, pack=False),
    'MMBench_Video_1fps_nopack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=False),
    'MMBench_Video_1fps_pack': partial(MMBenchVideo, dataset='MMBench-Video', fps=1.0, pack=True)
}

mvbench_dataset = {
    'MVBench_8frame': partial(MVBench, dataset='MVBench', nframe=8),
    # MVBench not support fps, but MVBench_MP4 does
    'MVBench_MP4_8frame': partial(MVBench_MP4, dataset='MVBench_MP4', nframe=8),
    'MVBench_MP4_1fps': partial(MVBench_MP4, dataset='MVBench_MP4', fps=1.0),
}

videomme_dataset = {
    'Video-MME_8frame': partial(VideoMME, dataset='Video-MME', nframe=8),
    'Video-MME_8frame_subs': partial(VideoMME, dataset='Video-MME', nframe=8, use_subtitle=True),
    'Video-MME_1fps': partial(VideoMME, dataset='Video-MME', fps=1.0),
    'Video-MME_0.5fps': partial(VideoMME, dataset='Video-MME', fps=0.5),
    'Video-MME_0.5fps_subs': partial(VideoMME, dataset='Video-MME', fps=0.5, use_subtitle=True),
}

longvideobench_dataset = {
    'LongVideoBench_8frame': partial(LongVideoBench, dataset='LongVideoBench', nframe=8),
    'LongVideoBench_8frame_subs': partial(LongVideoBench, dataset='LongVideoBench', nframe=8, use_subtitle=True),
    'LongVideoBench_1fps': partial(LongVideoBench, dataset='LongVideoBench', fps=1.0),
    'LongVideoBench_0.5fps': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5),
    'LongVideoBench_0.5fps_subs': partial(LongVideoBench, dataset='LongVideoBench', fps=0.5, use_subtitle=True)
}

mlvu_dataset = {
    'MLVU_8frame': partial(MLVU, dataset='MLVU', nframe=8),
    'MLVU_1fps': partial(MLVU, dataset='MLVU', fps=1.0)
}

tempcompass_dataset = {
    'TempCompass_8frame': partial(TempCompass, dataset='TempCompass', nframe=8),
    'TempCompass_1fps': partial(TempCompass, dataset='TempCompass', fps=1.0),
    'TempCompass_0.5fps': partial(TempCompass, dataset='TempCompass', fps=0.5)
}

# In order to reproduce the experimental results in CGbench paper,
# use_subtitle, use_subtitle_time and use_frame_time need to be set to True.
# When measuring clue-related results, if the number of frames used is greater
# than 32, the frame capture limit will be set to 32.
cgbench_dataset = {
    'CGBench_MCQ_Grounding_Mini_8frame_subs_subt': partial(
        CGBench_MCQ_Grounding_Mini,
        dataset='CG-Bench_MCQ_Grounding_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True
    ),
    'CGBench_OpenEnded_Mini_8frame_subs_subt_ft': partial(
        CGBench_OpenEnded_Mini,
        dataset='CG-Bench_OpenEnded_Mini',
        nframe=8,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_MCQ_Grounding_32frame_subs': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=32,
        use_subtitle=True
    ),
    'CGBench_OpenEnded_8frame': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=8
    ),
    'CGBench_MCQ_Grounding_16frame_subs_subt_ft': partial(
        CGBench_MCQ_Grounding,
        dataset='CG-Bench_MCQ_Grounding',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    ),
    'CGBench_OpenEnded_16frame_subs_subt_ft': partial(
        CGBench_OpenEnded,
        dataset='CG-Bench_OpenEnded',
        nframe=16,
        use_subtitle=True,
        use_subtitle_time=True,
        use_frame_time=True
    )
}

supported_video_datasets = {}

dataset_groups = [
    mmbench_video_dataset, mvbench_dataset, videomme_dataset, longvideobench_dataset,
    mlvu_dataset, tempcompass_dataset, cgbench_dataset
]

for grp in dataset_groups:
    supported_video_datasets.update(grp)
