import torch
import torch.distributed as dist
from vlmeval.config import supported_VLM
from vlmeval.utils import track_progress_rich
from vlmeval.smp import *

FAIL_MSG = 'Failed to obtain answer via API.'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, nargs='+', required=True)
    parser.add_argument('--nproc', type=int, default=4, required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    return args


# Only API model is accepted
def infer_data_api(work_dir, model_name, dataset, nframe=8, pack=False, samples_dict={}, api_nproc=4):
    rank, world_size = get_rank_and_world_size()
    assert rank == 0 and world_size == 1
    dataset_name = dataset.dataset_name
    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name
    assert getattr(model, 'is_api', False)

    indices = list(samples_dict.keys())
    structs = [dataset.build_prompt(samples_dict[idx], num_frames=nframe,
                                    video_llm=getattr(model, 'VIDEO_LLM', False)) for idx in indices]

    packstr = 'pack' if pack else 'nopack'
    out_file = f'{work_dir}/{model_name}_{dataset_name}_{nframe}frame_{packstr}_supp.pkl'
    res = load(out_file) if osp.exists(out_file) else {}

    structs = [s for i, s in zip(indices, structs) if i not in res]
    indices = [i for i in indices if i not in res]

    gen_func = model.generate
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]

    if len(structs):
        track_progress_rich(gen_func, structs, nproc=api_nproc, chunksize=api_nproc, save=out_file, keys=indices)

    res = load(out_file)
    return res


def infer_data(model_name, work_dir, dataset, out_file, nframe=8, pack=False, verbose=False, api_nproc=4):
    res = load(out_file) if osp.exists(out_file) else {}
    rank, world_size = get_rank_and_world_size()
    dataset_name = dataset.dataset_name

    sample_indices = list(dataset.videos) if pack else list(dataset.data['index'])
    samples = list(dataset.videos) if pack else list(range(len(dataset.data)))
    sample_map = {i: s for i, s in zip(sample_indices, samples)}

    sample_indices_sub = sample_indices[rank::world_size]
    if np.all([idx in res for idx in sample_indices_sub]):
        return model_name
    sample_indices_subrem = [x for x in sample_indices_sub if x not in res]

    model = supported_VLM[model_name]() if isinstance(model_name, str) else model_name

    is_api = getattr(model, 'is_api', False)
    if is_api:
        assert world_size == 1
        supp = infer_data_api(
            work_dir=work_dir,
            model_name=model_name,
            dataset=dataset,
            nframe=nframe,
            pack=pack,
            samples_dict={k: sample_map[k] for k in sample_indices_subrem},
            api_nproc=api_nproc)
        for k in sample_indices_subrem:
            assert k in supp
        res.update(supp)
        dump(res, out_file)
        return model_name

    for i, idx in tqdm(enumerate(sample_indices_subrem)):
        if idx in res:
            continue
        # adapt to model frame sample number first
        nframe = getattr(model, 'nframe', 0) if getattr(model, 'nframe', 0) > 0 else nframe
        # when using video-llm, build prompt returns video+question; otherwise, several frames+question
        struct = dataset.build_prompt(sample_map[idx], num_frames=nframe, video_llm=getattr(model, 'VIDEO_LLM', False))
        response = model.generate(message=struct, dataset=dataset_name)
        torch.cuda.empty_cache()

        if verbose:
            print(response, flush=True)

        res[idx] = response
        if (i + 1) % 20 == 0:
            dump(res, out_file)

    res = {k: res[k] for k in sample_indices_sub}
    dump(res, out_file)
    return model


# A wrapper for infer_data, do the pre & post processing
def infer_data_job_video(
        model,
        work_dir,
        model_name,
        dataset,
        nframe=8,
        pack=False,
        verbose=False,
        subtitle=False,
        api_nproc=4):

    dataset_name = dataset.dataset_name
    packstr = 'pack' if pack else 'nopack'
    rank, world_size = get_rank_and_world_size()
    result_file = osp.join(work_dir, f'{model_name}_{dataset_name}_{nframe}frame_{packstr}.xlsx')
    if dataset_name == 'Video-MME':
        subtitle_str = 'subs' if subtitle else 'nosubs'
        result_file = result_file.replace('.xlsx', f'_{subtitle_str}.xlsx')

    # Dump Predictions to Prev File if result file exists
    if osp.exists(result_file):
        return model_name

    tmpl = osp.join(work_dir, '{}' + f'{world_size}_{dataset_name}_{nframe}frame_{packstr}.pkl')
    if dataset_name == 'Video-MME':
        subtitle_str = 'subs' if subtitle else 'nosubs'
        tmpl = tmpl.replace('.pkl', f'_{subtitle_str}.pkl')
    out_file = tmpl.format(rank)

    model = infer_data(
        model,
        work_dir=work_dir,
        dataset=dataset,
        nframe=nframe,
        pack=pack,
        out_file=out_file,
        verbose=verbose,
        api_nproc=api_nproc)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        data_all = {}
        for i in range(world_size):
            data_all.update(load(tmpl.format(i)))

        meta = dataset.data
        if dataset_name == 'MMBench-Video' and pack:
            meta, vstats = dataset.load_pack_answers(data_all)
            print(f'Statitics of Pack Video Inference: {vstats}')
        else:
            for x in meta['index']:
                assert x in data_all
            meta['prediction'] = [str(data_all[x]) for x in meta['index']]
            if 'image' in meta:
                meta.pop('image')

        dump(meta, result_file)
        for i in range(world_size):
            os.remove(tmpl.format(i))
    return model
