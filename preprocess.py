import os
import sys
import json
import cv2
import numpy as np
import torch
import time
import torch.multiprocessing as mp

DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')
RAW_VIDEO_DIR = {
    'lv178k_1_2m_ytb': "/mnt/datasets/lmms-lab/LLaVA-Video-178K/1_2_m_youtube_v0_1/liwei_youtube_videos/videos",
    'lv178k_2_3m_ytb': "/mnt/datasets/lmms-lab/LLaVA-Video-178K/2_3_m_youtube_v0_1/liwei_youtube_videos/videos",
}

def init_video_list(dataset_name):
    assert dataset_name in RAW_VIDEO_DIR, f"Unsupported dataset: {dataset_name}"

    data_dir = os.path.join(DATA_ROOT, dataset_name)
    anno_dir = os.path.join(data_dir, 'anno')
    video_dir = os.path.join(data_dir, 'video')
    os.makedirs(anno_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    video_paths = []

    raw_video_dir = RAW_VIDEO_DIR[dataset_name]
    video_dirs = os.listdir(raw_video_dir)
    videos = []
    for video_dir in video_dirs:
        video_files = [file for file in os.listdir(os.path.join(raw_video_dir, video_dir)) if file.endswith(".mp4")]
        videos += [os.path.join(video_dir, video) for video in video_files]
    video_paths = [os.path.join(raw_video_dir, video) for video in videos]

    video_list = [{
        "video_id": f"{dataset_name}_{i}",
        "video_path": video_path, 
        "video_name": os.path.basename(video_path),
        "filtered": False,
    } for i, video_path in enumerate(video_paths)]

    with open(os.path.join(anno_dir, 'init.jsonl'), 'w') as f:
        for item in video_list:
            f.write(json.dumps(item) + "\n")


def load_jsonl(file):
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    return data


def process(process_fn, dataset_name, input_type, output_type, rank=None, world_size=None, barrier=None):
    print(f"Rank {rank}/{world_size}: Start processing {dataset_name} {input_type} to {output_type}")
    anno_dir = os.path.join(DATA_ROOT, dataset_name, 'anno')

    input_part_files = [file for file in os.listdir(anno_dir) if file.startswith(f"{input_type}.jsonl")]
    input_data = []
    for file in input_part_files:
        input_data += load_jsonl(os.path.join(anno_dir, file))
    all_ids = set([item['video_id'] for item in input_data])
    id2item = {item['video_id']: item for item in input_data}

    output_part_files = [file for file in os.listdir(anno_dir) if file.startswith(f"{output_type}.jsonl")]
    output_data = []
    for file in output_part_files:
        output_data += load_jsonl(os.path.join(anno_dir, file))
    output_ids = set([item['video_id'] for item in output_data])

    if barrier is not None:
        barrier.wait()

    unfinished_ids = all_ids - output_ids
    unfinished_ids_list = sorted(list(unfinished_ids))
    data = [id2item[video_id] for video_id in unfinished_ids_list]

    filtered_data = [item for item in data if item['filtered']]
    unfiltered_data = [item for item in data if not item['filtered']]
    if rank is None or rank == 0:
        print(f"Processing {dataset_name} {input_type} to {output_type}")
        print(f"Total items:", len(input_data))
        print(f"Unfinished items:", len(unfiltered_data))
        with open(os.path.join(anno_dir, f"{output_type}.jsonl.filtered"), 'w') as f:
            for item in filtered_data:
                f.write(json.dumps(item) + "\n")
    
    data = unfiltered_data
    if rank is not None and world_size is not None:
        split_size = (len(data) + world_size - 1) // world_size
        start_id = rank * split_size
        end_id = min((rank + 1) * split_size, len(data))
        output_file = os.path.join(anno_dir, f"{output_type}.jsonl.{rank}-{world_size}")
    else:
        start_id = 0
        end_id = len(data)
        output_file = os.path.join(anno_dir, f"{output_type}.jsonl")
    
    print(f"Rank {rank}/{world_size}: from {start_id} to {end_id - 1} ({end_id - start_id} items)")
    data = data[start_id:end_id]
    process_fn(data, output_file)
    print(f"Rank {rank}/{world_size}: Finished processing {dataset_name} {input_type} to {output_type}")

    if barrier is not None:
        barrier.wait()

    return data


def scene_detect(video_list, output_path=None):
    from tools.transnet import transnet_config, load_transnet_model, get_cuts, predict
    transnet = load_transnet_model(transnet_config.model_path, transnet_config.device)

    for item in video_list:
        video_path = item['video_path']
        try:
            results, frames, fps = predict(transnet, video_path, transnet_config)
        except Exception as e:
            if isinstance(e, KeyboardInterrupt):
                raise e
            item['filtered'] = True
            continue
        cuts = get_cuts(results, fps)
        item['raw_cuts'] = [[i, j.get_frames(), k.get_frames()] for i, j, k in cuts]
        item['fps'] = fps
        item['filtered_cuts'] = []
        for i, start_frame, end_frame in item['raw_cuts']:
            if end_frame - start_frame < 0.2 * fps:
                continue
            clip_frames = np.stack([
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                for frame in frames[start_frame:end_frame]
            ])
            if clip_frames.max() - clip_frames.min() < 10:
                continue
            item['filtered_cuts'].append([i, start_frame, end_frame])

        if output_path:
            with open(output_path, 'a') as f:
                f.write(json.dumps(item) + "\n")

    return video_list


def filter_by_scenes(video_list, output_path=None):
    for item in video_list:
        fps = item['fps']
        item['filtered'] = False
        if len(item['filtered_cuts']) > 32 or len(item['filtered_cuts']) < 4:
            item['filtered'] = True
        else:
            for i, start, end in item['filtered_cuts']:
                if end - start > 16 * fps:
                    item['filtered'] = True
                    break
        
        if output_path:
            with open(output_path, 'a') as f:
                f.write(json.dumps(item) + "\n")
    
    return video_list


def calculate_clip_similarity(video_list, output_path=None):
    from tools.similarity import load_embedding_model
    from tools.loader import load_video, load_video_from_indices, get_frame_indices
    model = load_embedding_model("siglip-so400m-patch14-384", device="cuda")
    batch_size = 32

    for item in video_list:
        video_path = item['video_path']
        indices = []
        for i, start, end in item['filtered_cuts']:
            clip_frame_indices, _, _ = get_frame_indices(video_path, n_frms=1, start=start, end=end)
            indices += clip_frame_indices
        frames, _ = load_video_from_indices(video_path, indices, output_type="pil")

        embeds = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            text_features, img_features = model(imgs=batch_frames)
            embeds.append(img_features)
        embeds = torch.cat(embeds, dim=0)
        similarity = torch.matmul(embeds, embeds.T).cpu().tolist()
        item['clip_similarity'] = similarity

        if output_path:
            with open(output_path, 'a') as f:
                f.write(json.dumps(item) + "\n")
    
    return video_list


def clip_grouping(video_list, output_path=None):
    from tools.similarity import merge_items
    threshold = 0.8

    for item in video_list:
        similarity_matrix = np.array(item['clip_similarity'])
        reduced_similarity_matrix, groups = merge_items(similarity_matrix, threshold)
        item['groups'] = [list(group) for group in groups]

        clip2group = {}
        for i, group in enumerate(groups):
            for clip in group:
                clip2group[clip] = i
        
        N = len(item['clip_similarity'])
        merged_groups = []
        group_id = -1
        for i in range(N):
            if clip2group[i] != group_id:
                group_id = clip2group[i]
                merged_groups.append((group_id, [i]))
            else:
                merged_groups[-1][1].append(i)
        
        item['merged_groups'] = merged_groups

        if output_path:
            with open(output_path, 'a') as f:
                f.write(json.dumps(item) + "\n")

    return video_list


def filter_by_groups(video_list, output_path=None):
    for item in video_list:
        if len(item['merged_groups']) < 4:
            item['filtered'] = True

        group_cnts = {}
        for i, c in item['merged_groups']:
            group_cnts[i] = group_cnts.get(i, 0) + 1
            if group_cnts[i] > len(item['merged_groups']) // 3 or group_cnts[i] >= 4:
                item['filtered'] = True
                break
        
        if output_path:
            with open(output_path, 'a') as f:
                f.write(json.dumps(item) + "\n")

    return video_list


def extract_frames(video_list, output_path=None):
    from tools.laplacian import laplacian
    from tools.loader import load_video

    for item in video_list:
        frame_idxs = []
        for i, group in item['merged_groups']:
            start_clip = group[0]
            end_clip = group[-1]
            start = item['filtered_cuts'][start_clip][1]
            end = item['filtered_cuts'][end_clip][2]
            frames, _, _ = load_video(item['video_path'], n_frms=-1, start=start, end=end, output_type="numpy")
            clarity_scores = [laplacian(frame).mean().item() for frame in frames]

            n_frames = 2
            interval = (end - start) / (n_frames + 1)
            
            score_frame_idxs = []
            for i in range(1, n_frames + 1):
                center_frame_idx = int(start + interval * i)
                _start_idx = center_frame_idx - int(interval) // 2
                _end_idx = center_frame_idx + int(interval) // 2
                raw_scores = np.array(clarity_scores[_start_idx - start:_end_idx - start])
                distance = np.arange(-(int(interval) // 2), int(interval) // 2)
                _interval = int(interval) // 2 * 2
                weights = np.cos(np.pi * distance / _interval)
                weighted_scores = (raw_scores * weights).tolist()
                frame_idx = _start_idx + weighted_scores.index(max(weighted_scores))
                score_frame_idxs.append(frame_idx)
            
            frame_idxs.append(score_frame_idxs)
        
        item['frame_indices'] = frame_idxs

        if output_path:
            with open(output_path, 'a') as f:
                f.write(json.dumps(item) + "\n")

    return video_list



def finalize(video_list, output_path):
    with open(output_path, 'w') as f:
        for item in video_list:
            f.write(json.dumps(item) + "\n")



def main(dataset_name, rank, world_size, barrier):
    torch.cuda.set_device(rank % torch.cuda.device_count())

    if rank is None or rank == 0:
        init_video_list(dataset_name)
    barrier.wait()
    
    process(scene_detect, dataset_name, 'init', 'scene', rank, world_size, barrier)
    process(filter_by_scenes, dataset_name, 'scene', 'filter_scene', rank, world_size, barrier)
    process(calculate_clip_similarity, dataset_name, 'filter_scene', 'similarity', rank, world_size, barrier)
    process(clip_grouping, dataset_name, 'similarity', 'group', rank, world_size, barrier)
    process(filter_by_groups, dataset_name, 'group', 'filter_group', rank, world_size, barrier)
    process(extract_frames, dataset_name, 'filter_group', 'extract_frame', rank, world_size, barrier)

    if rank is None or rank == 0:
        process(finalize, dataset_name, 'extract_frame', 'finalize', rank=None, world_size=None)
    


if __name__ == "__main__":
    dataset_name = sys.argv[1]               # lv178k_1_2m_ytb, lv178k_2_3m_ytb
    world_size = int(sys.argv[2])            # 8
    barrier = mp.Barrier(world_size)

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=main, args=(dataset_name, rank, world_size, barrier))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()