import json
import time
import os
import sys
import torch
import torch.multiprocessing as mp

DATA_ROOT = os.path.join(os.path.dirname(__file__), 'data')

def load_jsonl(file):
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def get_generator(model_name):
    if model_name.startswith("qwen25vl"):
        from inference.generator_qwen25vl import DPODataGeneratorQwen25VL
        return DPODataGeneratorQwen25VL
    elif model_name.startswith("qwen2vl"):
        from inference.generator_qwen2vl import DPODataGeneratorQwen2VL
        return DPODataGeneratorQwen2VL
    elif model_name.startswith("mimovl"):
        from inference.generator_mimovl import DPODataGeneratorMiMoVL
        return DPODataGeneratorMiMoVL
    else:
        raise ValueError(f"Invalid model name: {model_name}")

def process(process_fn, dataset_name, input_type, output_type, rank=None, world_size=None, barrier=None, **process_kwargs):
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
    process_fn(data, output_file, rank, **process_kwargs)
    print(f"Rank {rank}/{world_size}: Finished processing {dataset_name} {input_type} to {output_type}")

    if barrier is not None:
        barrier.wait()

    return data



def generate_clip_captions(video_list, output_path=None, rank=None, **process_kwargs):
    from tools.loader import load_video_from_indices
    device = f"cuda:{rank % torch.cuda.device_count()}" if rank is not None else "auto"

    model_name = process_kwargs.get("model_name")
    generator = get_generator(model_name)(device=device)

    for item in video_list:
        video_path = item["video_path"]
        frame_indices = sum(item["frame_indices"], [])
        start_time = time.time()
        print(f"Loading video from {video_path} with frame indices {frame_indices}")
        frames, _ = load_video_from_indices(video_path, frame_indices)
        clip_caps = generator.generate_diff_caption(frames, postprocess=False)
        item["raw_clip_captions"] = clip_caps
        end_time = time.time()
        print(f"Time taken: {end_time - start_time} seconds")

        if output_path:
            with open(output_path, "a") as f:
                f.write(json.dumps(item) + "\n")


def postprocess_clip_captions(video_list, output_path=None, rank=None, **process_kwargs):
    device = f"cuda:{rank % torch.cuda.device_count()}" if rank is not None else "auto"

    model_name = process_kwargs.get("model_name")
    generator = get_generator(model_name)(device=device)

    for item in video_list:
        item["clip_captions"] = [generator.postprocess_diff_cap_response(cap) for cap in item["raw_clip_captions"]]
        if output_path:
            with open(output_path, "a") as f:
                f.write(json.dumps(item) + "\n")



def main(dataset_name, rank, world_size, barrier):
    torch.cuda.set_device(rank % torch.cuda.device_count())
    process_kwargs = {
        "model_name": "qwen2vl7b" # "mimovl", "qwen25vl7b""
    }
    process(generate_clip_captions, dataset_name, 'videos', f'raw_clip_captions_{process_kwargs["model_name"]}', rank, world_size, barrier, **process_kwargs)
    process(postprocess_clip_captions, dataset_name, f'raw_clip_captions_{process_kwargs["model_name"]}', f'clip_captions_{process_kwargs["model_name"]}', rank, world_size, barrier, **process_kwargs)



if __name__ == "__main__":
    world_size = int(sys.argv[1])
    barrier = mp.Barrier(world_size)

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=main, args=('lv178k_1_2m_ytb', rank, world_size, barrier))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
