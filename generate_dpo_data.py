import json
import time
import os
import sys
import torch
import torch.multiprocessing as mp
from inference.vidpo_config import VidPOConfig

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


def process(process_fn, dataset_name, input_type, dir_suffix, vidpo_config: VidPOConfig, num_iter=0, rank=None, world_size=None, barrier=None, **process_kwargs):
    output_type = str(vidpo_config)

    print(f"Rank {rank}/{world_size}: Start processing {dataset_name} {input_type} to {output_type} {num_iter} iter")
    meta_dir = os.path.join(DATA_ROOT, dataset_name, 'anno')
    dpo_dir = os.path.join(DATA_ROOT, dataset_name, f'dpo-{dir_suffix}', f'iter_{num_iter}')
    os.makedirs(dpo_dir, exist_ok=True)

    input_part_files = [file for file in os.listdir(meta_dir) if file.startswith(f"{input_type}.jsonl")]
    input_data = []
    for file in input_part_files:
        input_data += load_jsonl(os.path.join(meta_dir, file))
    all_ids = set([item['video_id'] for item in input_data])
    id2item = {item['video_id']: item for item in input_data}

    output_part_files = [file for file in os.listdir(dpo_dir) if file.startswith(f"{output_type}.jsonl")]
    output_data = []
    for file in output_part_files:
        output_data += load_jsonl(os.path.join(dpo_dir, file))
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
        with open(os.path.join(dpo_dir, f"{output_type}.jsonl.filtered"), 'w') as f:
            for item in filtered_data:
                f.write(json.dumps(item) + "\n")
    
    data = unfiltered_data
    if rank is not None and world_size is not None:
        split_size = (len(data) + world_size - 1) // world_size
        start_id = rank * split_size
        end_id = min((rank + 1) * split_size, len(data))
        output_file = os.path.join(dpo_dir, f"{output_type}.jsonl.{rank}-{world_size}")
    else:
        start_id = 0
        end_id = len(data)
        output_file = os.path.join(dpo_dir, f"{output_type}.jsonl")
    
    print(f"Rank {rank}/{world_size}: from {start_id} to {end_id - 1} ({end_id - start_id} items)")
    data = data[start_id:end_id]
    process_fn(data, vidpo_config, output_file, rank, **process_kwargs)
    print(f"Rank {rank}/{world_size}: Finished processing {dataset_name} {input_type} to {output_type}")

    if barrier is not None:
        barrier.wait()

    return data


def generate_dpo_response_from_vidpo_config(video_list, vidpo_config: VidPOConfig, output_path=None, rank=None, **process_kwargs):
    from tools.loader import load_video_from_indices

    if rank is not None:
        device = f"cuda:{rank % torch.cuda.device_count()}"
    else:
        device = "auto"

    model_name = process_kwargs.get("model_name")
    generator = get_generator(model_name)(device=device)

    gen_kwargs = {
        "do_sample": True,
        "temperature": 0.7,
        "repetition_penalty": 1.05,
        "max_new_tokens": 8192,
    }

    for item in video_list:
        video_path = item["video_path"]
        frame_indices = sum(item["frame_indices"], [])
        frames = None
        drop_indices = None

        new_item = {
            "video_id": item["video_id"],
            "video_path": item["video_path"],
            "frame_indices": item["frame_indices"],
        }

        start_time = time.time()
        print(f"Generating response for {item['video_id']} with config {vidpo_config}")

        if vidpo_config.video_input_type != "caption" and frames is None:
            print(f"Loading video from {video_path} with frame indices {frame_indices}")
            frames, _ = load_video_from_indices(video_path, frame_indices)
            frames = [frames[i:i+2] for i in range(0, len(frames), 2)]

        response, info = generator.generate_rejected_response(item, vidpo_config, frames, drop_indices, **gen_kwargs)
        new_item["response"] = response
        new_item["info"] = info

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.2f} seconds")

        if output_path is not None:
            with open(output_path, "a") as f:
                f.write(json.dumps(new_item) + "\n")


def get_vidpo_configs():
    vidpo_configs = []

    task_type = "caption"
    video_input_type = "caption"
    video_noise_type = "none"
    vidpo_config = VidPOConfig(task_type=task_type, video_input_type=video_input_type, video_noise_type=video_noise_type)
    vidpo_configs.append(vidpo_config)

    task_type = "caption"
    video_input_type = "caption"
    for video_noise_type in ["drop", "reverse", "shuffle"]:
        for magnitude in [2, 4, 8, 16]:
            vidpo_config = VidPOConfig(task_type=task_type, video_input_type=video_input_type, video_noise_type=video_noise_type, magnitude=magnitude)
            vidpo_configs.append(vidpo_config)

    return vidpo_configs
    
def main(dataset_name, num_iter, rank, world_size, barrier):
    torch.cuda.set_device(rank % torch.cuda.device_count())
    process_kwargs = {
        "model_name": "qwen2vl7b"  # "qwen25vl7b", "mimovl"
    }
    vidpo_configs = get_vidpo_configs()
    for vidpo_config in vidpo_configs:
        process(generate_dpo_response_from_vidpo_config, dataset_name, f'clip_captions_{process_kwargs["model_name"]}', process_kwargs["model_name"], vidpo_config, num_iter, rank, world_size, barrier, **process_kwargs)


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    world_size = int(sys.argv[2])
    barrier = mp.Barrier(world_size)
    
    num_iter = 0

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=main, args=(dataset_name, num_iter, rank, world_size, barrier))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
