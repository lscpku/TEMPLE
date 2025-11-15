import os
import json
import random
from tqdm import tqdm
import sys
import re


def postprocess_response(response, model_name):
    response = response.split("</think>")[-1].strip()
    if model_name.startswith("mimovl"):
        start_idxs = [response.lower().rfind(s) for s in ["the video opens", "the video starts", "the video begins"]]
        start_idx = max(start_idxs)
        if start_idx != -1: 
            response = response[start_idx:]
        else:
            response = "N/A"

        # remove all parentheses containing the word "aption" but not containing ( and )
        response = re.sub(r'\([^()]*aption[^()]*\)', '', response)
        response = response.strip()
    return response


def collect_dpo_data(dpo_dir, all_splits, model_name):
    files = os.listdir(dpo_dir)
    data = {}
    for file in files:
        if file.endswith('filtered'):
            continue
        split = file.split('.')[0]
        if split not in all_splits:
            continue
        print(split)
        with open(os.path.join(dpo_dir, file)) as f:
            for line in tqdm(f):
                datum = json.loads(line)
                video_id = datum['video_id']
                if video_id not in data:
                    data[video_id] = {}
                task = "caption" if split.startswith('caption') else datum['question']
                if task not in data[video_id]:
                    data[video_id][task] = {}
                video_path = datum['video_path']
                if os.path.exists(video_path):
                    response = postprocess_response(datum['response'], model_name)
                    if response == "N/A":
                        continue
                    data[video_id][task][split] = {'video_path': video_path, 'response': response}
    return data


def build_items(dpo_data, chosen_split, rejected_split):
    data = []
    print(chosen_split, rejected_split)
    for video_id in tqdm(dpo_data):
        for task, responses in dpo_data[video_id].items():
            if not chosen_split in responses or not rejected_split in responses:
                continue
            prompt = get_caption_instruction()
            chosen = dpo_data[video_id][task][chosen_split]['response']
            rejected = dpo_data[video_id][task][rejected_split]['response']
            if ('drop' in rejected_split and len(chosen) <= len(rejected)) or \
                chosen in rejected or \
                (('shuffle' in rejected_split or 'reverse' in rejected_split) and (len(chosen) > 1.5 * len(rejected) or len(rejected) > 1.5 * len(chosen))):
                continue
            item = {
                "messages": [
                    {
                        "role": "user",
                        "content": f"<video>{prompt}"
                    },
                    {
                    "role": "assistant",
                    "content": chosen,
                    }
                ],
                "rejected_response": rejected,
                "videos": [dpo_data[video_id][task][chosen_split]['video_path']]
            }
            data.append(item)
    print(len(data))
    return data


cap_insts = [
    "Describe the video in detail.",
    "Provide a detailed description of the events of this video.",
    "Break down the video's content into a chronological sequence of events.",
    "Outline the main actions or developments shown in the video.",
    "Describe the timeline of events as they unfold in the footage.",
    "Explain the progression of events from start to finish in the video."
]


def get_caption_instruction():
    return random.choice(cap_insts)


def main(name, ratios):
    splits = []
    for ratio in ratios:
        splits += [
            ("caption_caption_none_1_False_False", f"caption_caption_drop_{ratio}_False_False"),
            ("caption_caption_none_1_False_False", f"caption_caption_reverse_{ratio}_False_False"),
            ("caption_caption_none_1_False_False", f"caption_caption_shuffle_{ratio}_False_False"),
        ]
    all_splits = set()
    for split_1, split_2 in splits:
        all_splits.add(split_1)
        all_splits.add(split_2)
    output_dir = "./swift_data"
    os.makedirs(output_dir, exist_ok=True)

    data = []
    for _video_split in ["1_2", "2_3"]:
        video_split = f"lv178k_{_video_split}m_ytb"
        print(video_split)
        dpo_dir = f"./data/{video_split}/dpo-{name}/iter_0"
        
        dpo_data = collect_dpo_data(dpo_dir, all_splits, name)
        print(len(dpo_data))

        for chosen_split, rejected_split in splits:
            data += build_items(dpo_data, chosen_split, rejected_split)
    
    print("Total:", len(data))
    with open(os.path.join(output_dir, f"{name}_i0_r{'-'.join(ratios)}.json"), 'w') as f:
        json.dump(data, f, indent=4)



if __name__ == "__main__":
    name = sys.argv[1]
    ratios = sys.argv[2:]
    main(name, ratios)
