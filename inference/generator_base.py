import random
import numpy as np
import copy
from .vidpo_config import VidPOConfig

class DPODataGeneratorBase:
    def __init__(self, device=None):
        self._model = None
        self.device = device

    @property
    def model(self):
        if self._model is None:
            self._model = self.load_model()
        return self._model

    def load_model(self):
        raise NotImplementedError
    
    def get_single_cap_message(self, clip):
        raise NotImplementedError
    
    def get_diff_cap_message(self, clip_1, clip_2):
        raise NotImplementedError
    
    def get_vidpo_message(self, clip_caps, task_type, task_input=None):
        raise NotImplementedError
    
    def postprocess_diff_cap_response(self, resp):
        raise NotImplementedError

    def generate_diff_caption(self, frames, num_frames_per_clip=2, num_context_clips=1, postprocess=True):
        assert len(frames) % num_frames_per_clip == 0

        n_clips = len(frames) // num_frames_per_clip
        clip_caps = []
        for i in range(n_clips):
            print(f"Generating clip {i} of {n_clips}")
            if i == 0:
                clip = frames[:num_frames_per_clip]
                msg = self.get_single_cap_message(clip)
            else:
                clip_1 = frames[max((i - num_context_clips) * num_frames_per_clip, 0) : i * num_frames_per_clip]
                clip_2 = frames[i * num_frames_per_clip : (i + 1) * num_frames_per_clip]
                msg = self.get_diff_cap_message(clip_1, clip_2)
            resp = self.model(**msg)
            if postprocess:
                cap = self.postprocess_diff_cap_response(resp)
            else:
                cap = resp
            clip_caps.append(cap)

        return clip_caps

    def generate_caption(self, clip_caps):
        msg = self.get_vidpo_message(clip_caps, 'caption', 'caption')
        video_cap = self.model(**msg)
        return video_cap

    def generate_rejected_response(
            self, 
            item: dict, 
            vidpo_config: VidPOConfig, 
            clips: list, 
            drop_indices: list | None = None, 
            **gen_kwargs
        ):
        """
        task_type: caption
        video_input_type: caption/clip/video
        video_noise_type: none/drop/reverse/shuffle
        base_clip_frames: 2
        magnitude: 2/4/8/16,
        drop_by_similarity: True/False
        drop_by_group: True/False
        """
        assert vidpo_config.task_type in ['caption']
        assert vidpo_config.video_input_type in ['caption', 'clip', 'video']
        assert vidpo_config.video_noise_type in ['none', 'drop', 'reverse', 'shuffle']

        info = {}

        if vidpo_config.video_input_type == "caption":
            clips = item["clip_captions"]
        
        if vidpo_config.video_noise_type == "drop":
            num_clips = len(clips)
            num_kept_clips = max(1, num_clips // vidpo_config.magnitude)
            num_drop_clips = num_clips - num_kept_clips

            msg = ""

            if drop_indices is None:
                drop_indices = []
            else:
                drop_indices = copy.deepcopy(drop_indices)
            
            if num_drop_clips == len(drop_indices):
                pass
            elif num_drop_clips < len(drop_indices):
                msg = "Too many clips in drop_indices"
                drop_indices = random.sample(drop_indices, k=num_drop_clips)
            else:
                if vidpo_config.drop_by_group:
                    groups = item['groups']
                    clip_idx_to_group_idx = {}
                    for group_idx, group in enumerate(groups):
                        for clip_idx in group:
                            clip_idx_to_group_idx[clip_idx] = group_idx
                
                    remaining_indices = set(range(num_clips)) - set(drop_indices)
                    while len(drop_indices) < num_drop_clips:
                        _drop_index = random.choice(list(remaining_indices))
                        drop_indices.append(_drop_index)
                        remaining_indices.remove(_drop_index)

                        if vidpo_config.drop_by_group:
                            _remaining_clip_idxs_in_group = [clip_idx for clip_idx in groups[clip_idx_to_group_idx[_drop_index]] if clip_idx in remaining_indices]
                            if len(_remaining_clip_idxs_in_group) < num_drop_clips - len(drop_indices):
                                msg = "Too many clips in the same group to drop"
                                drop_indices_in_group = random.sample(_remaining_clip_idxs_in_group, k=num_drop_clips - len(drop_indices))
                            else:
                                drop_indices_in_group = _remaining_clip_idxs_in_group
                            drop_indices.extend(drop_indices_in_group)
                            remaining_indices -= set(drop_indices_in_group)
                
                elif vidpo_config.drop_by_similarity:
                    clip_similarity = item['clip_similarity']

                    while len(drop_indices) < num_drop_clips:
                        if len(drop_indices) == 0:
                            _drop_index = random.choice(range(num_clips))
                        else:
                            probs = np.stack([clip_similarity[drop_index] for drop_index in drop_indices])
                            probs = probs.mean(axis=0)
                            probs[drop_indices] = 0
                            probs /= probs.sum()
                            _drop_index = np.random.choice(range(num_clips), p=probs)
                        drop_indices.append(_drop_index)

                else:
                    remaining_indices = set(range(num_clips)) - set(drop_indices)
                    drop_indices.extend(random.sample(list(remaining_indices), k=num_drop_clips - len(drop_indices)))

            clips = [clip for idx, clip in enumerate(clips) if idx not in drop_indices]
            selected_indices = [idx for idx in range(num_clips) if idx not in drop_indices]

            info["msg"] = msg
            
        elif vidpo_config.video_noise_type == "shuffle":
            num_clips = len(clips)
            group_size = max(1, num_clips // vidpo_config.magnitude)
            groups = [list(range(i, min(i + group_size, num_clips))) for i in range(0, num_clips, group_size)]
            random.shuffle(groups)
            selected_indices = [idx for group in groups for idx in group]
            clips = [clips[idx] for idx in selected_indices]
        elif vidpo_config.video_noise_type == "reverse":
            num_clips = len(clips)
            group_size = max(1, num_clips // vidpo_config.magnitude)
            groups = [list(range(i, min(i + group_size, num_clips))) for i in range(0, num_clips, group_size)]
            groups.reverse()
            selected_indices = [idx for group in groups for idx in group]
            clips = [clips[idx] for idx in selected_indices]
        elif vidpo_config.video_noise_type == "none":
            selected_indices = list(range(len(clips)))

        info["selected_indices"] = selected_indices

        if vidpo_config.task_type == "caption":
            task_input = ""
        else:
            raise ValueError(f"Task type {vidpo_config.task_type} not found")
        
        msg = self.get_vidpo_message(clips, vidpo_config.video_input_type, vidpo_config.task_type, task_input)
        response = self.model(**msg, **gen_kwargs)

        return response, info

