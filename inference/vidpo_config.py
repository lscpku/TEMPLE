from dataclasses import dataclass
from typing import Literal

@dataclass
class VidPOConfig:
    task_type: Literal['caption', 'qa']
    video_input_type: Literal['caption', 'clip', 'video']
    video_noise_type: Literal['none', 'drop', 'reverse', 'shuffle']
    magnitude: int = 1
    drop_by_similarity: bool = False
    drop_by_group: bool = False

    def __str__(self):
        return f"{self.task_type}_{self.video_input_type}_{self.video_noise_type}_{self.magnitude}_{self.drop_by_similarity}_{self.drop_by_group}"