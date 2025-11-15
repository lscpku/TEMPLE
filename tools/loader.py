import torch
from torchvision import transforms as T

from PIL import Image
from typing import List
import numpy as np

from scenedetect import FrameTimecode
import einops

import math

import decord
from decord import VideoReader
decord.bridge.set_bridge('native')


def concat(eles):
    if len(eles) == 0:
        return None
    if isinstance(eles[0], list):
        return sum(eles, [])
    elif isinstance(eles[0], np.ndarray):
        return np.concatenate(eles, axis=0)
    elif isinstance(eles[0], torch.Tensor):
        return torch.cat(eles, dim=0)
    else:
        raise ValueError("Unsupported type")


def get_frame_indices(
        video, 
        n_frms=None, 
        fps=None, 
        points=None, 
        start=None, 
        end=None
    ):
    assert n_frms is not None or fps is not None or points is not None

    if isinstance(video, str):
        vr = VideoReader(video)
    elif isinstance(video, VideoReader):
        vr = video
    else:
        raise ValueError(f"Unsupported video type")
    
    n_frames = len(vr)
    vidfps = vr.get_avg_fps()
    
    start = 0 if start is None else FrameTimecode(start, fps=vidfps).get_frames()
    end = n_frames if end is None else FrameTimecode(end, fps=vidfps).get_frames()
    n_frames = end - start

    if points is not None:
        indices = [int(p * n_frames) + start for p in points]
    else:
        if fps is not None:
            n_frms = math.ceil(n_frames / vidfps * fps)
        if n_frms == -1:
            n_frms = n_frames
        interval = n_frames / (n_frms + 1)
        indices = [int(interval * i) + start for i in range(1, n_frms + 1)]

    metadata = {
        "seconds": n_frames / vidfps,
        "frames": n_frames,
        "fps": vidfps,
        "start": FrameTimecode(start, fps=vidfps).get_timecode(),
        "end": FrameTimecode(end, fps=vidfps).get_timecode(),
    }
    timestamps = [1.0 / vidfps * i for i in indices]

    return indices, metadata, timestamps


def load_video_from_indices(
        video, 
        indices,
        output_type="pil",
        output_dtype="int",
        output_shape="THWC",
        resize=None,
    ):
    assert output_type in ["pil", "numpy", "torch"]
    assert output_dtype in ["int", "float"]
    assert output_shape in ["THWC", "TCHW"]

    if isinstance(video, str):
        vr = VideoReader(video)
    elif isinstance(video, VideoReader):
        vr = video
    else:
        raise ValueError(f"Unsupported video type")
    
    N = 1000
    all_frames = []
    for i in range(0, len(indices), N):
        frames = vr.get_batch(indices[i:i+N])
        frames = frames.asnumpy().astype(np.uint8)    # THWC, uint8
        _T, H, W, C = frames.shape

        frames = torch.from_numpy(frames).permute(0, 3, 1, 2)  # TCHW, uint8
        if resize is not None:
            frames = T.Resize(resize, antialias=False)(frames)  # TCHW, uint8
            H, W = resize
        
        if output_type == "pil":
            frames = frames.permute(0, 2, 3, 1).numpy()  # THWC, uint8
            frames = [Image.fromarray(f) for f in frames]
        else:
            if output_dtype == "float":
                frames = frames.float() / 255.0
            if output_shape == "THWC":
                frames = einops.rearrange(frames, "t c h w -> t h w c")
            if output_type == "numpy":
                frames = frames.numpy()
        all_frames.append(frames)
        
    frames = concat(all_frames)
    metadata = {
        "height": H,
        "width": W,
    }

    return frames, metadata



def load_video(
        video_path, 
        n_frms=None, 
        fps=None, 
        points=None, 
        start=None, 
        end=None, 
        output_type="pil",
        output_dtype="int",
        output_shape="THWC",
        resize=None,
    ) -> List[Image.Image]:
    """
    Sampling strategy:
        uniform, if n_frms is specified (if n_frms==-1, get all frames)
        fixed fps, if fps is specified
        relative points, if points is specified
    """
    vr = VideoReader(video_path)
    indices, metadata, timestamps = get_frame_indices(vr, n_frms, fps, points, start, end)
    
    if len(indices) == 0:
        raise ValueError("No frames to sample")
    
    frames, metadata1 = load_video_from_indices(vr, indices, output_type, output_dtype, output_shape, resize)

    metadata.update(metadata1)
    
    return frames, metadata, timestamps