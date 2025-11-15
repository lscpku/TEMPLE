import os
import cv2
from scenedetect import FrameTimecode
import torch
from .transnetv2_pytorch import TransNetV2
from attrdict import AttrDict

transnet_config = AttrDict(
    model_path = os.path.join(os.path.dirname(__file__), "transnetv2-pytorch-weights.pth"),
    device = "cuda",
    segment = 2000,
    overlap = 400,
    final = 400,
    height = 27,
    width = 48,
    threshold = 0.5,
    single_frame = False
)


def load_transnet_model(model_path, device="cuda"):
    model = TransNetV2()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def predict(model, video_path, config):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ori_frames = []
    frames = []
    preds = []
    start = 0
    while start < length:
        start = start - config.overlap if start != 0 else start
        end = start + config.segment
        if length - end < config.final:
            end = length
        while len(frames) < end:
            ret, frame = cap.read()
            ori_frames.append(frame)
            # if not ret:
            #     break
            frames.append(torch.from_numpy(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (config.width, config.height))).to(config.device))
        # print(end)
        
        segment_frames = torch.stack(frames[start:end])
        with torch.no_grad():
            # shape: batch dim x video frames x frame height x frame width x RGB (not BGR) channels, uint8
            input_video = segment_frames.unsqueeze(0)
            # print(input_video.shape, input_video.dtype, type(input_video))
            single_frame_pred, all_frame_pred = model(input_video)
            if config.single_frame:
                frame_pred = torch.sigmoid(single_frame_pred).cpu().numpy()[0,:,0].tolist()
            else:
                frame_pred = torch.sigmoid(all_frame_pred['many_hot']).cpu().numpy()[0,:,0].tolist()
        
        start_p = 0 if start == 0 else config.overlap // 2
        end_p = end - start if end == length else end - start - config.overlap // 2
        preds += frame_pred[start_p:end_p]

        start = end
    
    cap.release()

    results = [(pred > config.threshold) for pred in preds]

    return results, ori_frames, fps
    
def get_cuts(results, fps):
    cuts = []
    last = 0
    for i, result in enumerate(results):
        if result:
            start = last if len(cuts) == 0 else last + 1
            end = i
            if end - start + 1 >= 8:
                cuts.append([len(cuts), FrameTimecode(start, fps=fps), FrameTimecode(end, fps=fps)])
            last = i
    if len(results) > 0:
        start = 0 if len(cuts) == 0 else last + 1
        end = len(results) - 1
        if end - start + 1 >= 8:
            cuts.append([len(cuts), FrameTimecode(start, fps=fps), FrameTimecode(end, fps=fps)])
    return cuts