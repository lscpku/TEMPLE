import os
import torch
import re

from .generator_base import DPODataGeneratorBase

MODEL_PATH = "/mnt/ckpts/Qwen2-VL-7B-Instruct/"

diff_cap_prompt = """You will be presented with two continuous video clips from a video. 
First, describe the visual information in the video clip 2.
Then, analyze the possible connections between the two video clips.
Finally, focusing on the analyses above, write a detailed caption for video clip 2. 

Video clip 1: <vid>
Video clip 2: <vid>"""

single_cap_prompt = """Describe this video in detail. 

Video: <vid>"""

task_prompt = {
    "caption": "Your task is to write a detailed caption for the entire video which describes the events in the video in chronological order. Please narrate the video in a fluent and coherent way without any redundant information, while keeping as many details as possible. Provide your answer directly in a single paragraph. ",
}

input_prefix = {
    'caption': "You will be given a list of captions. Each caption corresponds to a video clip sequentially extracted from a video. ",
    'clip': "You will be given a list of continuous video clips. The clips are extracted from a video in chronological order. ",
    'video': "You will be given a video. ",
}

input_template = {
    'caption': '- Caption {i}: {clip}',
    'clip': 'Video clip {i}: <vid>',
    'video': 'Video: <vid>',
}

class DPODataGeneratorQwen2VL(DPODataGeneratorBase):
    def load_model(self):
        from qwen_vl_utils import process_vision_info
        from transformers import Qwen2_VLForConditionalGeneration, AutoProcessor

        model = Qwen2_VLForConditionalGeneration.from_pretrained(
            MODEL_PATH, 
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map=self.device
        )
        processor = AutoProcessor.from_pretrained(MODEL_PATH, max_pixels=28*28*512)

        def generate(messages, **gen_kwargs):
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            kwargs = {
                "max_new_tokens": 2048,
            }
            kwargs.update(gen_kwargs)
            
            # Inference: Generation of the output
            generated_ids = model.generate(
                **inputs, 
                **kwargs
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        
        return generate

    def postprocess_diff_cap_response(self, resp):
        if "\n" in resp:
            resp = resp.split('\n')[-1]
        if resp.startswith("**"):
            resp = resp[2:]
        if (resp.lower().startswith("caption") or resp.lower().startswith("detailed caption")) and ": " in resp:
            resp = resp.split(": ", 1)[1]
        if resp.startswith("**"):
            resp = resp[2:]
        if resp.endswith("\""):
            resp = resp[:-1]
            if resp.startswith("\""):
                resp = resp[1:]
        resp = resp.replace("video clip 2", "this video clip")
        resp = resp.replace("Video clip 2", "This video clip")
        resp = resp.replace("video clip 1", "the previous clip")
        resp = resp.replace("Video clip 1", "The previous clip")
        return resp
    
    def get_message_content(self, prompt, clips):
        if "<vid>" not in prompt:
            return prompt
        
        prompt_segments = prompt.split("<vid>")
        content = []
        for i, clip in enumerate(clips):
            segment = prompt_segments[i]
            if len(segment) > 0:
                content.append({"type": "text", "text": segment})
            content.append({"type": "video", "video": clip})
        if len(prompt_segments[-1]) > 0:
            content.append({"type": "text", "text": prompt_segments[-1]})
        return content
            
    def get_diff_cap_message(self, clip_1, clip_2):
        return {"messages": [
            {
                "role": "user",
                "content": self.get_message_content(diff_cap_prompt, [clip_1, clip_2]),
            }
        ]}

    def get_single_cap_message(self, clip):
        return {"messages": [
            {
                "role": "user",
                "content": self.get_message_content(single_cap_prompt, [clip]),
            }
        ]}
    
    def get_vidpo_message(self, clips, input_type, task_type, task_input=""):
        prompt = input_prefix[input_type]
        prompt += task_prompt[task_type]
        prompt += task_input
        prompt += '\n\n'
        prompt += '\n'.join(input_template[input_type].format(i=i+1, clip=clip) for i, clip in enumerate(clips))

        return {"messages": [
            {
                "role": "user",
                "content": self.get_message_content(prompt, clips),
            }
        ]}