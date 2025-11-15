
# TEMPLE

This repository contains code for "TEMPLE: Incentivizing Temporal Understanding of Video Large Language Models via Progressive Pre-SFT Alignment" (AAAI 2026), including code for automatically generating preference data from LLaVA-Video-178K and training scripts using the MS-SWIFT framework. 

## Overview

The pipeline processes raw video datasets (e.g., LLaVA-Video-178K subsets), extracts key frames, generates captions using vision-language models, creates preferred and rejected responses for DPO, converts the data to SWIFT format, and trains models with LoRA.

Key components:
- **Preprocessing**: Scene detection, filtering, grouping, and frame extraction.
- **Metadata Generation**: Captioning video clips using models like Qwen2-VL or MiMoVL.
- **DPO Data Generation**: Creating rejected responses with various noise types (drop, reverse, shuffle).
- **Conversion**: Formatting data for SWIFT training.
- **Training**: Shell scripts for running SFT/DPO training with MS-SWIFT.

Supporting directories:
- `tools/`: Utility functions (e.g., loaders, similarity calculators).
- `inference/`: Model-specific generators for captioning and response generation.
- `visualize/`: (for visualizing preference data).

## Requirements

- Python 3.12.8
- torch==2.5.1
- ms-swift==3.7.0.dev0
- transformers==4.53.2
- flash-attn==2.7.4post1
- deepspeed==0.17.2
- opencv-python==4.8.0.74
- decord==0.6.0
- scenedetect==0.6.6
- streamlit==1.47.0

## Datasets

The scripts use subsets of LLaVA-Video-178K:
- `lv178k_1_2m_ytb`: Youtube videos 1-2 minutes.
- `lv178k_2_3m_ytb`: Youtube videos 2-3 minutes.

Raw videos are expected in paths defined in `preprocess.py`.

## Procedure and Usage

Follow these steps to generate DPO data and train models. All scripts support multiprocessing with a `world_size` parameter (number of processes/GPUs).

### 1. Preprocess Videos (`preprocess.py`)

Detects scenes, filters clips, groups similar clips, and extracts key frames.

Usage:
```
python ./preprocess.py <dataset_name> <world_size>
```
- `<dataset_name>`: e.g., `lv178k_1_2m_ytb` or `lv178k_2_3m_ytb`.
- `<world_size>`: Number of processes (e.g., 8 for 8 GPUs).

Output: Annotations in `data/<dataset_name>/anno/` (e.g., `extract_frame.jsonl`).

### 2. Generate Metadata (`generate_meta_data.py`)

Generates captions for video clips using a specified model (hardcoded as `qwen2vl7b`; edit in script if needed).

Usage:
```
python ./generate_meta_data.py <world_size>
```
- Dataset is hardcoded; modify `main` function for other datasets.

Output: Caption files in `data/<dataset_name>/anno/` (e.g., `clip_captions_qwen2vl7b.jsonl`).

### 3. Generate DPO Data (`generate_dpo_data.py`)

Generates rejected responses based on VidPO configurations (noise on captions: drop, reverse, shuffle at magnitudes 2,4,8,16).

Usage:
```
python ./generate_dpo_data.py <dataset_name> <world_size>
```
- Model hardcoded as `qwen2vl7b`; edit in script.

Output: DPO data in `data/<dataset_name>/dpo-<model_name>/iter_0/` (e.g., `caption_caption_drop_2_False_False.jsonl`).

### 4. Convert to SWIFT Format (`convert_to_swift.py`)

Collects DPO pairs (chosen vs. rejected) and formats them for MS-SWIFT.

Usage:
```
python ./convert_to_swift.py <model_name> <ratio1> <ratio2> ...
```
- `<model_name>`: e.g., `qwen2vl7b`.
- `<ratio>`: Magnitudes like `2` `4` `8` `16`.

Output: JSON file in `./swift_data/` (e.g., `qwen2vl7b_i0_r2-4-8-16.json`).

### 5. Train Models (`train_scripts/run_**`)

Shell scripts for training with MS-SWIFT. Examples include training Qwen2.5-VL-7B with LoRA on SFT data, then DPO.

Usage (example):
```
bash ./train_scripts/qwen25vl7b=lr5e-6_r32_a64_bs16_freeze_r16-r8-r4-r2=lr1e-4_r32_a32_bs16_sft60k.sh
```
- Scripts handle merging checkpoints and running `swift sft`.
- Customize paths and parameters as needed.
- Multiple variants exist for different learning rates, ranks, etc.

Outputs: Trained models in `exp/` directories.


