set -e

MODEL_PATH="/mnt/ckpts/Qwen2.5-VL-7B-Instruct/"
DATA_PATH="./swift_data/qwen25vl7b_i0_r16.json"
OUTPUT_DIR="./exp/qwen25vl7b_temple_sft60k/outputs/qwen25vl7b=lr5e-6_r32_a64_bs16_freeze_r16"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VIDEO_MIN_PIXELS=3136 \
VIDEO_TOTAL_PIXELS=9000000 \
FPS_MAX_FRAMES=600 \
FPS=1 \
NPROC_PER_NODE=8 \
swift rlhf --rlhf_type dpo \
    --model $MODEL_PATH \
    --model_type qwen2_5_vl --template qwen2_5_vl \
    --dataset $DATA_PATH \
    --train_type lora --lora_rank 32 --lora_alpha 64 \
    --target_modules all-linear --freeze_vit True \
    --learning_rate 5e-6 --warmup_ratio 0.05 \
    --num_train_epochs 1 --max_length 16384 \
    --per_device_train_batch_size 1 --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $(expr 16 / 8 / 1) \
    --eval_steps 50000 --save_steps 500 --logging_steps 5 \
    --dataloader_num_workers 8 --dataset_num_proc 8 \
    --torch_dtype bfloat16 --attn_impl flash_attn --deepspeed zero2 \
    --report_to tensorboard --output_dir $OUTPUT_DIR
