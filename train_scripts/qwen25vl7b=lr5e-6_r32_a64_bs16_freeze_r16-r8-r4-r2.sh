set -e

version_dirs=( "./exp/qwen25vl7b_temple_sft60k/outputs/qwen25vl7b=lr5e-6_r32_a64_bs16_freeze_r16-r8-r4"/v*-* )
if [ ${#version_dirs[@]} -eq 0 ]; then
    echo "No version dirs found"
    exit 1
fi
sorted_versions=$(printf "%s\n" "${version_dirs[@]}" | sed -E 's|.*/v([0-9]+)-.*|\1 \0|' | sort -n -k1,1 | awk '{print $2}')
latest_version_dir=$(echo "$sorted_versions" | tail -n 1)

unmerged_checkpoints=( "$latest_version_dir"/checkpoint-[0-9]* )
unmerged_latest=""
if [ ${#unmerged_checkpoints[@]} -eq 0 ]; then
    echo "No unmerged checkpoints found"
    exit 1
fi

unmerged_sorted=$(printf "%s\n" "${unmerged_checkpoints[@]}" | grep -Ev 'checkpoint-[0-9]+-merged' | sed -E 's|.*/checkpoint-([0-9]+)$|\1 \0|' | sort -n -k1,1 | awk '{print $2}')
unmerged_latest=$(echo "$unmerged_sorted" | tail -n 1)
merged_latest=${unmerged_latest}-merged

if [ ! -d "$merged_latest" ]; then
    echo "No merged latest checkpoint found; Merging unmerged latest checkpoint: $unmerged_latest"
    swift export --merge_lora true \
        --adapters $unmerged_latest
    echo "Merged latest checkpoint: $merged_latest"
else
    echo "Merged latest checkpoint found: $merged_latest"
fi

DATA_PATH="./swift_data/qwen25vl7b_i0_r2.json"
OUTPUT_DIR="./exp/qwen25vl7b_temple_sft60k/outputs/qwen25vl7b=lr5e-6_r32_a64_bs16_freeze_r16-r8-r4-r2"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VIDEO_MIN_PIXELS=3136 \
VIDEO_TOTAL_PIXELS=9000000 \
FPS_MAX_FRAMES=600 \
FPS=1 \
NPROC_PER_NODE=8 \
swift rlhf --rlhf_type dpo \
    --model "$merged_latest" \
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
