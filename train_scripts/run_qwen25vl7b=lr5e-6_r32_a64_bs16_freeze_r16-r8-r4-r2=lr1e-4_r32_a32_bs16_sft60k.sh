set -e
bash ./qwen25vl7b=lr5e-6_r32_a64_bs16_freeze_r16.sh
bash ./qwen25vl7b=lr5e-6_r32_a64_bs16_freeze_r16-r8.sh
bash ./qwen25vl7b=lr5e-6_r32_a64_bs16_freeze_r16-r8-r4.sh
bash ./qwen25vl7b=lr5e-6_r32_a64_bs16_freeze_r16-r8-r4-r2.sh
bash ./qwen25vl7b=lr5e-6_r32_a64_bs16_freeze_r16-r8-r4-r2=lr1e-4_r32_a32_bs16_sft60k.sh