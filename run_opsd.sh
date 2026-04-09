#!/usr/bin/env bash
# Single-GPU OPSD training on the Countdown-Tasks-3to4 dataset.
# Adjust CUDA_VISIBLE_DEVICES, model_name_or_path, output_dir, and
# vllm_gpu_memory_utilization for your hardware.

CUDA_VISIBLE_DEVICES=0 python opsd_train.py \
    --model_name_or_path Qwen/Qwen3-1.7B \
    --learning_rate 5e-6 \
    --max_grad_norm 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --output_dir ./outputs/opsd_countdown/ \
    --run_config countdown_qwen31b \
    --num_train_epochs 1 \
    --max_completion_length 1024 \
    --save_steps 25 \
    --logging_steps 2 \
    --attn_implementation flash_attention_2 \
    --torch_dtype bfloat16 \
    --max_length 4096 \
    --beta 0 \
    --use_vllm \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.3 \
    --vllm_tensor_parallel_size 1 \
    --use_peft \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 20 \
    --lmbda 1 \
    --fixed_teacher \
    --jsd_token_clip 0.05 \
    --report_to none
