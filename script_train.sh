# !/bin/bash
accelerate launch --num_processes 1 train_brushnet_sdxl_custom.py \
--pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
--brushnet_model_name_or_path data/ckpt/segmentation_mask_brushnet_ckpt_sdxl_v1 \
--output_dir runs/logs/brushnetsdxl_segmentationmask \
--seed 42 \
--resolution 1024 \
--train_batch_size 1 \
--max_train_steps 80 \
--checkpointing_steps 8 \
--checkpoints_total_limit 3 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-9 \
--train_data_dir data/MyData \
--report_to tensorboard \
--validation_steps 8 \
--validation_prompt "asian, female, around 20 years old, black wavy hair, wearing red dress" "asian, female, around 20 years old, black wavy hair, wearing black top and white pants" \
--validation_image "/workspace/BrushNet/test_samples/img/image_1018.png" "/workspace/BrushNet/test_samples/img/image_2318.png" \
--validation_mask "/workspace/BrushNet/test_samples/inv_mask/image_1018.png" "/workspace/BrushNet/test_samples/inv_mask/image_2318.png" \
--num_validation_images 1 \
--resume_from_checkpoint latest \
--tracker_project_name brushnet

################################################################################
# --train_data_dir data/BrushData
# --gradient_checkpointing \
# --set_grads_to_none \
# --mixed_precision bf16 \
# --enable_xformers_memory_efficient_attention \
# --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
# --output_dir runs/logs/brushnetsdxl_segmentationmask \
# --train_data_dir data/BrushData \
# --resolution 1024 \
# --learning_rate 1e-5 \
# --train_batch_size 1 \
# --gradient_accumulation_steps 4 \
# --tracker_project_name brushnet \
# --report_to tensorboard \
# --resume_from_checkpoint latest \
# --validation_steps 300 \
# --checkpointing_steps 10000 
################################################################################