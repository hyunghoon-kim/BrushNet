# !/bin/bash
accelerate launch --num_processes 8 train_brushnet_sdxl_custom.py \
--pretrained_model_name_or_path RunDiffusion/Juggernaut-XL-v9 \
--variant fp16 \
--brushnet_model_name_or_path data/ckpt/segmentation_mask_brushnet_ckpt_sdxl_v1 \
--output_dir runs/logs/brushnetsdxl_segmentationmask \
--seed 42 \
--resolution 1024 \
--train_batch_size 1 \
--train_dataset_len 487 \
--max_train_steps 4870 \
--checkpointing_steps 487 \
--checkpoints_total_limit 3 \
--gradient_accumulation_steps 1 \
--learning_rate 1e-5 \
--train_data_dir data/MyData \
--report_to wandb \
--validation_steps 487 \
--validation_prompt "asian, female, around 20 years old, black wavy hair, wearing red dress" "asian, female, around 20 years old, black wavy hair, wearing black top and white pants" \
--validation_image "/workspace/BrushNet/test_samples/img/image_1018.png" "/workspace/BrushNet/test_samples/img/image_2318.png" \
--validation_mask "/workspace/BrushNet/test_samples/inv_mask/image_1018.png" "/workspace/BrushNet/test_samples/inv_mask/image_2318.png" \
--num_validation_images 1 \
--resume_from_checkpoint latest \
--tracker_project_name brushnet \
--wandb_run_name test_encode0

## pretrained_model_name_or_path
# RunDiffusion/Juggernaut-XL-v9 # if training this ckpt add argument --variant fp16
# stabilityai/stable-diffusion-xl-base-1.0

## report_to
# tensorboard
# wandb