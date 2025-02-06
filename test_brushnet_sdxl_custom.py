from diffusers import StableDiffusionXLBrushNetPipeline, BrushNetModel, DPMSolverMultistepScheduler, UniPCMultistepScheduler, AutoencoderKL
import torch
import cv2
import numpy as np
from PIL import Image
from glob import glob
import os

# choose the base model here
# base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"
# base_model_path = "RunDiffusion/Juggernaut-XL-v9"
base_model_path = "/workspace/sd_ckpts/d****"

# input brushnet ckpt path
# brushnet_path = "data/ckpt/segmentation_mask_brushnet_ckpt_sdxl_v1" # original
brushnet_path = "/workspace/BrushNet/runs/logs/brushnetsdxl_segmentationmask" # v1

# result_backbone_brushnet (e.x) result_draph-v1_org
output_dirname = "result_d****-v1_ft1"

if os.path.exists(f"test_samples/{output_dirname}") == False:
    os.makedirs(f"test_samples/{output_dirname}", exist_ok=True)

# choose whether using blended operation
blended = True
seed = 42

# input source image / mask image path and the text prompt
image_paths = sorted(glob("/workspace/BrushNet/test_samples/img/*.png"))
mask_paths = sorted(glob("/workspace/BrushNet/test_samples/mask/*.png"))
caption_paths = sorted(glob("/workspace/BrushNet/test_samples/cap/*.txt"))
captions = [open(path).read().strip() for path in caption_paths]

# conditioning scale
brushnet_conditioning_scale=1.0

brushnet = BrushNetModel.from_pretrained(
    brushnet_path, 
    torch_dtype=torch.float16
    )

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16
    )

pipe = StableDiffusionXLBrushNetPipeline.from_pretrained(
    base_model_path,
    vae=vae,
    brushnet=brushnet,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
    use_safetensors=True,
    variant="fp16"
    )

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
# pipe.enable_xformers_memory_efficient_attention()

# memory optimization.
pipe.enable_model_cpu_offload()


for image_path, mask_path, caption in zip(image_paths, mask_paths, captions):
    _, filename = os.path.split(image_path)
    print(filename, caption)

    init_image = cv2.imread(image_path)[:,:,::-1]
    mask_image = 1.*(cv2.imread(mask_path).sum(-1)>255)
    mask_image = 1. - mask_image # inversion

    # resize image
    h,w,_ = init_image.shape
    if w<h:
        scale=1024/w
    else:
        scale=1024/h
    new_h=int(h*scale)
    new_w=int(w*scale)

    init_image=cv2.resize(init_image,(new_w,new_h))
    mask_image=cv2.resize(mask_image,(new_w,new_h))[:,:,np.newaxis]

    init_image = init_image * (1-mask_image)

    init_image = Image.fromarray(init_image.astype(np.uint8)).convert("RGB")
    mask_image = Image.fromarray(mask_image.astype(np.uint8).repeat(3,-1)*255).convert("RGB")

    generator = torch.Generator("cuda").manual_seed(seed) # seed assign

    image = pipe(
        prompt=caption, 
        image=init_image, 
        mask=mask_image, 
        num_inference_steps=50, 
        generator=generator,
        brushnet_conditioning_scale=brushnet_conditioning_scale
    ).images[0]

    if blended:
        image_np=np.array(image)
        init_image_np=cv2.imread(image_path)[:,:,::-1]
        mask_np = 1.*(cv2.imread(mask_path).sum(-1)>255)[:,:,np.newaxis]
        mask_np = 1. - mask_np

        # blur, you can adjust the parameters for better performance
        mask_blurred = cv2.GaussianBlur(mask_np*255, (21, 21), 0)/255
        mask_blurred = mask_blurred[:,:,np.newaxis]
        mask_np = 1-(1-mask_np) * (1-mask_blurred)
        org_h, org_w, _ = init_image_np.shape
        image_np = cv2.resize(image_np, (org_w, org_h))
        image_pasted=init_image_np * (1-mask_np) + image_np*mask_np
        image_pasted=image_pasted.astype(image_np.dtype)
        image=Image.fromarray(image_pasted)

    # print(filename)
    image.save(f"test_samples/{output_dirname}/{filename}")
    # break
