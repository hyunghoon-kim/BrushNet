import os
from huggingface_hub import hf_hub_download, snapshot_download


save_dir = "input your abs_path"

local_path = snapshot_download(
    repo_id = "krnl/realisticVisionV60B1_v51VAE",
    local_dir = save_dir,
    revision = "main"
)

print(local_path)