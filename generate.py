# Adapted from the Diffsynth framework
# Modified for AttriCtrl-based controllable generation
import os
import time
import torch
from diffsynth import ModelManager
from diffsynth.models.utils import load_state_dict
from diffsynth.pipelines.flux_image import FluxImagePipeline
from diffsynth.attrictrl.attri_encoder import AttriEncoder, load_attri

# Please download FLUX.1-dev first.
model_path = "./models/black-forest-labs/FLUX.1-dev"
device = "cuda"
model_manager = ModelManager(torch_dtype=torch.bfloat16, device=device,
                                     file_path_list=[
                                         model_path + "/text_encoder/model.safetensors",
                                         model_path + "/text_encoder_2",
                                         model_path + "/ae.safetensors",
                                         model_path + "/flux1-dev.safetensors"
                                     ])
pipe = FluxImagePipeline.from_model_manager(model_manager)

attri = "brightness" # brightness, detail, realism
attri_len = 32
img_save_path = os.path.join("./results", attri)
os.makedirs(img_save_path, exist_ok=True)
ckpt_path = os.path.join(f"./models/attrictrl/{attri}.ckpt")
state_dict = load_state_dict(ckpt_path)

attri_encoder = AttriEncoder(256, 4096, attri_len).to(dtype=torch.bfloat16, device=device)
attri_encoder, other_state_dict = load_attri(attri_encoder, state_dict, "attri_encoder")
pipe.attri_encoder = attri_encoder
# pipe.enable_vram_management(num_persistent_param_in_dit=2e9)

prompt = "a ship"
seed = 100
for attri_value in [0.1,0.5,0.9]:
    image = pipe(
        prompt=prompt, num_inference_steps=30, embedded_guidance=3.5,
        seed=seed, attri_value=attri_value, attri_len=attri_len
    )
    tag_time = time.strftime("%Y%m%d-%H%M%S")
    image.save(f"{img_save_path}/{attri}_{attri_value}_{tag_time}.jpg")
