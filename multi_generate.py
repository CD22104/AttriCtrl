# Adapted from the Diffsynth framework
# Modified for AttriCtrl-based controllable generation
import os
import time
import torch
from diffsynth import ModelManager
from diffsynth.models.utils import load_state_dict
from diffsynth.pipelines.flux_image_multi import FluxImagePipeline
from diffsynth.attrictrl.attri_encoder import AttriEncoder, load_attri

def load_encoders(attri_list, device):
    attri_n = len(attri_list)
    attri_encoders = []
    for i in range(attri_n):
        attri = attri_list[i]
        ckpt_path = os.path.join(f"./models/attrictrl/{attri}.ckpt")
        state_dict = load_state_dict(ckpt_path)

        attri_encoder = AttriEncoder(256, 4096, 32).to(dtype=torch.bfloat16, device=device)
        attri_encoder, other_state_dict = load_attri(attri_encoder, state_dict, "attri_encoder")
        attri_encoders.append(attri_encoder)
    return attri_encoders

def get_encoder_res(attri_encoders, attri_values, device):
    attri_res = []
    for i in range(len(attri_values)):
        attri_value = attri_values[i]
        attri_encoder = attri_encoders[i]
        attri_value_tensor = torch.tensor([attri_value], dtype=torch.bfloat16, device=device)
        attri_value_emb = attri_encoder(attri_value_tensor*1000, dtype=torch.bfloat16)
        attri_res.append(attri_value_emb)
    return attri_res

if __name__ == "__main__":
    # Please download FLUX.1-dev first.
    model_path = "./models/black-forest-labs/FLUX.1-dev"
    device = "cpu"
    model_manager = ModelManager(torch_dtype=torch.bfloat16, device=device,
                                        file_path_list=[
                                            model_path + "/text_encoder/model.safetensors",
                                            model_path + "/text_encoder_2",
                                            model_path + "/ae.safetensors",
                                            model_path + "/flux1-dev.safetensors"
                                        ])
    pipe = FluxImagePipeline.from_model_manager(model_manager)
    pipe.enable_vram_management(num_persistent_param_in_dit=2e9)
    img_save_path = os.path.join("./results", "multi")
    os.makedirs(img_save_path, exist_ok=True)

    attri_list = ['brightness','realism','detail']
    attri_encoder = load_encoders(attri_list, device)

    prompt = "A small cabin in the forest."
    seed = 0
    attri_value_list = [
       [0.4,0.1,0.9],
       [0.9,0.4,0.9],
    ]

    for attri_value in attri_value_list:
        attri_res = get_encoder_res(attri_encoder, attri_value, device)

        image = pipe(
            prompt=prompt, num_inference_steps=30, embedded_guidance=3.5,
            seed=seed, attri_value=attri_res, attri_type=attri_list,
        ) 

        image_name = ""
        tag_time = time.strftime("%Y%m%d-%H%M%S")
        for i in range(len(attri_value)):
            image_name = image_name+"_"+attri_list[i]+"_"+str(attri_value[i])
        image.save(f"{img_save_path}/{image_name}.jpg")
        print(f"Save:{image_name}!")
