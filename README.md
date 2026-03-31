# AttriCtrl
## Introduction
Paper Link: [AttriCtrl: A Generalizable Framework for Controlling Semantic Attribute Intensity in Diffusion Models Metadata](https://arxiv.org/abs/2508.02151)

Our method enables **fine-grained control over the intensity of semantic attributes** in diffusion models through a **plug-and-play value encoder**.  
Unlike existing text encoders, which cannot interpret numeric intensity or continuous values, **AttriCtrl bridges this gap** and allows precise, interpretable adjustments of aesthetic attributes.
<img width="4455" height="2589" alt="intro" src="https://github.com/user-attachments/assets/d9b5fd9b-ec72-466c-9c35-5b34e21fecc6" />

## Effect
Examples of controlling individual aesthetic attributes.
<img width="4060" height="1934" alt="single" src="https://github.com/user-attachments/assets/cc379fe0-7493-4d22-b470-077641618b0a" />

## Applications
Demonstrations of seamless integration with other frameworks.
<img width="4936" height="1049" alt="app" src="https://github.com/user-attachments/assets/a979b346-5045-4447-aea6-88a49677c29b" />

---

AttriCtrl is **lightweight**, **model-agnostic**, and achieves **continuous controllability** without modifying the underlying diffusion backbone.

## Inference Code
Pretrained weights are provided in `./models/attrictrl`.
1. Installation
```shell
git clone https://github.com/CD22104/AttriCtrl.git
cd AttriCtrl
conda create -n attrictrl python==3.11
pip install -e .
```
2. Example Usage
```python
import os
import time
import torch
from diffsynth import ModelManager
from diffsynth.models.utils import load_state_dict
from diffsynth.pipelines.flux_image import FluxImagePipeline
from diffsynth.attrictrl.attri_encoder import AttriEncoder, load_attri

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


prompt = "a ship"
seed = 100
for attri_value in [0.1,0.5,0.9]:
    image = pipe(
        prompt=prompt, num_inference_steps=30, embedded_guidance=3.5,
        seed=seed, attri_value=attri_value, attri_len=attri_len
    )
    tag_time = time.strftime("%Y%m%d-%H%M%S")
    image.save(f"{img_save_path}/{attri}_{attri_value}_{tag_time}.jpg")

```

## Training
1. Dataset Preparation: Download the [EliGenTrainSet Dataset](https://www.modelscope.cn/datasets/DiffSynth-Studio/EliGenTrainSet)
2. Extract it to: `./dataset/train`. This directory  contains precomputed attribute values for each training image.
3. Training Command
```shell
bash train.sh
```

## Acknowledgement
This project is built upon the Diffsynth framework, with extensions for controllable attribute generation.
