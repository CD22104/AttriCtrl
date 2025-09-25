# AttriCtrl
## 📖 Introduction
Our method enables **fine-grained control over the intensity of semantic attributes** in diffusion models through a **plug-and-play value encoder**.  
Unlike existing text encoders, which cannot interpret numeric intensity or continuous values, **AttriCtrl bridges this gap** and allows precise, interpretable adjustments of aesthetic attributes.
<img width="4455" height="2589" alt="intro" src="https://github.com/user-attachments/assets/d9b5fd9b-ec72-466c-9c35-5b34e21fecc6" />



## 🎚️ Single-Attribute Control
Examples of controlling individual aesthetic attributes.
<img width="4060" height="1934" alt="single" src="https://github.com/user-attachments/assets/cc379fe0-7493-4d22-b470-077641618b0a" />



## 🎛️ Multi-Attribute Control
Multi-Attribute examples.



## 🔗 Applications
Demonstrations of seamless integration with other frameworks.
<img width="4936" height="1049" alt="app" src="https://github.com/user-attachments/assets/a979b346-5045-4447-aea6-88a49677c29b" />


## 🏆 Performance
AttriCtrl outperforms related baselines across user studies and quantitative benchmarks.



---

AttriCtrl is **lightweight**, **model-agnostic**, and achieves **continuous controllability** without modifying the underlying diffusion backbone.

## Inference Code

```shell
git clone https://github.com/modelscope/DiffSynth-Studio.git  
cd DiffSynth-Studio
pip install -e .
```

```python
import torch
from diffsynth.pipelines.flux_image_new import FluxImagePipeline, ModelConfig


pipe = FluxImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="flux1-dev.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder/model.safetensors"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="text_encoder_2/"),
        ModelConfig(model_id="black-forest-labs/FLUX.1-dev", origin_file_pattern="ae.safetensors"),
        ModelConfig(model_id="DiffSynth-Studio/AttriCtrl-FLUX.1-Dev", origin_file_pattern="models/detail.safetensors")
    ],
)

for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
    image = pipe(prompt="a cat on the beach", seed=2, value_controller_inputs=[i])
    image.save(f"value_control_{i}.jpg")
```
