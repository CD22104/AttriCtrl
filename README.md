# AttriCtrl
## 📖 Introduction
Our method enables **fine-grained control over the intensity of semantic attributes** in diffusion models through a **plug-and-play value encoder**.  
Unlike existing text encoders, which cannot interpret numeric intensity or continuous values, **AttriCtrl bridges this gap** and allows precise, interpretable adjustments of aesthetic attributes.
[intro.pdf](https://github.com/user-attachments/files/22526500/intro.pdf)


## 🎚️ Single-Attribute Control
- [Single.pdf](https://github.com/user-attachments/files/22526462/single.pdf): Examples of controlling individual aesthetic attributes.

## 🎛️ Multi-Attribute Control
- Multi-Attribute examples (see `multi.pdf`).

## 🔗 Applications
- [Application.pdf](https://github.com/user-attachments/files/22526464/application.pdf): Demonstrations of seamless integration with other frameworks.

## 🏆 Performance
- [US.pdf](https://github.com/user-attachments/files/22526490/us.pdf): AttriCtrl outperforms related baselines across user studies and quantitative benchmarks.

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
