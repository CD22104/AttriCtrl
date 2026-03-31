import torch
from diffsynth.models.svd_unet import TemporalTimesteps

class AttriEncoder(torch.nn.Module):
    def __init__(self, dim_in, dim_out, attri_len, computation_device=None):
        super().__init__()
        self.attri_len = attri_len
        self.attri_time_proj = TemporalTimesteps(num_channels=dim_in, flip_sin_to_cos=True, downscale_freq_shift=0, computation_device=computation_device)
        self.attri_value_encoder = torch.nn.Sequential(
            torch.nn.Linear(dim_in, dim_out), torch.nn.SiLU(), torch.nn.Linear(dim_out, dim_out)
        )
        self.positional_embedding = torch.nn.Parameter(
            torch.randn(self.attri_len, dim_out) 
        )
        self._initialize_weights()

    def _initialize_weights(self):
        last_linear = self.attri_value_encoder[-1]
        torch.nn.init.zeros_(last_linear.weight)
        torch.nn.init.zeros_(last_linear.bias)

    def forward(self, timestep, dtype):
        attri_emb = self.attri_time_proj(timestep).to(dtype)
        attri_emb = self.attri_value_encoder(attri_emb).squeeze(0)
        attri_emb = attri_emb.expand(self.attri_len, -1)
        final_emb = attri_emb + self.positional_embedding
        return final_emb 

def load_attri(model, state_dict, encoder_name):
    prefix = f"{encoder_name}."
    attri_state_dict = {}
    other_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            attri_state_dict[new_key] = value
        else:
            other_state_dict[key] = value
    missing1, unexpected1 = model.load_state_dict(attri_state_dict, strict=False)
    print(f"Load Attri: Missing: {missing1}, Unexpected: {unexpected1}")
    return model, other_state_dict