from ..hub.manage import snapshot_download, query_model
from ..hub.basic import Type


def set_vae(pipe, dtype, method: str):
    vae_info = query_model(Type.VAE.name, method)
    if not vae_info:
        return
    vae_path = snapshot_download(**vae_info)
    from diffusers import AutoencoderKL
    pipe.vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype)
    print(">>", vae_path)
