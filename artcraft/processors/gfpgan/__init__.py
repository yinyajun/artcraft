import os
import gradio as gr
from PIL import Image
import numpy as np

from ...hub import download_file, get_or_create_dir, get_default_cache_dir

bg_upscale_methods = ["None", "swin-ir"]


def require():
    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    model_dir = get_or_create_dir(get_default_cache_dir(), "Processors", "gfpgan")
    model_path = os.path.join(model_dir, os.path.basename(url))
    if not os.path.exists(model_path):
        download_file(url, model_path)

    try:
        from .helper import GFPGANHelper
    except:
        os.system("pip install facexlib>=0.2.5")
        os.system("pip install basicsr>=1.4.2")
        os.system("pip install gfpgan==1.3.8")
    return model_path


def process(img, scale: float = 2, bg_upsample_fn: str = "None"):
    if not img:
        return None

    from .helper import GFPGANHelper
    model_path = require()
    helper = GFPGANHelper(model_path, upscale=int(np.ceil(scale)))

    bg_upscaler = None
    if bg_upsample_fn == "swin-ir":
        from ..swin_ir import process as swin_ir_process
        bg_upscaler = swin_ir_process

    out = helper.enhance(np.array(img), bg_upscale_fn=bg_upscaler)
    out = Image.fromarray(out[..., ::-1])  # bgr -> rgb

    w, h = img.size[0], img.size[1]
    ww, hh = int(w * scale), int(h * scale)
    out = out.resize((ww, hh), Image.Resampling.LANCZOS)
    return out


def ui():
    with gr.Row():
        scale = gr.Slider(1., 4., 2., step=0.1, label="scale")
        bg_upsample_fn = gr.Dropdown(label="bg_upscale", choices=bg_upscale_methods, value="None")

    return scale, bg_upsample_fn
