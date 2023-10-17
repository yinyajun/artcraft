import numpy as np
import os
import gradio as gr
from PIL import Image

from ...hub import get_default_cache_dir, get_or_create_dir, download_file


def require():
    url = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
    model_dir = get_or_create_dir(get_default_cache_dir(), "Processors", "swin_ir")
    model_path = os.path.join(model_dir, os.path.basename(url))
    if not os.path.exists(model_path):
        download_file(url, model_path)

    try:
        from .helper import SwinIRHelper
    except:
        os.system("pip install timm==0.9.5")

    return model_path


def process(img, scale: float):
    if not img:
        return None

    from .helper import SwinIRHelper
    model_path = require()
    helper = SwinIRHelper(model_path)

    out = helper.enhance(np.array(img))
    out = Image.fromarray(out)

    w, h = img.size[0], img.size[1]
    ww, hh = int(w * scale), int(h * scale)
    out = out.resize((ww, hh), Image.Resampling.LANCZOS)
    return out


def ui():
    with gr.Row():
        scale = gr.Slider(1., 4., 2., step=0.1, label="scale")

    return scale
