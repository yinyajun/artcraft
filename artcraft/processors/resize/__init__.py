import gradio as gr

methods = ["NEAREST", "LANCZOS", "BILINEAR", "BICUBIC", "BOX", "HAMMING"]


def process(img, method="LANCZOS", scale=2, width=0, height=0):
    if not img:
        return None
    w, h = img.size[0], img.size[1]
    if width > 0 and height > 0 :
        ww, hh = int(width), int(height)
    else:
        ww = int(w * scale / 8) * 8
        hh = int(h * scale / 8) * 8
    out = img.resize((ww, hh), methods.index(method))
    return out


def ui():
    with gr.Group():
        with gr.Row():
            method = gr.Dropdown(label="method", choices=methods, value="LANCZOS")
            scale = gr.Slider(0, 4, 1, step=0.05, label="scale")
        with gr.Row():
            width = gr.Number(0, label="width")
            height = gr.Number(0, label="height")
    return method, scale, width, height
