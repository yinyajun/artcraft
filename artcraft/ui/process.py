import gradio as gr
from ..processors import load_processors


def process_ui(module_name: str, destinations: list):
    processors = load_processors(module_name)

    with gr.Column():
        for name, p in processors.items():
            with gr.Tab(name):
                components = p["ui"]()
                if isinstance(components, tuple):
                    p["args"] = list(components)
                else:
                    p["args"] = [components]
                p["btn"] = gr.Button("process", variant="primary")

        with gr.Row(variant="compact"):
            with gr.Column():
                input_img = gr.Image(label="input image", type="pil", height=350)
                with gr.Row():
                    input_w = gr.Number(0, label="width", interactive=False)
                    input_h = gr.Number(0, label="height", interactive=False)
            with gr.Column():
                processed_img = gr.Image(label="processed image", type="pil", height=350)
                with gr.Row():
                    processed_w = gr.Number(0, label="width", interactive=False)
                    processed_h = gr.Number(0, label="height", interactive=False)

        for name, p in processors.items():
            p["btn"].click(p["process"], [input_img] + p["args"], [processed_img])

        def image_size(img):
            if img:
                w, h = img.size[0], img.size[1]
            else:
                w, h = 0, 0
            return w, h

        input_img.change(image_size, [input_img], [input_w, input_h])
        processed_img.change(image_size, [processed_img], [processed_w, processed_h])

        transfer_ops = transfer_ui(destinations)

    return input_img, processed_img, transfer_ops


def transfer_ui(destinations: list):
    with gr.Row():
        buttons = [gr.Button(f"🔁 {d}") for d in destinations]
    return buttons
