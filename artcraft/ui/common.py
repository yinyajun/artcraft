import gradio as gr

from ..hub import list_model, Type
from ..networks import scheduler_list
from ..processors import load_processors


def prompt_ui(column_layout=False):
    if column_layout:
        with gr.Column():
            prompt = gr.Textbox(label="prompt", placeholder="inputs prompt", lines=6, max_lines=6)
            neg_prompt = gr.Text(label="neg prompt", placeholder="inputs negative prompt", lines=6, max_lines=6)
        return prompt, neg_prompt
    with gr.Row():
        prompt = gr.Text(label="prompt", placeholder="inputs prompt", lines=6, max_lines=6)
        neg_prompt = gr.Text(label="neg prompt", placeholder="inputs negative prompt", lines=6, max_lines=6)
    return prompt, neg_prompt


def input_image_ui():
    return gr.Image(label="input image", type="pil", height=312)


def output_image_ui():
    output_images = gr.Gallery(label="out image", columns=2, object_fit="contain", height=700)
    selected_output_img = gr.Image(visible=False)

    def select_img(evt: gr.SelectData, images):
        return images[evt.index]["name"]

    output_images.select(select_img, [output_images], [selected_output_img])
    return output_images, selected_output_img


def inference_args_ui(input_img=None):
    with gr.Group():
        with gr.Row():
            width = gr.Slider(64, 2048, 512, step=8, label="width")
            height = gr.Slider(64, 2048, 512, step=8, label="height")

        with gr.Row():
            num_images = gr.Slider(1, 16, 2, step=1, label="num images")
            cfg = gr.Slider(1, 20, 7.5, step=0.1, label="CFG scale")

        with gr.Row():
            scheduler = gr.Dropdown(scheduler_list(), label="scheduler", value=scheduler_list()[0])
            sampling_steps = gr.Slider(10, 60, 35, step=1, label="sampling steps")

        with gr.Row():
            seed = gr.Number(-1, precision=0, label="random seed")
        with gr.Row():
            strength = gr.Slider(0, 1, 0.75, step=0.01, label="strength")

        with gr.Row():
            with gr.Accordion("more setting", open=False):
                with gr.Row():
                    enable_vae_tiling = gr.Checkbox(value=False, label="enable vae tiling")

    if not input_img:
        strength.visible = False
        return cfg, num_images, width, height, scheduler, sampling_steps, seed, enable_vae_tiling
    else:
        height.interactive = False
        width.interactive = False
        enable_vae_tiling.visible = False  # not support
        input_img.change(lambda img: (gr.update(value=img.size[0] if img else 0),
                                      gr.update(value=img.size[1] if img else 0)),
                         [input_img], [width, height])

        return input_img, strength, cfg, num_images, scheduler, sampling_steps, \
               seed, enable_vae_tiling


def model_ui():
    # base
    base_models = list_model(Type.CheckpointSD.name, simplify=True)
    default_base_model = base_models[0] if len(base_models) > 0 else None
    vae_list = ["automatic"] + list_model(Type.VAE.name, simplify=True)
    with gr.Group():
        with gr.Column():
            with gr.Row():
                base_model = gr.Dropdown(choices=base_models, value=default_base_model, label="base model", scale=2)
                vae = gr.Dropdown(vae_list, value=vae_list[0], label="vae", scale=2)
                refresh_btn = gr.Button("🔄", elem_classes="gr-small-button", scale=1)

            with gr.Accordion("more setting", open=False):
                with gr.Row():
                    clip_skip = gr.Slider(0, 10, 0, step=1, label="clip skip")
                with gr.Row():
                    enable_lpw = gr.Checkbox(value=True, label="enable lpw")
                    enable_lora = gr.Checkbox(value=False, label="enable lora")
                    enable_embedding = gr.Checkbox(value=False, label="enable embedding")
                    enable_controlnet = gr.Checkbox(value=False, label="enable controlnet")

    def refresh():
        return gr.Dropdown.update(choices=list_model(Type.CheckpointSD.name, simplify=True)), \
               gr.Dropdown.update(choices=["automatic"] + list_model(Type.VAE.name, simplify=True))

    refresh_btn.click(fn=refresh, inputs=[], outputs=[base_model, vae])
    base_model_args = [base_model, vae, clip_skip, enable_lpw]

    # lora
    with gr.Accordion("lora", open=True, visible=False) as lora_item:
        lora_specs = _weighted_merger_ui(Type.Lora.name, 0.25)

    enable_lora.change(lambda x, y: (gr.update(visible=x), gr.update(value=y if x else [])),
                       [enable_lora, lora_specs], [lora_item, lora_specs])

    # embedding
    with gr.Accordion("embedding", open=True, visible=False) as embedding_item:
        embedding_specs = _weighted_merger_ui(Type.Embedding.name, 1.0)

    enable_embedding.change(lambda x, y: (gr.update(visible=x), gr.update(value=y if x else [])),
                            [enable_embedding, embedding_specs], [embedding_item, embedding_specs])

    # control net
    control_nets = gr.Highlightedtext(value=[], visible=False)
    control_images = gr.Gallery(visible=False)
    control_scales = gr.Highlightedtext(value=[], visible=False)

    with gr.Accordion("controlnet", open=True, visible=False) as controlnet_item:
        with gr.Tab("ControlNet 1"):
            _controlnet_ui("1", control_nets, control_images, control_scales)
        with gr.Tab("ControlNet 2"):
            _controlnet_ui("2", control_nets, control_images, control_scales)
        with gr.Tab("ControlNet 3"):
            _controlnet_ui("3", control_nets, control_images, control_scales)

        with gr.Accordion("more setting", open=False):
            guess_mode = gr.Checkbox(label="guess mode", value=False)
            with gr.Row():
                control_guidance_start = gr.Slider(0, 1, 0.0, step=0.05, label="control guidance start")
                control_guidance_end = gr.Slider(0, 1, 1.0, step=0.05, label="control guidance end")

    enable_controlnet.change(lambda x, y, z, k: (
        gr.update(visible=x),
        gr.update(value=y if x else []),
        gr.update(value=z if x else []),
        gr.update(value=k if x else []),
    ), [enable_controlnet, control_nets, control_images, control_scales],
                             [controlnet_item, control_nets, control_images, control_scales])

    contorlnet_args = [control_nets, control_images, control_scales, guess_mode, control_guidance_start,
                       control_guidance_end]
    return base_model_args, lora_specs, embedding_specs, contorlnet_args


def transfer_ui(destinations: list):
    with gr.Row():
        buttons = [gr.Button(f"🔁 {d}") for d in destinations]
    return buttons


def operation_ui():
    with gr.Row():
        generate_btn = gr.Button("generate", variant="primary")
    return generate_btn


def _controlnet_ui(i, nets, images, scales):
    models = list_model(Type.ControlNet.name, simplify=True)
    processors = ["canny"]
    with gr.Group():
        with gr.Row():
            refresh_btn = gr.Button("🔄", elem_classes="gr-small-button", scale=1)
            net = gr.Dropdown(label="model", choices=models, value=models[0] if len(models) > 0 else None)
            processor = gr.Dropdown(label="processor", choices=processors, value=processors[0])
            process_btn = gr.Button("➡", elem_classes="gr-small-button", variant="primary", scale=1)
        scale = gr.Slider(0, 1, 1, step=0.1, label="condition scale")
        refresh_btn.click(fn=lambda: gr.Dropdown.update(choices=list_model(Type.ControlNet.name, simplify=True)),
                          outputs=[net])

        with gr.Row():
            input_img = gr.Image(label="input image", type="pil")
            control_img = gr.Image(label="control image", type="pil")

        def process(img, method):
            p = load_processors("artcraft.processors")
            process_fn = p[method]["process"]
            return process_fn(img)

        process_btn.click(process, [input_img, processor], [control_img])

    def update(net, control_image, scale, nets, images, scales):
        _images = []
        for img in images:
            if type(img) == dict:
                _images.append(img["name"])
                continue
            _images.append(img)

        # update
        for idx, _net in enumerate(nets):
            if _net[1] == i:
                if control_image:
                    nets[idx] = (net, i)
                    _images[idx] = control_image
                    scales[idx] = (scale,)
                else:
                    nets.pop(idx)
                    _images.pop(idx)
                    scales.pop(idx)
                return nets, _images, scales

        # add
        if control_image:
            nets.append((net, i))
            _images.append(control_image)
            scales.append((scale,))
        return nets, _images, scales

    control_img.change(update, [net, control_img, scale, nets, images, scales], [nets, images, scales])
    scale.change(update, [net, control_img, scale, nets, images, scales], [nets, images, scales])
    net.change(update, [net, control_img, scale, nets, images, scales], [nets, images, scales])


def _weighted_merger_ui(type: str, default_multiply: float):
    def merge(model, multiplier, chosen):
        chosen.append((model, multiplier))
        existed = {i[0]: i[1] for i in chosen}
        return [[i[0], str(i[1])] for i in list(existed.items()) if i[1] != 0]

    def remove(evt: gr.SelectData, chosen):
        chosen.pop(evt.index)
        return chosen

    choices = list_model(type, simplify=True)
    default_choice = choices[0] if len(choices) > 0 else None
    with gr.Group():
        with gr.Row():
            refresh_btn = gr.Button("🔄", elem_classes="gr-small-button", scale=1)
            model = gr.Dropdown(choices=choices, value=default_choice, label="model", scale=3)
            multiplier = gr.Number(default_multiply, minimum=0, maximum=1, step=0.01, label="multiplier",
                                   scale=3)
            merge_btn = gr.Button("➕", elem_classes="gr-small-button", scale=1, variant="primary")
        chosen = gr.Highlightedtext(value=[], label="chosen")

    refresh_btn.click(fn=lambda: gr.Dropdown.update(choices=list_model(type, simplify=True)), outputs=[model])
    merge_btn.click(merge, inputs=[model, multiplier, chosen], outputs=[chosen])
    chosen.select(remove, inputs=[chosen], outputs=[chosen])

    return chosen
