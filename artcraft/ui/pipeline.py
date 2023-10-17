from .common import *
from ..pipeline import text2image, image2image


def text2image_ui(destinations):
    base_model_args, lora_specs, embedding_specs, controlnet_args = model_ui()
    generate_btn = operation_ui()
    with gr.Row():
        with gr.Column():
            prompt, neg_prompt = prompt_ui(column_layout=True)
            inference_args = inference_args_ui()
        with gr.Column():
            output_images, output_img = output_image_ui()
            transfer_ops = transfer_ui(destinations)

    model_args = base_model_args + [lora_specs, embedding_specs] + controlnet_args
    predict_args = [prompt, neg_prompt] + list(inference_args)
    generate_btn.click(fn=text2image,
                       inputs=model_args + predict_args,
                       outputs=output_images)
    return None, output_img, transfer_ops


def image2image_ui(destinations):
    base_model_args, lora_specs, embedding_specs, controlnet_args = model_ui()
    generate_btn = operation_ui()
    prompt, neg_prompt = prompt_ui()

    with gr.Row():
        with gr.Column():
            input_img = input_image_ui()
            inference_args = inference_args_ui(input_img)

        with gr.Column():
            output_images, selected_output_img = output_image_ui()
            transfer_ops = transfer_ui(destinations)

    model_args = base_model_args + [lora_specs, embedding_specs] + controlnet_args
    predict_args = [prompt, neg_prompt] + list(inference_args)
    generate_btn.click(fn=image2image,
                       inputs=model_args + predict_args,
                       outputs=output_images)
    return input_img, selected_output_img, transfer_ops
