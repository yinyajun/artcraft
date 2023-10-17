import cv2
import numpy as np
from PIL import Image
import gradio as gr


def process(img, low_threshold=100, high_threshold=200):
    if not img:
        return None

    image = np.array(img)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def ui():
    with gr.Row():
        low_th = gr.Number(100, label="low threshold")
        high_th = gr.Number(200, label="high threshold")
    return low_th, high_th
