import gradio as gr

from .pipeline import text2image_ui, image2image_ui
from .process import process_ui
from .hub import model_manager_ui


def ui(proxy=None):
    css = """.gr-small-button {
    max-width: 1.5em !important;
    min-width: 1.5em !important;
    color: #464646 !important;
    }
   """
    with gr.Blocks(title="ArtCraft", css=css) as block:
        with gr.Tabs() as tabs:
            with gr.TabItem("text2image", id=1):
                _, tab1_output, tab1_ops = text2image_ui(["image2image", "process"])
            with gr.TabItem("image2image", id=2):
                tab2_input, tab2_output, tab2_ops = image2image_ui(["image2image", "process"])
            with gr.TabItem("process", id=3):
                tab3_input, tab3_output, tab3_ops = process_ui("artcraft.processors", ["image2image", "process"])
            with gr.TabItem("models", id=4):
                model_manager_ui(proxy=proxy)

        tab1_ops[0].click(lambda x: (gr.update(value=x), gr.Tabs.update(selected=2)), [tab1_output], [tab2_input, tabs])
        tab1_ops[1].click(lambda x: (gr.update(value=x), gr.Tabs.update(selected=3)), [tab1_output], [tab3_input, tabs])

        tab2_ops[0].click(lambda x: (gr.update(value=x), gr.Tabs.update(selected=2)), [tab2_output], [tab2_input, tabs])
        tab2_ops[1].click(lambda x: (gr.update(value=x), gr.Tabs.update(selected=3)), [tab2_output], [tab3_input, tabs])

        tab3_ops[0].click(lambda x: (gr.update(value=x), gr.Tabs.update(selected=2)), [tab3_output], [tab2_input, tabs])
        tab3_ops[1].click(lambda x: (gr.update(value=x), gr.Tabs.update(selected=3)), [tab3_output], [tab3_input, tabs])

    return block


def launch():
    ui().launch()
