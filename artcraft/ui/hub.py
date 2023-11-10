import gradio as gr
from ..hub import Type, Source, list_model, snapshot_download, register_model, unregister_model

def model_ui():
    with gr.Group():
        with gr.Row():
            name = gr.Textbox(label="name", value=None)
            type = gr.Dropdown(choices=list(Type.__members__), label="type")
            source = gr.Dropdown(choices=list(Source.__members__), label="source")
        with gr.Row():
            model_id = gr.Text(label="model id")
            revision = gr.Text(label="revision", value=None)
            sub_path = gr.Text(label="sub path", value=None)
        with gr.Accordion("more", open=False):
            with gr.Row():
                weight_file = gr.Text(label="weight file", value=None)
                trained_word = gr.Text(label="trained word", value=None)
    return name, type, source, model_id, revision, sub_path, weight_file, trained_word


def registered_models_ui(headers):
    def models_table(value):
        table = gr.DataFrame(
            headers=headers,
            datatype=["str", "str", "str", "str", "str", "str", "str", "str"],
            label="registered models",
            interactive=False)
        if len(value) > 0:
            table.value = value
        return table

    with gr.Tabs():
        tables = {}
        for k in list(Type.__members__):
            with gr.TabItem(k):
                tables[k] = models_table([[m[t] for t in headers] for m in list_model(k)])
    return tables


def model_manager_ui(proxy):
    headers = ["name", "type", "source", "model_id", "revision", "sub_path", "weight_file", "trained_word"]

    with gr.Row(variant="compact"):
        with gr.Column():
            name, type, source, model_id, revision, sub_path, weight_file, trained_word = model_ui()
            with gr.Row():
                remove_btn = gr.Button("remove")
                refresh_btn = gr.Button("refresh")
                add_btn = gr.Button("add", variant="primary")
    info = gr.Text(label="model info")
    tables = registered_models_ui(headers)

    def fill(evt: gr.SelectData, table):
        row, _ = evt.index
        return list(table.iloc[row, :].to_dict().values())

    def register(name, type, source, model_id, sub_path, weight_file, revision, trained_word):
        if not revision:
            revision = None

        path = snapshot_download(source, type, model_id, sub_path, weight_file, revision, proxy=proxy)
        register_model(name, type, source, model_id, revision, sub_path, weight_file, trained_word)

        return path

    def unregister(type, name):
        unregister_model(type, name)
        return f"remove {name}"

    def refresh_models():
        res = []
        for k, table in tables.items():
            new_list = [[m[t] for t in headers] for m in list_model(k)]
            if len(new_list) == 0:
                res.append([["", "", "", "", "", "", "", ""]])
            else:
                res.append(new_list)
        return res

    for k, table in tables.items():
        table.select(fill, inputs=[table],
                     outputs=[name, type, source, model_id, revision, sub_path, weight_file, trained_word])

    add_btn.click(register, inputs=[name, type, source, model_id, sub_path, weight_file, revision, trained_word],
                  outputs=[info])
    remove_btn.click(unregister, inputs=[type, name], outputs=[info])
    refresh_btn.click(refresh_models, outputs=list(tables.values()))
