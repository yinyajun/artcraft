from ..hub.manage import snapshot_download, query_model
from ..hub.basic import Type


def set_textual_inversion(pipe, embedding_tags: list):
    if len(embedding_tags) == 0:
        return
    paths = []
    tokens = []
    for tag in embedding_tags:
        name, weight = tag[0], tag[1]
        embedding_info = query_model(Type.Embedding.name, name)
        if not embedding_info:
            continue
        embedding_path = snapshot_download(**embedding_info)
        paths.append(embedding_path)
        tokens.append(embedding_info["trained_word"])
    pipe.load_textual_inversion(pretrained_model_name_or_path=paths, token=tokens)
