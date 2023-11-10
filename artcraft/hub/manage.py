import json
import os
from typing import Optional

from .basic import Source, Type, get_default_cache_dir, get_or_create_dir
from .civitai import Civitai


def snapshot_download(source: str,
                      type: str,
                      model_id: str,
                      sub_path: Optional[str] = None,
                      weight_file: Optional[str] = None,
                      revision: Optional[str] = None,
                      cache_dir=None,
                      **kwargs):
    if not cache_dir:
        cache_dir = get_default_cache_dir()
    source_dir = get_or_create_dir(cache_dir, type, source)

    if not source:
        raise ValueError("source is empty")
    if not type:
        raise ValueError("type is empty")

    supported_types = list(Type.__members__)
    if type not in supported_types:
        raise ValueError(f"unsupported type: {type}, must be one of {supported_types}")

    supported_sources = list(Source.__members__)
    if source not in supported_sources:
        raise ValueError(f"unsupported source: {source}, must be one of {supported_sources}")

    if source == Source.Local.name:
        model_dir = model_id
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Local {model_dir} not found")

    elif source == Source.ModelScope.name:
        import modelscope
        model_dir = modelscope.snapshot_download(model_id, revision, cache_dir=source_dir)

    elif source == Source.HuggingFace.name:
        import huggingface_hub
        model_dir = huggingface_hub.snapshot_download(model_id, revision=revision, cache_dir=source_dir)

    elif source == Source.Civitai.name:
        proxy = kwargs.get("proxy", None)
        model_dir, spec = Civitai().snapshot_download(type, model_id, revision, cache_dir=source_dir, proxy=proxy)

    else:
        raise ValueError(f"unsupported source type {source}.")

    path = model_dir
    if sub_path:
        path = os.path.join(model_dir, sub_path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    if weight_file:
        path = os.path.join(path, weight_file)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    return path


def register_model(name, type, source, model_id, revision=None, sub_path=None, weight_file=None, trained_word=None,
                   cache_dir=None):
    if not name:
        raise ValueError("need a name")
    if not cache_dir:
        cache_dir = get_default_cache_dir()

    config = os.path.join(cache_dir, "models.config")
    data = {}
    if os.path.exists(config):
        with open(config, "r") as f:
            data = json.load(f)
    data[type] = data.get(type, {})
    data[type][name] = {"name": name, "type": type, "source": source, "model_id": model_id,
                        "revision": revision, "sub_path": sub_path, "weight_file": weight_file,
                        "trained_word": trained_word}
    with open(config, "w") as f:
        json.dump(data, f)
    return data


def unregister_model(type, name, cache_dir=None):
    if not cache_dir:
        cache_dir = get_default_cache_dir()

    config = os.path.join(cache_dir, "models.config")
    if os.path.exists(config):
        with open(config, "r") as f:
            data = json.load(f)
        data[type] = data.get(type, {})
        data[type].pop(name, None)
        with open(config, "w") as f:
            json.dump(data, f)
    return data


def query_model(type: str, name: str, cache_dir=None):
    if type not in list(Type.__members__):
        raise ValueError(f"invalid type:{type}")

    if not cache_dir:
        cache_dir = get_default_cache_dir()

    config = os.path.join(cache_dir, "models.config")
    if not os.path.exists(config):
        return None
    with open(config, "r") as f:
        data = json.load(f)
    models = data.get(type, {})
    model = models.get(name, None)
    return model


def list_model(type: str, cache_dir=None, simplify=False):
    if type not in list(Type.__members__):
        raise ValueError(f"invalid type:{type}")

    if not cache_dir:
        cache_dir = get_default_cache_dir()

    config = os.path.join(cache_dir, "models.config")
    if not os.path.exists(config):
        return []

    with open(config, "r") as f:
        data = json.load(f)
    models = data.get(type, {})

    if simplify:
        res = list(models.keys())
    else:
        res = [m for m in models.values()]
    return res


def get_model_path(type: str, name: str):
    model = query_model(type, name)
    if not model:
        raise ValueError(f"{type}: {name} not found")
    return snapshot_download(**model)

# if __name__ == '__main__':
#     res = snapshot_download(source=Source.Civitai.name,
#                             type=Type.Embedding.name,
#                             model_id="2032",
#                             revision="Empire")
#     print(res)
#     res = snapshot_download(source=Source.ModelScope.name,
#                             type=Type.Lora.name,
#                             model_id="blackleaverth/rot_bgr",
#                             revision="v1")
#     print(res)
#     # res = snapshot_download(source=Source.HuggingFace.name,
#     #                         type=Type.Lora.name,
#     #                         model_id="Linaqruf/pastel-anime-xl-lora")
#     # print(res)
#
#     register_model("empire", Type.Embedding.name, Source.Civitai.name, "2032", "Empire", trained_word="234")
#     register_model("rot", Type.Lora.name, Source.ModelScope.name, "blackleaverth/rot_bgr", "v1")
#     print(query_model(Type.Embedding.name, "empire"))
#     print(query_model(Type.Lora.name, "rot"))
#     #
#     # print(list_model(Type.Embedding.name, simplify=True))
#     #
#     # print(list_model(Type.Lora.name, simplify=True))
