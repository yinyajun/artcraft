import os
import json
import requests
import shutil
from .basic import ModelMeta, Type, Source, \
    download_file, convert_to_diffusers, add_model_info, get_or_create_dir, extract_model_info


class Civitai:
    def query_info(self, model_id: str, revision: str, proxy=None):
        url = f"https://civitai.com/api/v1/models/{model_id}"
        resp = json.loads(requests.get(url, proxies=proxy).text)
        versions = resp["modelVersions"]

        version = [v for v in versions if str(v["name"]) == revision]
        if len(version) == 0:
            raise ValueError(f"version {revision} not found")
        version = version[0]

        file = version["files"]
        if len(file) == 0:
            raise ValueError(f"civitai {model_id}:{revision} has no file")
        file = file[0]

        file_name = file.get("name")
        url = file.get("downloadUrl")
        trained_word = version.get("trainedWords")
        images = version.get("images", [])
        if len(images) == 0:
            thumbnail = None
        else:
            thumbnail = images[0]["url"]

        return url, file_name, trained_word, thumbnail

    def snapshot_download(self, _type, model_id, revision, cache_dir=None, proxy=None):
        model_dir = get_or_create_dir(cache_dir, model_id)
        info = extract_model_info(model_dir)
        if info and info.revision == revision:
            print(f"load from local file: {model_dir}")
            return model_dir, extract_model_info(model_dir)
        else:
            shutil.rmtree(model_dir, ignore_errors=True)
            model_dir = get_or_create_dir(cache_dir, model_id)

        url, file_name, trained_words, thumbnail = self.query_info(model_id, revision, proxy)

        # download model
        tmp_dir = get_or_create_dir(cache_dir, "tmp")
        tmp_path = os.path.join(tmp_dir, file_name)
        if not os.path.exists(tmp_path):
            download_file(url, tmp_path, proxy)

        # move
        if _type == Type.CheckpointSD.name:
            convert_to_diffusers(tmp_path, model_dir)
            os.remove(tmp_path)
            weight_file = None
        else:
            shutil.move(tmp_path, model_dir)
            weight_file = file_name

        if type(trained_words) == list and len(trained_words) > 0:
            trained_word = trained_words[0]
        else:
            trained_word = None
        meta = ModelMeta(
            source=Source.Civitai.name,
            type=_type,
            model_id=model_id,
            revision=revision,
            weight_file=weight_file,
            trained_word=trained_word)
        add_model_info(model_dir, meta)
        return model_dir, meta
