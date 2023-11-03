import dataclasses
import json
import os
import requests
import shutil
from tqdm import tqdm
from enum import Enum
from pathlib import Path
from dataclasses import dataclass


class Source(Enum):
    Local = 1
    Civitai = 2
    HuggingFace = 3
    ModelScope = 4


class Type(Enum):
    CheckpointSD = 1
    CheckpointSDXL = 2
    VAE = 3
    Lora = 4
    Embedding = 5
    ControlNet = 6


@dataclass
class ModelMeta:
    type: str
    source: str
    model_id: str
    revision: str = None
    sub_path: str = None
    weight_file: str = None
    trained_word: str = None


def download_file(url: str, target_file: str, proxy=None):
    response = requests.get(url, stream=True, proxies=proxy)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    progress_bar = tqdm(total=total_size_in_bytes, unit_scale=True, unit_divisor=1024)
    try:
        block_size = 1024
        with open(target_file, 'wb') as f:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                f.write(data)
        progress_bar.close()
    finally:
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            os.remove(target_file)
            raise ValueError(f"download file: {target_file} fails")


def download_thumbnail(thumbnail: str, target_dir: str):
    file = os.path.join(target_dir, "thumbnail.jpg")
    if os.path.exists(file):
        return file

    # copy local
    if os.path.exists(thumbnail):
        shutil.copyfile(thumbnail, file)
        return file

    # download
    try:
        if not os.path.exists(file):
            response = requests.get(thumbnail, stream=True)
            with open(file, 'wb') as out_file:
                shutil.copyfileobj(response.raw, out_file)
            del response
        return file
    except:
        return None


def download_convert_script(tmp_dir: str):
    file = "convert_original_stable_diffusion_to_diffusers.py"
    script_file = os.path.join(tmp_dir, file)
    if not os.path.exists(script_file):
        cmd = f"cd {tmp_dir} && curl -JL --remote-name  https://raw.githubusercontent.com/huggingface/diffusers/v0.20.0/scripts/convert_original_stable_diffusion_to_diffusers.py"
        os.system(cmd)
    if not os.path.exists(script_file):
        raise FileNotFoundError(f"{script_file} download fails")
    return script_file


def convert_to_diffusers(tmp_path: str, dump_path: str):
    # pip install s3fs
    tmp_dir = os.path.dirname(tmp_path)
    convert_script = download_convert_script(tmp_dir)
    cmd = f"python {convert_script} --checkpoint_path {tmp_path} --dump_path {dump_path} --from_safetensors --half"
    os.system(cmd)


def get_default_cache_dir():
    default_cache_dir = Path.home().joinpath(".cache", "artcraft")  # '~/.cache/artcraft'
    return os.environ.get("ARTCRAFT_CACHE", default_cache_dir)


def get_or_create_dir(path: str, *paths: str):
    _path = os.path.join(path, *paths)
    if not os.path.exists(_path):
        Path(_path).mkdir(parents=True)
    return _path


def extract_model_info(model_dir: str):
    file = os.path.join(model_dir, ".extra")
    if not os.path.exists(file):
        return None
    with open(file) as f:
        return ModelMeta(**json.load(f))


def add_model_info(model_dir: str, meta: ModelMeta):
    file = os.path.join(model_dir, ".extra")
    with open(file, "w") as f:
        info = dataclasses.asdict(meta)
        json.dump(info, f)
    return info
