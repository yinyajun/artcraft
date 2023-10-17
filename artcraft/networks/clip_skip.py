def set_clip_skip(pipe, base_model_path: str, dtype, num=0):
    if 0 < num < 12:
        from transformers import CLIPTextModel
        text_encoder = CLIPTextModel.from_pretrained(
            base_model_path,
            subfolder="text_encoder",
            num_hidden_layers=12 - num,
            torch_dtype=dtype)
        pipe.text_encoder = text_encoder
