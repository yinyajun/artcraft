from collections import defaultdict
from ..hub.manage import snapshot_download, query_model
from ..hub.basic import Type


def set_lora(pipe, device, dtype, lora_tags: list):
    # [['light_pollution', '0.25'], ['tiara', '0.25']]
    for tag in lora_tags:
        lora_name = tag[0]
        lora_multiplier = float(tag[1])
        lora_model_info = query_model(Type.Lora.name, lora_name)
        if not lora_model_info:
            continue
        lora_path = snapshot_download(**lora_model_info)
        load_lora_weights(pipe, lora_path, lora_multiplier, device, dtype)
        print(f">> merge lora {lora_name}:{lora_multiplier} done")


def load_lora_weights(pipe, lora_path, multiplier, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"

    import torch
    from safetensors.torch import load_file
    state_dict = load_file(lora_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"
        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():
        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipe.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipe.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        if 'alpha' in elems.keys():
            alpha = elems['alpha'].item() / weight_up.shape[1]
        else:
            alpha = 1.0

        curr_layer.weight.data = curr_layer.weight.data.to(device)
        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * \
                                      torch.mm(weight_up.squeeze(3).squeeze(2),
                                               weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipe
