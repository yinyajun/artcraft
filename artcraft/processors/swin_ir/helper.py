import torch
import numpy as np

from .network_swinir import SwinIR


class SwinIRHelper:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        scale = 4

        model = SwinIR(upscale=scale, in_chans=3, img_size=64, window_size=8,
                       img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                       mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')

        param_key_g = 'params_ema'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(
            pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
            strict=True)

        model.eval()
        self.model = model.to(self.device)

    def enhance(self, img: np.array):  # hwc-rgb-uint8
        window_size = 8
        scale = 4

        # load image
        img = img.astype(np.float32) / 255.  # hwc
        img = np.transpose(img, (2, 0, 1))  # hwc -> chw
        img = torch.from_numpy(img).float().unsqueeze(0).to(self.device)  # chw -> nchw

        # pad image
        _, _, h_old, w_old = img.size()
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]

        with torch.no_grad():
            output = self.model(img)
            output = output[..., :h_old * scale, :w_old * scale]

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output, (1, 2, 0))  # chw -> hwc
        output = (output * 255.0).round().astype(np.uint8)
        return output
