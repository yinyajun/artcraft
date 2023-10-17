import os
import torch
import numpy as np
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize
from facexlib.utils.face_restoration_helper import FaceRestoreHelper
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean


class GFPGANHelper:
    def __init__(self, model_path: str, upscale=2):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=2,
            fix_decoder=False,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True)

        model_dir = os.path.dirname(model_path)

        self.face_helper = FaceRestoreHelper(upscale_factor=upscale,
                                             face_size=512,
                                             crop_ratio=(1, 1),
                                             det_model='retinaface_resnet50',
                                             save_ext='png',
                                             use_parse=True,
                                             device=self.device,
                                             model_rootpath=model_dir)

        model_dict = torch.load(model_path)
        model.load_state_dict(model_dict['params_ema' if 'params_ema' in model_dict else 'params'], strict=True)
        model.eval()
        self.gfpgan = model.to(self.device)

    @torch.no_grad()
    def enhance(self, img: np.array, bg_upscale_fn=None):  # hwc-rgb-uint8
        img = img[:, :, [2, 1, 0]]  # hwc-rgb -> hwc-bgr
        self.face_helper.clean_all()
        self.face_helper.read_image(img)
        self.face_helper.get_face_landmarks_5(only_center_face=False, eye_dist_threshold=5)
        self.face_helper.align_warp_face()

        # face restoration
        for cropped_face in self.face_helper.cropped_faces:
            # prepare data
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to(self.device)

            try:
                output = self.gfpgan(cropped_face_t, return_rgb=False, weight=0.5)[0]
                # convert to image
                restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            except RuntimeError as error:
                print(f'\tFailed inference for GFPGAN: {error}.')
                restored_face = cropped_face

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        if bg_upscale_fn:
            bg_img = bg_upscale_fn(img, scale=self.face_helper.upscale_factor)
            bg_img = bg_img[:, : [2, 1, 0]]  # rgb -> bgr
        else:
            bg_img = None

        self.face_helper.get_inverse_affine(None)
        # paste each restored face to the input image
        restored_img = self.face_helper.paste_faces_to_input_image(upsample_img=bg_img)
        return restored_img
