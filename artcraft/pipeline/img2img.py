import torch
from PIL import Image

from ..hub import Type, get_model_path
from ..networks import set_vae, set_lora, set_textual_inversion, set_clip_skip, set_scheduler


class Image2Image:
    def __init__(self,
                 base_model_path: str,
                 vae: str = None,
                 clip_skip: int = 0,
                 enable_lpw: bool = False,
                 lora_specs: list[tuple[str, str | float]] = (),
                 embedding_specs: list[tuple[str, str | float]] = (),
                 control_net_paths: list[str] = (),
                 **kwargs):
        # todo: support cpu
        dtype = torch.float16

        if len(control_net_paths) > 0:
            from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

            control_nets = [ControlNetModel.from_pretrained(path, torch_dtype=dtype) for path in control_net_paths]
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                base_model_path,
                controlnet=control_nets,
                torch_dtype=dtype,
                local_files_only=True,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False)

        elif enable_lpw:
            from .lpw import StableDiffusionLongPromptWeightingPipeline
            pipe = StableDiffusionLongPromptWeightingPipeline.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                local_files_only=True,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False)
        else:
            from diffusers import StableDiffusionImg2ImgPipeline
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                base_model_path,
                torch_dtype=dtype,
                local_files_only=True,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False)

        set_vae(pipe, dtype, vae)  # todo
        set_lora(pipe, 'cuda', dtype, lora_specs)  # todo
        set_clip_skip(pipe, base_model_path, dtype, clip_skip)
        set_textual_inversion(pipe, embedding_specs)

        pipe.to("cuda")

        self.pipe = pipe

    def run(self,
            image,
            prompt,
            neg_prompt="",
            strength=0.8,
            guidance_scale=7.5,
            scheduler=None,
            sampling_steps=30,
            num_images=2,
            seed=-1,
            enable_lpw=False,
            **kwargs):

        set_scheduler(self.pipe, scheduler)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        if getattr(self.pipe, "controlnet", None):
            return self.pipe(
                image=image,
                prompt=prompt,
                negative_prompt=neg_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=sampling_steps,
                generator=generator,
                num_images_per_prompt=num_images,
                control_image=kwargs["control_images"],
                controlnet_conditioning_scale=kwargs["control_condition_scales"],
                guess_mode=kwargs["control_guess_mode"],
                control_guidance_start=kwargs["control_guidance_start"],
                control_guidance_end=kwargs["control_guidance_end"]).images

        run_method = self.pipe.__call__ if not enable_lpw else self.pipe.img2img
        return run_method(image=image, prompt=prompt, negative_prompt=neg_prompt,
                          strength=strength,
                          guidance_scale=guidance_scale,
                          num_inference_steps=sampling_steps,
                          generator=generator,
                          num_images_per_prompt=num_images).images


def image2image(base_model: str,
                vae: str,
                clip_skip: int,
                enable_lpw: bool,
                # -- extra --
                lora_specs: list[tuple[str, str | float]],
                embedding_specs: list[tuple[str, str | float]],
                # -- control net --
                control_nets: list[tuple[str, str]],
                control_images: list[dict],
                control_condition_scales: list[list[float]],
                control_guess_mode: bool,
                control_guidance_start: float,
                control_guidance_end: float,
                # -- run args --
                prompt: str,
                neg_prompt: str,
                image,
                strength: float,
                cfg: float,
                num_images: int,
                scheduler: str,
                sampling_steps: int,
                seed: int,
                enable_vae_tiling: bool, ):
    if prompt == "":
        raise ValueError("prompt is empty")
    if image is None:
        raise ValueError("input image is None")

    base_model_path = get_model_path(Type.CheckpointSD.name, base_model)
    control_net_paths = [get_model_path(Type.ControlNet.name, i[0]) for i in control_nets]
    control_images = [Image.open(i["name"]) for i in control_images]
    control_scales = [i[0] for i in control_condition_scales]

    p = Image2Image(base_model_path=base_model_path,
                    vae=vae,
                    clip_skip=clip_skip,
                    enable_lpw=enable_lpw,
                    lora_specs=lora_specs,
                    embedding_specs=embedding_specs,
                    control_net_paths=control_net_paths)

    return p.run(image,
                 prompt,
                 neg_prompt,
                 strength=strength,
                 guidance_scale=cfg,
                 scheduler=scheduler,
                 sampling_steps=sampling_steps,
                 num_images=num_images,
                 seed=seed,
                 enable_lpw=enable_lpw,
                 enable_vae_tiling=enable_vae_tiling,
                 # -- control net --
                 control_images=control_images,
                 control_condition_scales=control_scales,
                 control_guess_mode=control_guess_mode,
                 control_guidance_start=control_guidance_start,
                 control_guidance_end=control_guidance_end)
