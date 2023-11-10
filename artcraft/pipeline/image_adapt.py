import torch
from ..networks import set_vae, set_lora, set_textual_inversion, set_clip_skip, set_scheduler


class Adapter:
    def __init__(self,
                 adapt_model: str,
                 adapt_image_encoder: str,
                 base_model_path: str,
                 vae: str = None,
                 clip_skip: int = 0,
                 lora_specs: list[tuple[str, str | float]] = (),
                 embedding_specs: list[tuple[str, str | float]] = (),
                 **kwargs):
        dtype = torch.float16

        from .sd import MyselfStableDiffusionPipeline
        pipe = MyselfStableDiffusionPipeline.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            local_files_only=True,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False
        )

        set_vae(pipe, dtype, vae)  # todo
        set_lora(pipe, 'cuda', dtype, lora_specs)  # todo
        set_clip_skip(pipe, base_model_path, dtype, clip_skip)
        set_textual_inversion(pipe, embedding_specs)
        pipe.to("cuda")
        self.pipe = pipe

        from .ip_adapter.ip_adapter import IPAdapterPlus
        self.ip_model = IPAdapterPlus(pipe, image_encoder_path=adapt_image_encoder,
                                      ip_ckpt=adapt_model, device="cuda", num_tokens=16)

    def run(self,
            reference_image,
            prompt,
            neg_prompt="",
            guidance_scale=7.5,
            scheduler=None,
            sampling_steps=30,
            num_images=2,
            seed=-1,
            adapt_scale=0.5,
            **kwargs):
        set_scheduler(self.pipe, scheduler)

        return self.ip_model.generate(
            pil_image=reference_image,
            prompt=prompt,
            negative_prompt=neg_prompt,
            guidance_scale=guidance_scale,
            seed=seed,
            num_inference_steps=sampling_steps,
            num_samples=num_images,
            scale=adapt_scale)
