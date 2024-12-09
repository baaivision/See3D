import torch
import torch.nn.functional as F
import inspect
import numpy as np
from typing import Callable, List, Optional, Union
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor,CLIPVisionModelWithProjection
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.utils import (
    deprecate,
    is_accelerate_available,
    is_accelerate_version,
    logging,
)
from diffusers import DDPMScheduler
from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

from einops import rearrange, repeat

from mv_unet import MultiViewUNetModel, get_camera


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def custom_decay_function_weight(t):
    # t = torch.tensor(t, dtype=torch.float32)
    
    t_peak = 200
    # t_peak = 333
    t_slow_decay_end = 60
    # t_slow_decay_end = 40
    value_at_slow_decay_end = 0.8
    
    # Initialize output tensor
    values = torch.zeros_like(t)
    
    # Slow decay mask (t >= t_slow_decay_end)
    slow_decay_mask = t >= t_slow_decay_end
    values[slow_decay_mask] = 1.0 - (1.0 - value_at_slow_decay_end) * (t_peak - t[slow_decay_mask]) / (t_peak - t_slow_decay_end)
    
    # Fast decay mask (t < t_slow_decay_end)
    fast_decay_mask = t < t_slow_decay_end
    values[fast_decay_mask] = value_at_slow_decay_end * torch.exp(-0.075 * (t_slow_decay_end - t[fast_decay_mask]))
    
    # Ensure values are within [0, 1]
    values = torch.clamp(values, 0, 1)
    
    return values


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def compute_weights(timesteps, max_time=333.0, min_time=0.0):
    weights = (timesteps - min_time) / (max_time - min_time)
    weights = torch.clamp(weights, 0.0, 1.0)  # 限制权重在0和1之间
    return weights

class MVDreamPipeline(DiffusionPipeline):

    # _optional_components = ["feature_extractor", "image_encoder"]
    # _optional_components = ["image_encoder"]

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: MultiViewUNetModel,
        tokenizer: CLIPTokenizer,
        text_encoder: CLIPTextModel,
        scheduler: DDIMScheduler,
        # imagedream variant
        feature_extractor: CLIPImageProcessor,
        image_encoder: CLIPVisionModelWithProjection,
        requires_safety_checker: bool = False,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:  # type: ignore
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "  # type: ignore
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:  # type: ignore
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.register_to_config(requires_safety_checker=requires_safety_checker)

    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding.

        When this option is enabled, the VAE will split the input tensor in slices to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding.

        When this option is enabled, the VAE will split the input tensor into tiles to compute decoding and encoding in
        several steps. This is useful to save a large amount of memory and to allow the processing of larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously invoked, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def enable_sequential_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        Note that offloading happens on a submodule basis. Memory savings are higher than with
        `enable_model_cpu_offload`, but performance is lower.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.14.0"):
            from accelerate import cpu_offload
        else:
            raise ImportError(
                "`enable_sequential_cpu_offload` requires `accelerate v0.14.0` or higher"
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae]:
            cpu_offload(cpu_offloaded_model, device)

    def enable_model_cpu_offload(self, gpu_id=0):
        r"""
        Offloads all models to CPU using accelerate, reducing memory usage with a low impact on performance. Compared
        to `enable_sequential_cpu_offload`, this method moves one whole model at a time to the GPU when its `forward`
        method is called, and the model remains in GPU until the next model runs. Memory savings are lower than with
        `enable_sequential_cpu_offload`, but performance is much better due to the iterative execution of the `unet`.
        """
        if is_accelerate_available() and is_accelerate_version(">=", "0.17.0.dev0"):
            from accelerate import cpu_offload_with_hook
        else:
            raise ImportError(
                "`enable_model_offload` requires `accelerate v0.17.0` or higher."
            )

        device = torch.device(f"cuda:{gpu_id}")

        if self.device.type != "cpu":
            self.to("cpu", silence_dtype_warnings=True)
            torch.cuda.empty_cache()  # otherwise we don't see the memory savings (but they probably exist)

        hook = None
        for cpu_offloaded_model in [self.text_encoder, self.unet, self.vae]:
            _, hook = cpu_offload_with_hook(
                cpu_offloaded_model, device, prev_module_hook=hook
            )

        # We'll offload the last model manually.
        self.final_offload_hook = hook

    @property
    def _execution_device(self):
        r"""
        Returns the device on which the pipeline's models will be executed. After calling
        `pipeline.enable_sequential_cpu_offload()` the execution device can only be inferred from Accelerate's module
        hooks.
        """
        if not hasattr(self.unet, "_hf_hook"):
            return self.device
        for module in self.unet.modules():
            if (
                hasattr(module, "_hf_hook")
                and hasattr(module._hf_hook, "execution_device")
                and module._hf_hook.execution_device is not None
            ):
                return torch.device(module._hf_hook.execution_device)
        return self.device

    def _encode_prompt(
        self,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance: bool,
        negative_prompt=None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
             prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(
                f"`prompt` should be either a string or a list of strings, but got {type(prompt)}."
            )

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(
            prompt, padding="longest", return_tensors="pt"
        ).input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
            text_input_ids, untruncated_ids
        ):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1]
            )
            logger.warning(
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )

        if (
            hasattr(self.text_encoder.config, "use_attention_mask")
            and self.text_encoder.config.use_attention_mask
        ):
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device),
            attention_mask=attention_mask,
        )
        prompt_embeds = prompt_embeds[0]

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            bs_embed * num_images_per_prompt, seq_len, -1
        )

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            if (
                hasattr(self.text_encoder.config, "use_attention_mask")
                and self.text_encoder.config.use_attention_mask
            ):
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(
                dtype=self.text_encoder.dtype, device=device
            )

            negative_prompt_embeds = negative_prompt_embeds.repeat(
                1, num_images_per_prompt, 1
            )
            negative_prompt_embeds = negative_prompt_embeds.view(
                batch_size * num_images_per_prompt, seq_len, -1
            )

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        return image

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(
            inspect.signature(self.scheduler.step).parameters.keys()
        )
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(
                shape, generator=generator, device=device, dtype=dtype
            )
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def encode_image(self, image, device, num_images_per_prompt):
        # dtype = next(self.feature_extractor.parameters()).dtype
        dtype = self.text_encoder.dtype
        # if image.dtype == np.float32:
        #     image = (image * 255).astype(np.uint8)
        
        # image_clip = (((image + 1.0)/2.0)* 255)
        image_clip = (((image + 1.0)/2.0)* 1.0)
        image_clip = self.feature_extractor(image_clip, return_tensors="pt",do_rescale=False).pixel_values
        image_clip = image_clip.to(device=device, dtype=dtype)
        
        encoded_embeds = self.image_encoder(image_clip, output_hidden_states=False)
        image_embeds = encoded_embeds.image_embeds
        image_embeds = 0.2 * image_embeds.unsqueeze(-2).repeat(1, 77, 1).to(dtype)

        return torch.zeros_like(image_embeds), image_embeds

    def encode_image_latents(self, image, device, num_images_per_prompt):
        
        dtype = next(self.image_encoder.parameters()).dtype

        image = torch.from_numpy(image).unsqueeze(0).permute(0, 3, 1, 2).to(device=device) # [1, 3, H, W]
        image = 2 * image - 1
        image = F.interpolate(image, (256, 256), mode='bilinear', align_corners=False)
        image = image.to(dtype=dtype)

        posterior = self.vae.encode(image).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor # [B, C, H, W]
        latents = latents.repeat_interleave(num_images_per_prompt, dim=0)

        return torch.zeros_like(latents), latents

    def decode_latents_to_pil_save(self, latents, path):
        image_list = self.decode_latents(latents)
        image_list = self.numpy_to_pil(image_list)
        for i, image in enumerate(image_list):
            image.save(path+f'_{i}.jpg')

    
    def save_latents(self, noisy_latents_warp,latents,mix,name,t):
        image_warp = self.decode_latents(noisy_latents_warp)
        image_warp = self.numpy_to_pil(image_warp)
        image = self.decode_latents(latents)
        image = self.numpy_to_pil(image)
        image_mix = self.decode_latents(mix)
        image_mix = self.numpy_to_pil(image_mix)
        grid = np.concatenate(
            [
                np.concatenate([image_warp[0], image_warp[1], image_warp[2]], axis=0),
                np.concatenate([image_warp[3], image_warp[4], image_warp[5]], axis=0),
                np.concatenate([image[0], image[1], image[2]], axis=0),
                np.concatenate([image[3], image[4], image[5]], axis=0),
                np.concatenate([image_mix[0], image_mix[1], image_mix[2]], axis=0),
                np.concatenate([image_mix[3], image_mix[4], image_mix[5]], axis=0),
            ],
            axis=1,
        )
        kiui.write_image(f'/share/project/mabaorui/lucidDream/LucidDreamer-main/outputs/tt/warp/{t}_{name}.jpg', grid)
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: str = "",# 'a cute owl'
        image: Optional[np.ndarray] = None,# NOne
        masks: Optional[np.ndarray] = None,# NOne
        height: int = 256,# 256
        width: int = 256,# 256
        elevation: float = 0,# 0
        num_inference_steps: int = 50,# 30
        guidance_scale: float = 7.0,# 5
        negative_prompt: str = None,# ''
        num_images_per_prompt: int = 1,# 1
        eta: float = 0.0,# 0.0
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,# None
        output_type: Optional[str] = "numpy", # pil, numpy, latents | numpy
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        num_frames: int = 4, # 4
        device=torch.device("cuda:0"),
        guidance_rescale: float = 0.0,
        # gt_num: int=3,
        condition_num_frames: int=3,
        gt_frame: Optional[np.ndarray] = None,
    ):
        gt_num = condition_num_frames
        
        self.unet = self.unet.to(device=device)
        self.vae = self.vae.to(device=device)
        self.text_encoder = self.text_encoder.to(device=device)
        weight_dtype = next(self.vae.parameters()).dtype
        self.guidance_rescale = guidance_rescale

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        # do_classifier_free_guidance = guidance_scale > 1.0
        do_classifier_free_guidance = (guidance_scale != 0.0)
        
        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)# DDIMScheduler
        timesteps = self.scheduler.timesteps
        # import pdb;pdb.set_trace()
        if self.scheduler.timestep_scaling == "trailing":
            step_ratio = 999 // num_inference_steps
            timesteps = torch.from_numpy(np.round(np.arange(999, 0, -step_ratio)).astype(np.int64).copy()).to(timesteps.device)

        # imagedream variant
        if image is not None:
            image = rearrange(image, "b f c h w -> (b f) c h w", f=num_frames)
            img_latents = self.vae.encode(image.to(weight_dtype).to(device)).latent_dist.sample()
            img_latents = img_latents * self.vae.config.scaling_factor # [b*f, c, h, w] shape=[6, 4, 32, 32]
            # import pdb;pdb.set_trace()
            # img_latents[:gt_num] = img_latents[:gt_num] * self.vae.config.scaling_factor
        if gt_frame is not None:    
            image_gt = rearrange(gt_frame, "b f c h w -> (b f) c h w", f=num_frames)
            img_latents_gt = self.vae.encode(image_gt.to(weight_dtype).to(device)).latent_dist.sample()
            img_latents_gt = img_latents_gt * self.vae.config.scaling_factor # [b*f, c, h, w] shape=[6, 4, 32, 32]
        
        self.image_encoder = self.image_encoder.to(device=device)
        image_embeds_neg, image_embeds_pos = self.encode_image(image, device, num_images_per_prompt) 
        image_embeds_pos = image_embeds_pos[0:1]
        
        # import pdb;pdb.set_trace()
        if masks is not None:
            masks = rearrange(masks, "b f c h w -> (b f) c h w", f=num_frames)# [b*f, c, h, w]
            mask_latents = torch.nn.functional.interpolate(
                masks, 
                size=(
                    height // 8, 
                    width // 8
                )
            ).to(weight_dtype).to(device)# [b*f, c, h, w]
        
        if do_classifier_free_guidance:
            uncond_img_latents = torch.zeros_like(img_latents)
        
        prompt_embeds = self._encode_prompt(
            prompt=prompt,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=negative_prompt,
        )  # type: ignore torch.Size([1, 77, 1024])
        # prompt_embeds_neg, prompt_embeds_pos = _prompt_embeds.chunk(2)

        # Prepare latent variables
        actual_num_frames = num_frames
        noisy_latents = self.prepare_latents(
            actual_num_frames * num_images_per_prompt,
            4,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents=None,
        )# shape=[3, 4, 32, 32]
        
        
        
        noise_scheduler = self.scheduler
        if gt_frame is not None:
            noise_gt = torch.randn_like(img_latents)
            # t = timesteps[0]
            t = 999
            tt = torch.tensor([t] * actual_num_frames, dtype=img_latents.dtype, device=device)
            noisy_latents = noise_scheduler.add_noise(img_latents_gt, noise_gt, tt.long())
        

        latents = torch.cat([img_latents[:gt_num], noisy_latents[gt_num:]], dim=0)
        
        camera = None

        # Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # Denoising loop
        assert image.shape[0] == actual_num_frames, "Currently only supports batchsize==1"
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        time_ratio = 5
        timestep_warp = (timesteps[0]//time_ratio).long()
        print(timestep_warp)
        noise_warp = torch.randn_like(noisy_latents)
        tt = torch.tensor([timestep_warp] * actual_num_frames, dtype=noisy_latents.dtype, device=device)
        noisy_latents_warp = noise_scheduler.add_noise(img_latents, noise_warp, tt.long())
        
   
        
        mix_latents_warp = torch.cat([img_latents[:gt_num], noisy_latents_warp[gt_num:]], dim=0)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                multiplier = 2 if do_classifier_free_guidance else 1
                
                latents = torch.cat([img_latents[:gt_num], latents[gt_num:]], dim=0)# [6, 4, 32, 32]
                
                timestep_warp = (t//time_ratio).long()
                noise_warp = torch.randn_like(noisy_latents)
                tt = torch.tensor([timestep_warp] * actual_num_frames, dtype=noisy_latents.dtype, device=device)
                noisy_latents_warp = noise_scheduler.add_noise(img_latents, noise_warp, tt.long())
                weights = custom_decay_function_weight(tt.float())
                weights = weights.view(tt.shape[0], 1, 1, 1).to(noisy_latents_warp.dtype)
                noisy_latents_warp_weight = weights*noisy_latents_warp + (1. - weights) * latents
                mix_latents_warp = torch.cat([img_latents[:gt_num], noisy_latents_warp_weight[gt_num:]], dim=0)
                
                
                if self.unet.input_blocks[0].__dict__['_modules']['0'].__dict__['in_channels'] == 9:
                    latents = torch.cat([latents, mix_latents_warp, mask_latents], dim=1)# [b*f, c, h, w] shape=[6, 9, 32, 32]
                elif self.unet.input_blocks[0].__dict__['_modules']['0'].__dict__['in_channels'] == 5:
                    latents = torch.cat([latents, mask_latents], dim=1)# [b*f, c, h, w] shape=[6, 9, 32, 32]
                
                if do_classifier_free_guidance:
                    
                    unc_mix_latents_warp = torch.cat([img_latents[:gt_num], uncond_img_latents[gt_num:]], dim=0)
                    
                    
                    unc_mask_latents = torch.cat([mask_latents[:gt_num], torch.zeros_like(mask_latents[gt_num:])], dim=0)
                    uncond_latents = torch.cat([latents[:,:4], unc_mix_latents_warp, unc_mask_latents], dim=1)
                    latent_model_input = torch.cat([latents, uncond_latents], dim=0)# torch.Size([12, 9, 64, 64])
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                else:
                    latent_model_input = torch.cat([latents] * multiplier)
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                image_embeds_pos_input = image_embeds_pos.repeat(multiplier,1,1)
                unet_inputs = {
                    'x': latent_model_input,# torch.Size([num_frames, 5, 32, 32])
                    'timesteps': torch.tensor([t] * actual_num_frames * multiplier, dtype=latent_model_input.dtype, device=device),# torch.Size([num_frames])
                    'context': (prompt_embeds + image_embeds_pos_input).repeat(actual_num_frames,1,1),
                    'num_frames': actual_num_frames,# 4
                    'camera': None,# 
                }

                
                noise_pred = self.unet.forward(**unet_inputs)# [6, 4, 32, 32]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                    noise_pred = (1 + guidance_scale) * noise_pred_cond - guidance_scale * noise_pred_uncond
                    
                    
                if do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    
                    noise_pred[:,:4] = rescale_noise_cfg(noise_pred[:,:4], noise_pred_cond[:,:4], guidance_rescale=self.guidance_rescale)

  
                latents: torch.Tensor = self.scheduler.step(
                    noise_pred[:,:4], t, latents[:, :4], **extra_step_kwargs, return_dict=False
                )[0]
                
                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)  # type: ignore

        # Post-processing
        if output_type == "latent":
            image = latents
            noisy_latents_warp = noisy_latents_warp
        elif output_type == "pil":
            image = self.decode_latents(latents)
            image = self.numpy_to_pil(image)
            image_warp = self.decode_latents(noisy_latents_warp)
            image_warp = self.numpy_to_pil(image_warp)
        else: # numpy
            image = self.decode_latents(latents)
            image_warp = self.decode_latents(noisy_latents_warp)

        # Offload last model to CPU
        if hasattr(self, "final_offload_hook") and self.final_offload_hook is not None:
            self.final_offload_hook.offload()

        return image, image_warp