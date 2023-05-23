# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from collections import OrderedDict
from easydict import EasyDict as edict
from .utils import default, extract_into_tensor, make_beta_schedule, pixelshuffle, pixelshuffle_invert, noise_like
import numpy as np
from functools import partial
from .openaimodel import UNetModel
from tqdm import tqdm

class diffusionModel(nn.Module):

    def __init__(
            self,
            img_size,
            scale,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            num_classes=None,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=True,    # custom transformer support
            transformer_depth=1,              # custom transformer support
            context_dim=None,                 # custom transformer support
            n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=True,
            given_betas=None, 
            beta_schedule="linear", 
            timesteps=1000,
            linear_start=1e-4, 
            linear_end=2e-2, 
            cosine_s=8e-3,
            masked_loss_only = False,
            v_posterior=0.0,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
            log_every_t=10,
            loss_type = 'l1',
            original_elbo_weight=0.,
            l_simple_weight=1.,
            parameterization = 'x0',
            SpatialTransformercheckpoint=True,
        ):
        super().__init__()
        
        self.parameterization = parameterization
        self.masked_loss_only = masked_loss_only
        
        assert in_channels == out_channels and in_channels == context_dim
        self.in_channels = in_channels
        scaled_size = img_size // scale
        assert scaled_size * scale == img_size
        self.scale = scale
        self.unet_channel = in_channels * np.square(self.scale)
        
        self.clip_denoised = False
        
        self.log_every_t = log_every_t
        
        self.loss_type = loss_type
        
        self.UNet = UNetModel(
            self.unet_channel,
            model_channels,
            self.unet_channel,
            num_res_blocks,
            attention_resolutions,
            dropout,
            channel_mult,
            conv_resample,
            dims,
            num_classes,
            use_checkpoint,
            use_fp16,
            num_heads,
            num_head_channels,
            num_heads_upsample,
            use_scale_shift_norm,
            resblock_updown,
            use_new_attention_order,
            use_spatial_transformer,    # custom transformer support
            transformer_depth,              # custom transformer support
            self.unet_channel,                 # custom transformer support
            n_embed,                     # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy,
            SpatialTransformercheckpoint=SpatialTransformercheckpoint
        )
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.mean = torch.tensor(IMAGENET_DEFAULT_MEAN).cuda(non_blocking=True).unsqueeze(1).unsqueeze(2)
        self.std = torch.tensor(IMAGENET_DEFAULT_STD).cuda(non_blocking=True).unsqueeze(1).unsqueeze(2)        
        
        
      

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3, min_snr_gamma=5):
        if given_betas is not None:
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)
        #to_torch = partial(torch.tensor, dtype=torch.float16)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        
        # if self.parameterization == "eps":
        #     lvlb_weights = self.betas ** 2 / (
        #                 2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        # elif self.parameterization == "x0":
        #     lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        # else:
        #     raise NotImplementedError("mu not supported")
        
        # # TODO how to choose this term
        # lvlb_weights[0] = lvlb_weights[1]
        
        
        
        # Another implementation: https://arxiv.org/abs/2303.09556
        lvlb_weights = self.alphas_cumprod / (1 - self.alphas_cumprod) 
        clipped_lvlb_weights = lvlb_weights.clone()
        clipped_lvlb_weights.clamp_(max = min_snr_gamma)
        lvlb_weights = clipped_lvlb_weights / lvlb_weights
        
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        
        assert not torch.isnan(self.lvlb_weights).all()
    
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)
        
    def normalize(self, x):
        return x.mul(self.std).add(self.mean) * 2. - 1.
    
    def unnormalize(self, x):
        return ((x + 1.) / 2.).sub(self.mean).div(self.std)

    def apply_model(self, x_noisy, t, cond):
        return self.UNet(x_noisy, t, cond)
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def p_mean_variance(self, x, c, t):
        t_in = t
        model_out = self.apply_model(x, t_in, c)
        
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError("mu not supported")
        

        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
            
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        
        return model_mean, posterior_variance, posterior_log_variance
        
        
    @torch.no_grad()
    def p_sample(self, x, c, t, repeat_noise=False,
                 temperature=1.):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t)
        model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, cond, shape, verbose=True, timesteps=None, log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        
        if timesteps is None:
            timesteps = self.num_timesteps

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))  

        img = torch.randn(shape, device=device)
            
        intermediates = [img]

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, cond, ts)
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)

        return img, intermediates


    @torch.no_grad()
    def sample(self, cond, batch_size=12, verbose=True, timesteps=None,**kwargs):
        
        cond = self.shuffle_cond(cond)
        shape = cond.shape
        cond = rearrange(cond, 'b c h w -> b (h w) c')

        
        x_recon, _intermediates = self.p_sample_loop(cond,
                                  shape, verbose=verbose, timesteps=timesteps)
        
        x_recon = self.unnormalize(pixelshuffle(x_recon,self.scale))

        intermediates = [self.unnormalize(pixelshuffle(x,self.scale)) for x in _intermediates]
        
        return x_recon, intermediates
    
    @torch.no_grad()
    def sample_uncon(self, shape):
        
        B, iC, iH, iW = shape
        pH = pW = self.scale
        oC, oH, oW = iC*(pH*pW), iH//pH, iW//pW
        
        assert oC * oH * oW == iC * iH * iW, "shape must be divisible by scale"
        
        shape = (B, oC, oH, oW)
        
        x, _intermediates = self.p_sample_loop(cond=None,
                                  shape=shape, verbose=True, timesteps=None)
        
        x = self.unnormalize(pixelshuffle(x,self.scale))

        intermediates = [self.unnormalize(pixelshuffle(x,self.scale)) for x in _intermediates]
        
        return x, intermediates

    @torch.no_grad()
    def predict_denoised(self, cond, **kwargs):    
        
        B,C,H,W = cond.shape
        
        cond = self.shuffle_cond(cond)
        x_t = torch.randn_like(cond)
        cond = rearrange(cond, 'b c h w -> b (h w) c')

        ts = torch.full((B,), self.num_timesteps-1, device = self.betas.device, dtype=torch.long)
    
        model_out = self.apply_model(x_t, ts, cond)
        
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x_t, t=ts, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError("mu not supported")
        x_recon = self.unnormalize(pixelshuffle(x_recon,self.scale))
        
        return x_recon
    
    def shuffle_cond(self, cond):
        B, C, H, W = cond.shape
        scale = int(np.sqrt(C // self.in_channels))
        assert self.in_channels * np.square(scale) == C
        return pixelshuffle_invert(pixelshuffle(cond, scale), self.scale)
        
    def get_loss(self, pred, target):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss
    
    def predict():
        pass
        
    def forward(self, x, cond=None, noise=None, mask=None, return_recon=False): 
    
        x = self.normalize(x)
        
        # unshuffle (B, C, H*t, W*t) => (B, C*t*t, H, W)
        x_start = pixelshuffle_invert(x, self.scale)
        if cond is not None:
            
            cond = self.shuffle_cond(cond)
            
            assert x_start.shape == cond.shape,"Unexpected shape of x_start or cond."
            
            # rearrange condition to 2 dims 
            cond = rearrange(cond, 'b c h w -> b (h w) c')
        
        # sampling t
        time_steps = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device='cuda').long()
        
        # sampling noise
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        # calculate x_t
        x_t = self.q_sample(x_start=x_start, t=time_steps, noise=noise)

        # predict noise
        model_out = self.apply_model(x_t, time_steps, cond)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x_t, t=time_steps, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError("obj not supported")
        
        if return_recon:
            return  model_out
        
        assert mask is not None, "Mask is None"
        
        mask = mask if self.masked_loss_only else torch.ones(size=mask.shape,device=mask.device)
        
        loss = (self.get_loss(x_recon, x_start) * pixelshuffle_invert(mask, self.scale)).mean(dim=[1, 2, 3])
        loss_simple = self.l_simple_weight * loss.mean()
        loss_vlb = self.original_elbo_weight * (self.lvlb_weights[time_steps] * loss).mean()
        loss = loss_simple + loss_vlb
        
        return  loss
