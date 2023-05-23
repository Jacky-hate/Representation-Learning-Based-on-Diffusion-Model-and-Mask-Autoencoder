from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from .vision_transformer import VisionTransformer,Mlp
from einops import rearrange
from .diffusion import diffusionModel
from tqdm import tqdm
from .utils import pixelshuffle, pixelshuffle_invert, module_no_grade

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss
    
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class VisionTransformerFordmim(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forward(self, x, mask):
        x = self.patch_embed(x)

        assert mask is not None
        B, L, _ = x.shape

        mask_token = self.mask_token.expand(B, L, -1)
        w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        return x
        
class VisionTransformerFordmimWithLDM(VisionTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.num_classes == 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self._trunc_normal_(self.mask_token, std=.02)

    def _trunc_normal_(self, tensor, mean=0., std=1.):
        trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)

    def forwardx(self, x):
        B, L, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)
        x = self.norm(x)

        x = x[:, 1:]
        B, L, C = x.shape
        H = W = int(L ** 0.5)
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, H, W)
        return x
    
    def forward(self, x, mask=None):
        x = self.patch_embed(x)

        if mask is not None:
            B, L, _ = x.shape

            mask_token = self.mask_token.expand(B, L, -1)
            w = mask.flatten(1).unsqueeze(-1).type_as(mask_token)
            
            x = x * (1 - w) + mask_token * w
        
        #x: batch_size * embedding d * patch_n * patch_n
        z = self.forwardx(x)
        
        return z
        
class dmim(nn.Module):
    def __init__(self, encoder, encoder_stride):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=self.encoder.num_features,
                out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
            nn.PixelShuffle(self.encoder_stride),
        )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size
    
    @torch.no_grad()
    def mask_x(self, x, mask):
        scale = self.encoder.patch_size
        x = pixelshuffle_invert(x, scale)
        w = mask.unsqueeze(1).type_as(x)
        x = x * (1 - w) + 0 * w
        x = pixelshuffle(x, scale)
        return x
        
    @torch.no_grad()
    def sample(self, x, mask, timesteps=None):
        
        z = self.encoder(x, mask)
        B, C, H, W = z.shape
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)
        x_masked = self.mask_x(x, mask)
        return x_masked, x_rec, []

    def forward(self, x, mask):
        z = self.encoder(x, mask)
        x_rec = self.decoder(z)

        mask = mask.repeat_interleave(self.patch_size, 1).repeat_interleave(self.patch_size, 2).unsqueeze(1).contiguous()
        loss_recon = F.l1_loss(x, x_rec, reduction='none')
        loss = (loss_recon * mask).sum() / (mask.sum() + 1e-5) / self.in_chans
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}
    
class dmimWithLDM(nn.Module):
    def __init__(
            self, 
            encoder, 
            encoder_stride,
            diffusion,
            use_contrastive: bool=False, 
            noise_ratio: float=0.1
        ):
        super().__init__()
        self.encoder = encoder
        self.encoder_stride = encoder_stride

        # self.decoder = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=self.encoder.num_features,
        #         out_channels=self.encoder_stride ** 2 * 3, kernel_size=1),
        #     nn.PixelShuffle(self.encoder_stride),
        # )

        self.in_chans = self.encoder.in_chans
        self.patch_size = self.encoder.patch_size
        self.use_contrastive = use_contrastive
        self.diffusion=diffusion
        if use_contrastive:
            self.header = Mlp(self.encoder.num_features)
            self.noise_ratio = noise_ratio
            self.T = 1.0
    
    @torch.no_grad()
    def sample(self, x, mask, timesteps=None):
        
        z = self.encoder(x, mask)
        B, C, H, W = z.shape
        x_recon, intermediates = self.diffusion.sample(cond=z, batch_size=B,
               verbose=True, timesteps=timesteps)
        
        x_masked = self.mask_x(x, mask)
        return x_masked, x_recon, intermediates
    
    @torch.no_grad()
    def sample_uncon(self, shape):
        
        x, intermediates = self.diffusion.sample_uncon(shape)
        return x, intermediates
    
    @torch.no_grad()
    def mask_x(self, x, mask):
        scale = self.encoder.patch_size
        x = pixelshuffle_invert(x, scale)
        w = mask.unsqueeze(1).type_as(x)
        x = x * (1 - w) + 0 * w
        x = pixelshuffle(x, scale)
        return x
    
    @torch.no_grad()
    def predict_denoised(self, x, mask):
              
        z = self.encoder(x, mask)
        
        B, C, H, W = z.shape

        x_recon = self.diffusion.predict_denoised(cond=z)
        
        x_recons=[]
        for _ in tqdm(range(40),desc='random denoising'):
            x_recons.append(self.diffusion.predict_denoised(cond=z))
            
        x_masked = self.mask_x(x, mask)
        
        return x_masked, x_recon, x_recons
    
    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)
    
    def forward(self, x, mask, augmented_pair):
        
        loss_contrastive = 0.0
        
        pixel_mask = self.mask_x(torch.ones_like(x), mask) == 0
        z = self.encoder(x, mask)
        
        if self.use_contrastive:
            
            # only pos pair and add noise on z
            # aug = torch.cat(augmented_pair, dim=0)
            # z_masked_aug = self.encoder(aug, None)
            # normed_1, normed_2 = [nn.functional.normalize(header(x), dim=1) for header, x in zip(self.header_pair,z_masked_aug.flatten(2).mean(2).chunk(2, dim=0))]
            # loss_contrastive = (1 - (normed_1 * normed_2).sum(dim=-1)).mean() / 2.0
            # z = z * (1 - self.noise_ratio) + torch.randn_like(z) * self.noise_ratio
            
            # pos and neg pair
            aug_1, aug_2 = augmented_pair
            z_1 = self.header(self.encoder(aug_1, None).flatten(2).mean(2))
            with torch.no_grad():
                z_2 = self.header(self.encoder(aug_2, None).flatten(2).mean(2))
            loss_contrastive = self.contrastive_loss(z_1, z_2) / 20.0
            
        loss_recon = self.diffusion(x, cond=z, mask=pixel_mask)
        loss = loss_recon  + loss_contrastive
        
        return loss

    @torch.jit.ignore
    def no_weight_decay(self):
        if hasattr(self.encoder, 'no_weight_decay'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay()}
        return {}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        if hasattr(self.encoder, 'no_weight_decay_keywords'):
            return {'encoder.' + i for i in self.encoder.no_weight_decay_keywords()}
        return {}


def build_dmim(config):
    model_type = config.MODEL.TYPE
    if model_type == 'vit':
        encoder = VisionTransformerFordmim(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
        encoder_stride = 16
        model = dmim(encoder=encoder, encoder_stride=encoder_stride)
    elif model_type == 'vit_ldm':
        encoder = VisionTransformerFordmimWithLDM(
            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.VIT.PATCH_SIZE,
            in_chans=config.MODEL.VIT.IN_CHANS,
            num_classes=0,
            embed_dim=config.MODEL.VIT.EMBED_DIM,
            depth=config.MODEL.VIT.DEPTH,
            num_heads=config.MODEL.VIT.NUM_HEADS,
            mlp_ratio=config.MODEL.VIT.MLP_RATIO,
            qkv_bias=config.MODEL.VIT.QKV_BIAS,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            init_values=config.MODEL.VIT.INIT_VALUES,
            use_abs_pos_emb=config.MODEL.VIT.USE_APE,
            use_rel_pos_bias=config.MODEL.VIT.USE_RPB,
            use_shared_rel_pos_bias=config.MODEL.VIT.USE_SHARED_RPB,
            use_mean_pooling=config.MODEL.VIT.USE_MEAN_POOLING)
        encoder_stride = 16
        diffusion=diffusionModel(
            img_size=config.DATA.IMG_SIZE,
            scale=config.MODEL.DIFFUSION.SCALE,
            in_channels=config.MODEL.VIT.IN_CHANS,
            model_channels=config.MODEL.DIFFUSION.MODEL_CHANNELS,
            out_channels=config.MODEL.VIT.IN_CHANS,  #in equals out
            num_res_blocks=config.MODEL.DIFFUSION.NUM_RES_BLOCKS,
            attention_resolutions=config.MODEL.DIFFUSION.ATTENTION_RESOLUTIONS,
            dropout=config.MODEL.DIFFUSION.DROPOUT,
            channel_mult=config.MODEL.DIFFUSION.CHANNEL_MULT,
            conv_resample=config.MODEL.DIFFUSION.CONV_RESAMPLE,
            dims=config.MODEL.DIFFUSION.DIMS,
            num_classes=config.MODEL.DIFFUSION.NUM_CLASSES,
            use_checkpoint=config.MODEL.DIFFUSION.USE_CHECKPOINT,
            use_fp16=config.MODEL.DIFFUSION.USE_FP16,
            num_heads=config.MODEL.DIFFUSION.NUM_HEADS,
            num_head_channels=config.MODEL.DIFFUSION.NUM_HEAD_CHANNELS,
            num_heads_upsample=config.MODEL.DIFFUSION.NUM_HEADS_UPSAMPLE,
            use_scale_shift_norm=config.MODEL.DIFFUSION.USE_SCALE_SHIFT_NORM,
            resblock_updown=config.MODEL.DIFFUSION.RESBLOCK_UPDOWN,
            use_new_attention_order=config.MODEL.DIFFUSION.USE_NEW_ATTENTION_ORDER,
            use_spatial_transformer=config.MODEL.DIFFUSION.USE_SPATIAL_TRANSFORMER,
            transformer_depth=config.MODEL.DIFFUSION.TRANSFORMER_DEPTH,
            context_dim=config.MODEL.VIT.IN_CHANS,
            n_embed=config.MODEL.DIFFUSION.N_EMBED,
            legacy=config.MODEL.DIFFUSION.LEGACY,
            given_betas=config.MODEL.DIFFUSION.GIVEN_BETAS,
            beta_schedule=config.MODEL.DIFFUSION.BETA_SCHEDULE,
            timesteps=config.MODEL.DIFFUSION.TIMESTEPS,
            linear_start=config.MODEL.DIFFUSION.LINEAR_START,
            linear_end=config.MODEL.DIFFUSION.LINEAR_END,
            cosine_s=config.MODEL.DIFFUSION.COSINE_S,
            masked_loss_only=config.MODEL.DIFFUSION.MASKED_LOSS_ONLY,
            log_every_t=config.MODEL.DIFFUSION.LOG_EVERY_T,
            SpatialTransformercheckpoint=config.MODEL.DIFFUSION.SPATIAL_TRANSFORMER_CHECKPOINT,
        )
        model = dmimWithLDM(encoder=encoder, encoder_stride=encoder_stride, diffusion=diffusion, use_contrastive=config.MODEL.USE_CONTRASTIVE)
    else:
        raise NotImplementedError(f"Unknown pre-train model: {model_type}")

    return model
