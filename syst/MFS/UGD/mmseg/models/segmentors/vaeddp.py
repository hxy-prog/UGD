import torch
import torch.nn as nn
from mmseg.core import add_prefix
from mmseg.ops import resize
from torch.special import expm1
from einops import rearrange, reduce, repeat
from mmcv.cnn import ConvModule
import math
import torch.nn.functional as F
import numpy as np
from ..vae.model import Encoder as VAEencoder
from ..vae.model import Decoder as VAEdecoder
np.set_printoptions(threshold=np.inf)
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
# torch.set_printoptions(threshold=float('inf'), linewidth=200)

def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    # not sure if this accounts for beta being clipped to 0.999 in discrete version
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1, eps=1e-5)


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered
    
    
from mmcv.runner import BaseModule

class DDPVAEencoder(BaseModule):
    def __init__(self):
        super(DDPVAEencoder, self).__init__()
        self.encoder = VAEencoder()
        self.visit=0
    def init_weights(self):
        ckpt=torch.load('/home/hexingyang/DDP/segmentation/mmseg/models/vae/vae_parameters.pth',map_location='cpu')
        newcheck = {}
        for k, v in ckpt.items():
            if not k.startswith('embedding'):
                if not k.startswith('decoder'):
                    newcheck[k] = v
        self.load_state_dict(newcheck,strict=True)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()        
    def forward(self,x):
        if self.visit==0:
            self.eval()
            self.visit+=1
        return self.encoder(x)
class DDPVAEdecoder(BaseModule):
    def __init__(self):
        # 定义解码器部分
        super(DDPVAEdecoder, self).__init__()
        self.decoder = VAEdecoder()
        self.visit=0
    def init_weights(self):
        ckpt=torch.load('mmseg/models/vae/vae_parameters.pth',map_location='cpu')
        newcheck = {}
        for k, v in ckpt.items():
            if not k.startswith('embedding'):
                if k.startswith('decoder'):
                    newcheck[k] = v
        self.load_state_dict(newcheck,strict=True)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()     
    def forward(self,x):
        if self.visit==0:
            self.eval()
            self.visit+=1
        return self.decoder(x)
decoder=DDPVAEdecoder()
decoder.init_weights()



class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor, leaky_slope=0.01):
        super(UpsampleBlock, self).__init__()
        # 转置卷积上采样
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels,
            kernel_size=scale_factor*2,
            stride=scale_factor,
            padding=scale_factor//2,
            output_padding=scale_factor%2
        )
        # 调整通道数并进一步细化特征
        self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        # 使用层归一化
        self.bn = nn.BatchNorm2d(out_channels)
        # 使用 LeakyReLU 代替 ReLU
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_slope, inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x) 
        x = self.leaky_relu(x)
        return x

class refine(nn.Module):
    def __init__(self):
        super(refine, self).__init__()
        # 从 4x32x32 上采样到 64x64x64
        self.up1 = UpsampleBlock(4, 64, scale_factor=2)
        # 从 64x64x64 上采样到 128x128x128
        self.up2 = UpsampleBlock(64, 256, scale_factor=2)

    def forward(self, x):
        x = self.up1(x)  # 上采样到 64x64x64
        x = self.up2(x)  # 上采样到 128x128x128
        return x


@SEGMENTORS.register_module()
class VAEDDP(EncoderDecoder):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 bit_scale=0.1,
                 timesteps=1,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 noise_schedule='linear',
                 diffusion='ddim',
                 accumulation=False,
                 **kwargs):
        super(VAEDDP, self).__init__(**kwargs)

        self.bit_scale = bit_scale
        self.timesteps = timesteps
        self.randsteps = randsteps
        self.diffusion = diffusion
        self.time_difference = time_difference
        self.sample_range = sample_range
        self.use_gt = False
        self.accumulation = accumulation
        self.embedding_table =DDPVAEencoder()
        
        self.refine =refine()
        
        print(f" timesteps: {timesteps},"
              f" randsteps: {randsteps},"
              f" sample_range: {sample_range},"
              f" diffusion: {diffusion}")

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.transform = ConvModule(
            self.decode_head.in_channels[0] * 2,
            self.decode_head.in_channels[0],
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        )

        # time embeddings
        time_dim = self.decode_head.in_channels[0] * 4  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,]
            sinu_pos_emb,  # [2, 17]
            nn.Linear(fourier_dim, time_dim),  # [2, 1024]
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [2, 1024]
        )

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)[0]
        
        if self.diffusion == "ddim":
            out = self.ddim_sample(x, img_metas)
        elif self.diffusion == 'ddpm':
            out = self.ddpm_sample(x, img_metas)
        else:
            raise NotImplementedError
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # backbone & neck

        # np.savetxt('img.txt',img[0][0].cpu().numpy())
        # exit(0)
        
        x = self.extract_feat(img)[0]  # bs, 256, h/4, w/4
        batch, c, h, w, device, = *x.shape, x.device
        
        # print( batch, c, h, w, device)
        # print('bbbbbbbbbbbbbbbbbbbbb')
        # print(gt_semantic_seg.size())
        
        gt_down = resize(gt_semantic_seg.float(), size=(h, w), mode="nearest")
        gt_down = gt_down.to(gt_semantic_seg.dtype)
        
        
        # print(gt_down)
        # numpy_array = gt_down.cpu().numpy()[0][0]
        # # 将numpy数组写入到txt文件
        # with open('tensor2.txt', 'a') as f:
        #     for row in numpy_array:
        #         np.savetxt(f, row[None], fmt='%f')
        gt_down[gt_down == 255] = self.num_classes
        
        # print(gt_down)
        
        # print(gt_down)
        # numpy_array = gt_down.cpu().numpy()[0][0]
        # # # 将numpy数组写入到txt文件
        # with open('tensor2.txt', 'a') as f:
        #     for row in numpy_array:
        #         np.savetxt(f, row[None], fmt='%f')
        # exit(0)
        # print(gt_semantic_seg)
        # print('aaaaaaaaaaaaaaaaaaaaaaaaaw')
        # print(gt_semantic_seg)
        
        # mask = gt_semantic_seg < 0

        # # 使用掩码来获取所有负数
        # negative_numbers = gt_semantic_seg[mask]

        # print("负数包括:", negative_numbers)
        # exit(0)

        gt_down = self.embedding_table(gt_down)
        
        with torch.no_grad():
            import cv2
            decoder.to(x.device)
            outtt=decoder(gt_down)
            outtt=torch.softmax(outtt,dim=1)
            outtt=torch.argmax(outtt, dim=1)
            cv2.imwrite('ddimvis/be.jpg',outtt[0].cpu().numpy()*255)
        # gt_down = (torch.sigmoid(gt_down) * 2 - 1) * self.bit_scale


        # sample time
        times = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0],
                                                                      self.sample_range[1])  # [bs]

        # random noise
        noise = torch.randn_like(gt_down)

        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * gt_down + sigma * noise
        
        with torch.no_grad():
            import cv2
            decoder.to(x.device)
            outtt=decoder(noised_gt)
            outtt=torch.softmax(outtt,dim=1)
            outtt=torch.argmax(outtt, dim=1)
            cv2.imwrite('ddimvis/aft.jpg',outtt[0].cpu().numpy()*255)
        
        # print('2222')
        # print(noised_gt.size())
        noised_gt_refine=self.refine(noised_gt)
        
        # conditional input
        feat = torch.cat([x, noised_gt_refine], dim=1)
        feat = self.transform(feat)

        losses = dict()
        input_times = self.time_mlp(noise_level)
        with torch.no_grad():
            pred_noise=self._decode_head_forward_test([feat], input_times, img_metas=img_metas)
            np.savetxt('ddimvis/noisepred.txt',pred_noise[0][0].cpu().numpy())
            np.savetxt('ddimvis/noisetrue.txt',noise[0][0].cpu().numpy())
            
            
            gt_pred=(noised_gt-pred_noise*sigma.clamp(min=1e-8))/alpha
            outtt=decoder(gt_pred)
            outtt=torch.softmax(outtt,dim=1)
            outtt=torch.argmax(outtt, dim=1)
            cv2.imwrite('ddimvis/pred.jpg',outtt[0].cpu().numpy()*255)
            
        with torch.no_grad():
            outtt=decoder((noised_gt-noise*sigma.clamp(min=1e-8))/alpha)
            outtt=torch.softmax(outtt,dim=1)
            outtt=torch.argmax(outtt, dim=1)
            cv2.imwrite('ddimvis/gt.jpg',outtt[0].cpu().numpy()*255)
        
        loss_decode = self._decode_head_forward_train([feat], input_times, img_metas, noise)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                [x], img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses

    def _decode_head_forward_train(self, x, t, img_metas, noise):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train_vae(x, t, img_metas,
                                                     noise,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, t, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, t, img_metas, self.test_cfg)
        return seg_logits

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times

    @torch.no_grad()
    def ddim_sample(self, x, img_metas):
        decoder.to(x.device)
        b, c, h, w, device = *x.shape, x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)
        x = repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps,4, 32, 32), device=device)
        outs = list()
        for idx, (times_now, times_next) in enumerate(time_pairs):
            
            import cv2
            outtt=decoder(mask_t)
            outtt=torch.softmax(outtt,dim=1)
            outtt=torch.argmax(outtt, dim=1)
            cv2.imwrite('ddimvis/'+str(idx)+'.jpg',outtt[0].cpu().numpy()*255)
            np.savetxt('ddimvis/'+str(idx)+'.txt',mask_t[0][0].cpu().numpy())
            
            feat_mask_t=self.refine(mask_t)

            feat_mask_t = nn.functional.interpolate(feat_mask_t, size=(h,w), mode='bilinear', align_corners=False)
            
            feat = torch.cat([x, feat_mask_t], dim=1)
            feat = self.transform(feat)
            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(feat_mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(feat_mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            pred_noise = self._decode_head_forward_test([feat], input_times, img_metas=img_metas)  # [bs, 150, ]
          
            
            # np.savetxt('ddimvis/noise'+str(idx)+'.txt',pred_noise[0][0].cpu().numpy())
            
            print(alpha, sigma,alpha_next, sigma_next)
            gt_pred=(mask_t-pred_noise*sigma.clamp(min=1e-8))/alpha
            
            
            outtt=decoder(gt_pred)
            outtt=torch.softmax(outtt,dim=1)
            outtt=torch.argmax(outtt, dim=1)
            cv2.imwrite('ddimvis/'+str(idx)+'pred.jpg',outtt[0].cpu().numpy()*255)
            
            np.savetxt('ddimvis/pre.txt',pred_noise[0][0].cpu().numpy())
            np.savetxt('ddimvis/m.txt',mask_t[0][0].cpu().numpy())
            np.savetxt('ddimvis/gt.txt',gt_pred[0][0].cpu().numpy())
            mask_t = gt_pred * alpha_next + pred_noise * sigma_next
            np.savetxt('ddimvis/maf.txt',mask_t[0][0].cpu().numpy())
            
            # mask_t=(torch.sigmoid(mask_t) * 2 - 1) * self.bit_scale
            
        #     if self.accumulation:
        #         outs.append(mask_logit.softmax(1))
        # if self.accumulation:
        #     mask_logit = torch.cat(outs, dim=0)
        # logit = mask_logit.mean(dim=0, keepdim=True)
        return  decoder(mask_t)

    @torch.no_grad()
    def ddpm_sample(self, x, img_metas):
        b, c, h, w, device = *x.shape, x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)

        x = repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, self.decode_head.in_channels[0], h, w), device=device)
        outs = list()
        for times_now, times_next in time_pairs:
            feat = torch.cat([x, mask_t], dim=1)
            feat = self.transform(feat)

            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            mask_logit = self._decode_head_forward_test([feat], input_times, img_metas=img_metas)  # [bs, 150, ]
            mask_pred = torch.argmax(mask_logit, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (mask_t * (1 - c) / alpha + c * mask_pred)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)
            noise = torch.where(
                rearrange(times_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(mask_t),
                torch.zeros_like(mask_t)
            )
            mask_t = mean + (0.5 * log_variance).exp() * noise

            if self.accumulation:
                outs.append(mask_logit.softmax(1))
        if self.accumulation:
            mask_logit = torch.cat(outs, dim=0)
        logit = mask_logit.mean(dim=0, keepdim=True)
        return logit
