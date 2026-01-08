import random
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    from torch.distributions.poisson import Poisson
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "torch is required to run data_maker scripts. Install PyTorch in your current environment first."
    ) from e


def crop_to_bounding_box(image, offset_height, offset_width, target_height,
                         target_width, is_batch):
    # BHWC -> BHWC
    cropped = image[:, offset_height: offset_height + target_height, offset_width: offset_width + target_width, :]

    if not is_batch:
        cropped = cropped[0]

    return cropped

def crop_to_bounding_box_list(image, offset_height, offset_width, target_height,
                         target_width):
    # HWC
    cropped = [_image[offset_height: offset_height + target_height, offset_width: offset_width + target_width, :] for _image in image]

    return cropped

def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                        target_width, is_batch):
    _,height,width,_ = image.shape
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height

    paddings = (0, 0, offset_width, after_padding_width, offset_height, after_padding_height, 0, 0)
    
    padded = torch.nn.functional.pad(image, paddings)
    if not is_batch:
      padded = padded[0]

    return padded

def resize_with_crop_or_pad_torch(image, target_height, target_width):
    # BHWC -> BHWC
    
    is_batch = True
    if image.ndim == 3:
        is_batch = False
        image = image[None]   # 1HWC

    def max_(x, y):
        return max(x, y)

    def min_(x, y):
        return min(x, y)

    def equal_(x, y):
        return x == y

    _, height, width, _ = image.shape
    width_diff = target_width - width
    offset_crop_width = max_(-width_diff // 2, 0)
    offset_pad_width = max_(width_diff // 2, 0)

    height_diff = target_height - height
    offset_crop_height = max_(-height_diff // 2, 0)
    offset_pad_height = max_(height_diff // 2, 0)

    # Maybe crop if needed.
    cropped = crop_to_bounding_box(image, offset_crop_height, offset_crop_width,
                                    min_(target_height, height),
                                    min_(target_width, width), is_batch)

    # Maybe pad if needed.
    if not is_batch and cropped.ndim == 3:
        cropped = cropped[None]
    resized = pad_to_bounding_box(cropped, offset_pad_height, offset_pad_width,
                                    target_height, target_width, is_batch)

    return resized


def ensure_psf_npy(psf_path: Path, *, n_psf: int = 81, k: int = 135, seed: int = 42):
    psf_path = Path(psf_path)
    if psf_path.is_file():
        return

    psf_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    grid = np.arange(k, dtype=np.float32) - (k - 1) / 2.0
    xx, yy = np.meshgrid(grid, grid, indexing="xy")
    psf = np.zeros((n_psf, k, k, 3), dtype=np.float32)

    for i in range(n_psf):
        sigma = float(rng.uniform(6.0, 18.0))
        x0 = float(rng.uniform(-8.0, 8.0))
        y0 = float(rng.uniform(-8.0, 8.0))
        g = np.exp(-(((xx - x0) ** 2 + (yy - y0) ** 2) / (2.0 * sigma * sigma)))
        s = float(g.sum())
        if s > 0:
            g = g / s
        psf[i, :, :, 0] = g
        psf[i, :, :, 1] = g
        psf[i, :, :, 2] = g

    np.save(str(psf_path), psf)



def psf2otf(psf, h=None, w=None, permute=False):
    '''
    psf = (b) h,w,c
    '''
    if h is not None:
        psf = resize_with_crop_or_pad_torch(psf, h, w)
    if permute:
        if psf.ndim == 3:
            psf = psf.permute(2,0,1)    # HWC -> CHW
        else:
            psf = psf.permute(0,3,1,2)    # HWC -> CHW 
    psf = psf.to(torch.complex64)
    psf = torch.fft.fftshift(psf, dim=(-1,-2))
    otf = torch.fft.fft2(psf)
    return otf

def fft(img):   # CHW
    img = img.to(torch.complex64)
    Fimg = torch.fft.fft2(img)
    return Fimg

def ifft(Fimg):
    img = torch.abs(torch.fft.ifft2(Fimg)).to(torch.float32)
    return img


def create_contrast_mask(image):
    return 1 - torch.mean(image, dim=(-1,-2), keepdim=True)  # (B), C,1,1

def apply_tikhonov(lr_img, psf, K, norm=True, otf=None):
    h,w = lr_img.shape[-2:]
    if otf is None:
        psf_norm = resize_with_crop_or_pad_torch(psf, h, w)
        if norm:
            psf_norm = psf_norm / psf_norm.sum((0, 1))
        otf = psf2otf(psf_norm, h, w, permute=True)

    otf = otf[:,None,...]   # B,1,C,H,W
    contrast_mask = create_contrast_mask(lr_img)[:,None,...]  # B,1,C,1,1
    K_adjusted = K * contrast_mask  # B,M,C,1,1
    tikhonov_filter = torch.conj(otf) / (torch.abs(otf) ** 2 + K_adjusted)  # B,M,C,H,W
    lr_fft = fft(lr_img)[:,None,...]    # B,1,C,H,W
    deconvolved_fft = lr_fft * tikhonov_filter
    deconvolved_image = torch.fft.ifft2(deconvolved_fft).real
    deconvolved_image = torch.clamp(deconvolved_image, min=0.0, max=1.0)

    return deconvolved_image    # B,M,C,H,W
    

def add_noise_all_new(image, poss=4e-5, gaus=1e-5):
    p = Poisson(image / poss)
    sampled = p.sample((1,))[0]
    poss_img = sampled * poss
    gauss_noise = torch.randn_like(image) * gaus
    noised_img = poss_img + gauss_noise

    noised_img = torch.clamp(noised_img, 0.0, 1.0)

    return noised_img


def apply_convolution(image, psf, pad):
    '''
    input: hr img (b,c,h,w, [0,1])
    output: noised lr img (b,c,h+P,w+P, [0,1])
    '''

    # metalens simulation
    image = F.pad(image, (pad, pad, pad, pad))
    h,w = image.shape[-2:]
    psf_norm = resize_with_crop_or_pad_torch(psf, h, w)
    otf = psf2otf(psf_norm, h, w, permute=True)
    lr_img = fft(image) * otf
    lr_img = torch.clamp(ifft(lr_img), min=1e-20, max=1.0)

    # noise addition
    noised_img = add_noise_all_new(lr_img)

    return noised_img, otf

def apply_conv_n_deconv(image, otf, padding, M, psize, ks=None, ph=135, num_psf=9, sensor_h=1215, crop=True, conv=True):
    '''
    input:  hr img (b,c,h,w)
    otf: 1,N,C,H,W
    output: noised lr img (N,c,h,w)
    '''

    b,_,_,_ = image.shape
    if conv:
        img_patch = F.unfold(image, kernel_size=ph*3, stride=ph).view(b,3,ph*3,ph*3,num_psf**2).permute(0,4,1,2,3).contiguous() # B,N,C,H,W

        # metalens simulation
        lr_img = fft(img_patch) * otf
        lr_img = torch.clamp(ifft(lr_img), min=1e-20, max=1.0)

        # noise addtion
        lr_img = add_noise_all_new(lr_img)

    else:   # load convolved image for validation
        b = 1
        lr_img = image

    # apply deconvolution
    if ks is not None:   
        lr_img = apply_tikhonov(lr_img, None, ks, otf=otf) # B,M,N,C,405,405
        lr_img = lr_img[..., ph:-ph, ph:-ph] # BMNCHW
        lr_img = lr_img.view(b, M, num_psf, num_psf, 3, ph, ph).permute(0,1,4,2,5,3,6).reshape(b,M,3,sensor_h,sensor_h)
    else:
        lr_img = lr_img[..., ph:-ph, ph:-ph] # BNCHW
        lr_img = lr_img.view(b, num_psf, num_psf, 3, ph, ph).permute(0,3,1,4,2,5).reshape(b,3,sensor_h,sensor_h)

    lq_patches = []
    gt_patches = []
    for i in range(b):
        cur = lr_img[i] # (M),C,H,W
        cur_gt = image[i]

        # remove padding for lq and gt
        pt,pb,pl,pr = padding[i]
        if pb and pt:
            cur = cur[...,pt: -pb, :]
            cur_gt = cur_gt[...,pt+ph: -(pb+ph), ph:-ph]  
        elif pl and pr:
            cur = cur[...,pl:-pr]
            cur_gt = cur_gt[...,ph:-ph, pl+ph: -(pr+ph)]
        else:
            cur_gt = cur_gt[...,ph:-ph, ph: -ph]
        h,w = cur.shape[-2:]

        # randomly crop patch for training
        if crop:    # train
            top = random.randint(0, h - psize)
            left = random.randint(0, w - psize)
            lq_patches.append(cur[..., top:top + psize, left:left + psize])
            gt_patches.append(cur_gt[..., top:top + psize, left:left + psize])
    if crop:    # training
        lq_patches = torch.stack(lq_patches)
        gt_patches = torch.stack(gt_patches)
    else:   # validation
        return cur, cur_gt

    return lq_patches, gt_patches # B,(M),C,H,W


def apply_convolution_square_val(image, otf, padding, M, psize, ks=None, ph=135, num_psf=9, sensor_h=1215, crop=False):
    '''
    merge to above one.
    image = lr_image
    '''
    lr_img = image
    b = 1
    if M:   # apply deconvolution
        lr_img = apply_tikhonov(lr_img, None, ks, otf=otf) # B,M,N,C,H,W
        lr_img = lr_img[..., ph:-ph, ph:-ph] # B,M,N,C,H,W
        lr_img = lr_img.view(b, M, num_psf, num_psf, 3, ph, ph).permute(0,1,4,2,5,3,6).reshape(b,M,3,sensor_h,sensor_h)
    else:
        lr_img = lr_img[..., ph:-ph, ph:-ph] # B,N,C,H,W
        lr_img = lr_img.view(b, num_psf, num_psf, 3, ph, ph).permute(0,3,1,4,2,5).reshape(b,3,sensor_h,sensor_h)
    

    for i in range(b):
        cur = lr_img[i] # (M),C,H,W

        # remove padding for lq and gt
        pt,pb,pl,pr = padding[i]
        if pb and pt: 
            cur = cur[...,pt: -pb, :]
        elif pl and pr:
            cur = cur[...,pl:-pr]

    return cur 