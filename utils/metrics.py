"""
Fusion quality evaluation metrics for TIFDFusion.
Standard IVIF metrics matching CDDFuse/FreeFusion/SFMFusion conventions.
All computations on [0, 255] float64 grayscale images.
"""

import torch
import numpy as np
import math
from scipy.signal import convolve2d
from skimage.metrics import structural_similarity as skimage_ssim
import sklearn.metrics as skm


# ========================= Preprocessing =========================

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _to_gray_255(img):
    """Convert to 2D grayscale numpy array in [0, 255] float64.
    Handles Tanh [-1,1], ToTensor [0,1], and uint8 [0,255]."""
    img = _to_numpy(img)
    if img.ndim == 4:
        img = img[0]
    if img.ndim == 3 and img.shape[0] == 3:
        img = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
    elif img.ndim == 3 and img.shape[0] == 1:
        img = img[0]
    img = img.astype(np.float64)
    if img.min() < -0.5:  # Tanh output [-1, 1]
        img = (img + 1.0) / 2.0 * 255.0
    elif img.max() <= 1.0:  # ToTensor [0, 1]
        img = img * 255.0
    img = np.clip(img, 0.0, 255.0)
    return img


# ========================= Metrics =========================

def EN(img):
    """Information Entropy. Input: 2D [0,255] float."""
    a = np.uint8(np.round(img)).flatten()
    h = np.bincount(a, minlength=256).astype(np.float64)
    h = h / h.sum()
    h = h[h > 0]
    return -np.sum(h * np.log2(h))


def SD(img):
    """Standard Deviation. Input: 2D [0,255] float."""
    return np.std(img)


def SF(img):
    """Spatial Frequency. Input: 2D [0,255] float."""
    return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) +
                   np.mean((img[1:, :] - img[:-1, :]) ** 2))


def AG(img):
    """Average Gradient. Input: 2D [0,255] float."""
    Gx, Gy = np.zeros_like(img), np.zeros_like(img)
    Gx[:, 0] = img[:, 1] - img[:, 0]
    Gx[:, -1] = img[:, -1] - img[:, -2]
    Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2
    Gy[0, :] = img[1, :] - img[0, :]
    Gy[-1, :] = img[-1, :] - img[-2, :]
    Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
    return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))


def MI(image_F, image_A, image_B):
    """Mutual Information using sklearn (natural log, nats). Same as CDDFuse."""
    F_int = np.uint8(np.round(image_F)).flatten()
    A_int = np.uint8(np.round(image_A)).flatten()
    B_int = np.uint8(np.round(image_B)).flatten()
    return (skm.mutual_info_score(F_int, A_int) +
            skm.mutual_info_score(F_int, B_int))


def SCD(image_F, image_A, image_B):
    """Sum of Correlations of Differences."""
    imgF_A = image_F - image_A
    imgF_B = image_F - image_B
    corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
        np.sum((image_A - np.mean(image_A)) ** 2) * np.sum((imgF_B - np.mean(imgF_B)) ** 2) + 1e-10)
    corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
        np.sum((image_B - np.mean(image_B)) ** 2) * np.sum((imgF_A - np.mean(imgF_A)) ** 2) + 1e-10)
    return corr1 + corr2


def VIFF(image_F, image_A, image_B):
    """Multi-scale VIF. Same as CDDFuse/FreeFusion."""
    return _compare_viff(image_A, image_F) + _compare_viff(image_B, image_F)


def _compare_viff(ref, dist):
    sigma_nsq = 2
    eps = 1e-10
    num = 0.0
    den = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2
        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12
        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0
        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0
        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den if den != 0 else 1.0
    return 1.0 if np.isnan(vifp) else vifp


def _Qabf_getArray(img):
    """Sobel gradients for Qabf. Exact CDDFuse implementation."""
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
    SAx = convolve2d(img, h3, mode='same')
    SAy = convolve2d(img, h1, mode='same')
    gA = np.sqrt(SAx * SAx + SAy * SAy)
    aA = np.zeros_like(img)
    aA[SAx == 0] = math.pi / 2
    aA[SAx != 0] = np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
    return gA, aA


def _Qabf_getQabf(aA, gA, aF, gF):
    """Edge preservation quality. Exact CDDFuse parameters."""
    Tg = 0.9994; kg = -15; Dg = 0.5
    Ta = 0.9879; ka = -22; Da = 0.8
    GAF = np.zeros_like(aA)
    GAF[gA > gF] = gF[gA > gF] / gA[gA > gF]
    GAF[gA == gF] = gF[gA == gF]
    GAF[gA < gF] = gA[gA < gF] / gF[gA < gF]
    AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
    QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
    return QgAF * QaAF


def Qabf(image_F, image_A, image_B):
    """Qabf: Xydeas-Petrovic. Exact CDDFuse implementation."""
    gA, aA = _Qabf_getArray(image_A)
    gB, aB = _Qabf_getArray(image_B)
    gF, aF = _Qabf_getArray(image_F)
    QAF = _Qabf_getQabf(aA, gA, aF, gF)
    QBF = _Qabf_getQabf(aB, gB, aF, gF)
    deno = np.sum(gA + gB)
    nume = np.sum(QAF * gA + QBF * gB)
    return nume / deno if deno != 0 else 0


def SSIM(image_F, image_A, image_B):
    """SSIM using skimage. data_range=255. Same as SFMFusion."""
    return (skimage_ssim(image_F, image_A, data_range=255.0) +
            skimage_ssim(image_F, image_B, data_range=255.0))


# ========================= Main API =========================

def compute_fusion_metrics(fused, vis, ir):
    """
    Compute all standard IVIF fusion metrics.
    Accepts tensors or numpy arrays in any range.
    Auto-converts to [0,255] float64 grayscale.
    Returns dict matching CDDFuse/FreeFusion conventions.
    """
    f = _to_gray_255(fused)
    v = _to_gray_255(vis)
    i = _to_gray_255(ir)

    metrics = {}
    metrics['EN'] = EN(f)
    metrics['SD'] = SD(f)
    metrics['SF'] = SF(f)
    metrics['AG'] = AG(f)
    metrics['MI'] = MI(f, v, i)
    metrics['SCD'] = SCD(f, v, i)
    metrics['VIF'] = VIFF(f, v, i)
    metrics['Qabf'] = Qabf(f, v, i)
    metrics['SSIM'] = SSIM(f, v, i)
    return metrics
