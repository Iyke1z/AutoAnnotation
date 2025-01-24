import os
from typing import Dict, List, Any

import numpy as np
import torch
from crp.image import imgify
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from zennit.image import gridify
from zennit.image import imsave as imsave_zennit

def save_img(data: torch.Tensor, name: str, dataset_name: str, other_dir: str = None, norm: bool = False):
    """
    Plot and save explanation heat map.
    Args:
        explanation: gradient/attribution tensor
        name: name of file
        dataset_name: dataset name

    """
    data = np.array(data.detach().cpu())

    # normalization of data
    if norm:
        max_val = np.abs(data).max((1, 2), keepdims=True)
        data = data / 2 / (max_val + 1e-12) + 0.5
    grid = gridify(data, fill_value=0.5)

    if other_dir is None:
        dir = "images"
    else:
        dir = other_dir

    os.makedirs(f"results/{dir}/{dataset_name}", exist_ok=True)
    imsave_zennit(f"results/{dir}/{dataset_name}/{name}.png",
                  grid,
                  vmin=0.,
                  vmax=1.,
                  level=1.0,
                  cmap='bwr')


def plot_grid(ref_c: Dict[int, List[torch.Tensor]], cmap="bwr", vmin=None, vmax=None, symmetric=False,
              resize=(224, 224), padding=True, figsize=None, dpi=100) -> None:
    nrows = len(ref_c)
    ncols = len(next(iter(ref_c.values())))

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, dpi=dpi)
    plt.subplots_adjust(wspace=0.05)

    for r, key in enumerate(ref_c):

        for c, img in enumerate(ref_c[key]):

            img = imgify(img, cmap=cmap, vmin=vmin, vmax=vmax, symmetric=symmetric, resize=resize, padding=padding)

            if nrows == 1:
                ax = axs[c]
            elif ncols == 1:
                ax = axs[r]
            else:
                ax = axs[r, c]

            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(key)


def gauss_p_norm(x: Any, sigma: int = 6) -> Any:
    """ Applies Gaussian filter and normalizes"""
    return normalize(gaussian_filter(x, sigma=sigma))


def normalize(a: Any) -> Any:
    """ Applies normalization"""
    return a / a.max()


def mask_img(img: torch.Tensor, mask: torch.Tensor, alpha: int = 0.5) -> torch.Tensor:
    """ Masks input sample with mask"""
    minv = 1 - mask  # inverse mask
    return img * mask + img * minv * alpha


def get_masked(imgs: Dict, hms: Dict, thresh=0.2):
    """ Masks img dict from CRP library using heatmaps dict."""
    return {k: [mask_img(img.to(hm), gauss_p_norm(hm) > thresh) for img, hm in zip(imgs[k], hms[k])] for k in
            imgs.keys()}

def get_masks(hms: Dict, thresh=0.2):
    """ Masks img dict from CRP library using heatmaps dict."""
    return {k: [gauss_p_norm(hm) > thresh for hm in zip(hms[k])] for k in
            hms.keys()}

def slice_img(img: Any, pad_x: int, pad_y: int) -> Any:
    if pad_x == 0 and pad_y == 0:
        return img
    if len(img.shape) == 3:
        return img[:, pad_y:-pad_y] if pad_y else img[..., pad_x:-pad_x]
    elif len(img.shape) == 2:
        return img[pad_y:-pad_y] if pad_y else img[..., pad_x:-pad_x]


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False, dpi=200)
    for i, img in enumerate(imgs):
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
