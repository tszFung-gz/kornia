"""In this module several equalization methods are exposed: he, ahe, clahe."""

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from kornia.utils.helpers import _torch_histc_cast
from kornia.utils.image import perform_keep_shape_image
from kornia.color import rgb_to_lab, lab_to_rgb

from .histogram import histogram


def white_balance(imgs: torch.Tensor, grid_size: Tuple[int, int],
                 even_tile_size: bool = False ) -> torch.Tensor:
    """_summary_

    Args:
        imgs (torch.Tensor): _description_
        grid_size (Tuple[int, int]): _description_
        even_tile_size (bool, optional): _description_. Defaults to False.

    Raises:
        TypeError: _description_
        ValueError: _description_

    Returns:
        torch.Tensor: _description_
    """
    if not isinstance(imgs, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(imgs)}")      
    if len(imgs.shape) < 3 or imgs.shape[-3] != 3:         
        raise ValueError(f"Input size must have a shape of (*, 3, H, W). Got {imgs.shape}")
    result = rgb_to_lab(imgs)
    for i in range(result.shape[0]):
        avg_a = torch.mean(result[i, :, :, 1])
        avg_b = torch.mean(result[i, :, :, 2])
        result[i, :, :, 1] = result[i, :, :, 1] - ((avg_a - 128) * (result[i,:, :, 0] / 255.0) * 1.1)
        result[i,:, :, 2] = result[i,:, :, 2] - ((avg_b - 128) * (result[i,:, :, 0] / 255.0) * 1.1)
        result[i,:, :, :] = lab_to_rgb(result[i,:, :, :])
    return result

