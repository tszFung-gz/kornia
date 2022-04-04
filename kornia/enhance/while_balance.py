"""In this module several equalization methods are exposed: he, ahe, clahe."""

from typing import Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms

from kornia.utils.helpers import _torch_histc_cast
from kornia.utils.image import perform_keep_shape_image
from kornia.color import rgb_to_lab, lab_to_rgb
from PIL import Image


def gray_world_assumption(imgs: torch.Tensor) -> torch.Tensor:
    result = rgb_to_lab(imgs)
    avg_a = torch.mean(result[:, 1, :, :])
    avg_b = torch.mean(result[:, 2, :, :])
    result[:, 1, :, :] = result[:, 1, :, :] - (avg_a * (result[:, 1, :, :] / 100.0) * 1.1)
    result[:, 2, :, :] = result[:, 2, :, :] - (avg_b * (result[:, 2, :, :] / 100.0) * 1.1)
    result[:, :, :, :] = lab_to_rgb(result[:, :, :, :])
    return result

def perfect_reflector_assumption(imgs: torch.Tensor) -> torch.Tensor:
    r, g, b = torch.split(imgs, split_size_or_sections=1, dim=1)
    MaxVal = torch.max(torch.max(torch.maximum(torch.maximum(r, g), b), dim=2)[0], dim=2)[0]
    ratio = 0.05
    ratio = imgs.shape[2] * imgs.shape[3] * ratio


def dynamic_threshold(imgs: torch.Tensor):
    print(imgs.shape)



def white_balance(imgs: torch.Tensor, gray_world = False, perfect_reflect = False) -> torch.Tensor:
    """_summary_

    Args:
        imgs (torch.Tensor): _description_

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
    if gray_world:
        result = gray_world_assumption(imgs)
    elif perfect_reflect:
        result = perfect_reflector_assumption(imgs)

    return result


if __name__ == '__main__':
    # img = Image.open().convert('RGB')
    # # img.show()
    # tf1 = transforms.Compose([transforms.ToTensor()])
    # toImg = transforms.ToPILImage()
    # img = tf1(img).unsqueeze(0)
    img = torch.rand((16,3,256,256))
    output = white_balance(img, perfect_reflect=True)
    output = toImg(output[0])
    output.show()
    # print(output.shape)
