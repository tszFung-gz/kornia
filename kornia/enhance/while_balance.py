"""In this module several equalization methods are exposed: he, ahe, clahe."""

import torch

from kornia.color import rgb_to_lab, lab_to_rgb


def white_balance_gwa(imgs: torch.Tensor) -> torch.Tensor:
    """_summary_
        gray_world_assumption
    Args:
        imgs (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    result = rgb_to_lab(imgs)
    avg_a = torch.mean(result[:, 1, :, :])
    avg_b = torch.mean(result[:, 2, :, :])
    result[:, 1, :, :] = result[:, 1, :, :] - (avg_a * (result[:, 1, :, :] / 100.0) * 1.1)
    result[:, 2, :, :] = result[:, 2, :, :] - (avg_b * (result[:, 2, :, :] / 100.0) * 1.1)
    result[:, :, :, :] = lab_to_rgb(result[:, :, :, :])
    return result

def white_balance_pra(imgs: torch.Tensor) -> torch.Tensor:
    """perfect_reflector_assumption

    Args:
        imgs (torch.Tensor): _description_

    Returns:
        torch.Tensor: _description_
    """
    r, g, b = torch.split(imgs, split_size_or_sections=1, dim=1)
    MaxVal = torch.max(torch.max(torch.maximum(torch.maximum(r, g), b), dim=2)[0], dim=2)[0]
    ratio = 0.05
    ratio = imgs.shape[2] * imgs.shape[3] * ratio


def white_balance_dynamic(imgs: torch.Tensor):
    """dynamic_threshold

    Args:
        imgs (torch.Tensor): _description_
    """
    print(imgs.shape)



def white_balance(imgs: torch.Tensor) -> torch.Tensor:
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
    r, g, b = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    # 求各個通道所佔增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r = kornia.enhance.add_weighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = kornia.enhance.add_weighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = kornia.enhance.add_weighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([b, g, r])
    return balance_img

    return result
