

import numpy as np
import torch

Max_SIE = 25889  # 通过data/MAX_SIE.py计算得到

def mse_func(pred, true, mask):
    mse = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
        mask
    )
    return mse.mean().item()


def rmse_func(pred, true, mask):
    mse = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
        mask
    )
    rmse = torch.sqrt(mse)
    return rmse.mean().item()


def mae_func(pred, true, mask):
    mae = torch.abs(pred - true) * mask
    mae = torch.sum(mae, dim=[2, 3, 4]).mean(dim=1) / torch.sum(mask)
    return mae.mean().item()


def nse_func(pred, true, mask):
    squared_error = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]).mean(dim=1)
    mean_observation = torch.sum(true * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
        mask
    )
    mean_observation = (
        mean_observation.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    )
    squared_deviation = torch.sum(
        (true - mean_observation) ** 2 * mask, dim=[2, 3, 4]
    ).mean(dim=1)
    nse = 1 - squared_error / squared_deviation
    return nse.mean().item()


def PSNR(pred, true, mask):
    mse = torch.sum((pred - true) ** 2 * mask, dim=[2, 3, 4]).mean(dim=1) / torch.sum(
        mask
    )
    psnr = 10 * np.log10(1 * 1 / mse.mean().item())
    return psnr

def BACC_func(pred, true, mask):
    # 使用布尔索引将大于0.15的位置设置为1，其他地方设置为0
    pred[pred > 0.15] = 1
    pred[pred <= 0.15] = 0
    true[true > 0.15] = 1
    true[true <= 0.15] = 0

    # a = torch.sum(torch.abs(pred - true) * mask, dim=[2, 3, 4])
    # BACC = 1 - a / Max_SIE
    # print(BACC)

    # IIEE = torch.sum(torch.abs(pred - true) * mask, dim=[2, 3, 4]).mean(dim=1)
    IIEE = torch.sum(torch.abs(pred - true) * mask, dim=[2, 3, 4])
    BACC = 1 - IIEE.mean() / Max_SIE
    # BACC = 1 - IIEE.mean().item() / Max_SIE
    return BACC


def BACC_pic_func(pred, true, mask):
    # 二值化处理，将大于0.15的位置设为1，其余为0
    pred = (pred > 0.15).float()
    true = (true > 0.15).float()

    # 调整mask的形状以匹配pred和true的维度（B, C, H, W）
    mask = mask.unsqueeze(0).unsqueeze(0)  # 扩展为(1, 1, H, W)，便于广播

    # 计算误差绝对值并与mask相乘，随后在C, H, W维度求和
    IIEE = torch.sum(torch.abs(pred - true) * mask, dim=[1, 2, 3])

    # 计算BACC，IIEE均值除以Max_SIE后从1中减去
    BACC = 1 - IIEE.mean() / Max_SIE
    return BACC

