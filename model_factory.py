

import torch.nn as nn
from utils import unfold_StackOverChannel, fold_tensor

from models.FH_Mamba import FH_Mamba


class IceNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        if configs.model == "FH_Mamba":
            self.base_net = FH_Mamba(
                configs.input_length,
                configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                configs.img_size,
                configs.patch_size,
                configs.hid_S,
                configs.hid_T_channels,
                configs.N_S,
                configs.drop,
                configs.drop_path,
            )
        else:
            raise ValueError("错误的网络名称，不存在%s这个网络" % configs.model)
        self.patch_size = configs.patch_size
        self.img_size = configs.img_size

    def forward(self, inputs, targets):
        outputs, loss = self.base_net(
            unfold_StackOverChannel(inputs, self.patch_size),
            unfold_StackOverChannel(targets, self.patch_size),
        )

        outputs = fold_tensor(outputs, self.img_size, self.patch_size)

        return outputs, loss
