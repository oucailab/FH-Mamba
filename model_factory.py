

import torch.nn as nn
from utils import unfold_StackOverChannel, fold_tensor

from models.FH_Mamba import FH_Mamba


class IceNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        if configs.model == "FH_Mamba":
            self.base_net = FH_Mamba(
                t=configs.input_length,
                c=configs.input_dim * configs.patch_size[0] * configs.patch_size[1],
                img_size=configs.img_size,
                patch_size=configs.patch_size,
                hid_s=configs.hid_S,
                hid_t=configs.hid_T_channels,
                n_s=configs.N_S,
                n_t=getattr(configs, "N_T", 4),
                spatio_kernel_enc=getattr(configs, "spatio_kernel_enc", 3),
                spatio_kernel_dec=getattr(configs, "spatio_kernel_dec", 3),
                act_inplace=getattr(configs, "act_inplace", False),
                mlp_ratio=getattr(configs, "mlp_ratio", 4.0),
                drop=configs.drop,
                drop_path=configs.drop_path,
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
