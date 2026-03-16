import torch
from torch import nn

from .icemamba import *
from .HilbertScan3DMambaBlock_HSA import *


class BasicConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        upsampling=False,
        act_norm=False,
        act_inplace=True,
        num_groups=8,
    ):
        super(BasicConv2d, self).__init__()
        self.act_norm = act_norm
        if upsampling is True:
            self.conv = nn.Sequential(
                *[
                    nn.Conv2d(
                        in_channels,
                        out_channels * 4,
                        kernel_size=kernel_size,
                        padding=(kernel_size - 1) // 2,
                    ),
                    nn.PixelShuffle(2),
                ]
            )
        else:
            self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) // 2,
            )
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

    def forward(self, x):
        y = self.conv(x)
        if self.act_norm:
            y = self.norm(y)
            y = self.act(y)
        return y


class DoubleBranchConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        downsampling=False,
    ):
        super(DoubleBranchConv, self).__init__()

        stride = 2 if downsampling is True else 1

        self.branch1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.branch2_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

        self.branch2_dwconv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            stride=1,
            groups=out_channels,
        )

        self.final_block = nn.Sequential(
            nn.Conv2d(
                out_channels * 2,
                out_channels,
                kernel_size=3,
                padding=1,
                stride=stride,
            ),
            nn.InstanceNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        out1 = self.branch1(x)

        out2_temp = self.branch2_conv1(x)
        out2 = self.branch2_dwconv(out2_temp)

        out = torch.cat([out1, out2], dim=1)

        out = self.final_block(out)

        return out


def sampling_generator(n, reverse=False):
    samplings = [False, True] * (n // 2)
    if reverse:
        return list(reversed(samplings[:n]))
    return samplings[:n]


class Encoder(nn.Module):
    def __init__(self, c_in, c_hid, n_s, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(n_s)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            DoubleBranchConv(
                c_in,
                c_hid,
                downsampling=samplings[0],
            ),
            *[
                DoubleBranchConv(
                    c_hid,
                    c_hid,
                    downsampling=s,
                )
                for s in samplings[1:]
            ]
        )

    def forward(self, x):
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, c_hid, c_out, n_s, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(n_s, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[
                BasicConv2d(
                    c_hid,
                    c_hid,
                    spatio_kernel,
                    upsampling=s,
                    act_inplace=act_inplace,
                )
                for s in samplings[:-1]
            ],
            BasicConv2d(
                c_hid,
                c_hid,
                spatio_kernel,
                upsampling=samplings[-1],
                act_inplace=act_inplace,
            ),
        )
        self.readout = nn.Conv2d(c_hid, c_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        y = self.dec[-1](hid + enc1)
        y = self.readout(y)
        return y


class MetaBlock(nn.Module):
    def __init__(
        self,
        t,
        in_channels,
        input_resolution=None,
        drop=0.0,
        drop_path=0.0,
        d_state=16,
    ):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = t * in_channels

        self.hi_mamba = HilbertScan3DMambaBlock(
            dim=in_channels,
            hid_dim=int(in_channels / 2),
            patch_size=(1, 9, 9),
            mlp_ratio=1.2,
            expand=1,
            d_state=d_state,
            drop=drop,
            drop_path=drop_path,
        )

        self.wt = WaveletTransform2D()
        self.iwt = WaveletTransform2D(inverse=True)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hid_channels * 4,
                out_channels=self.hid_channels * 4,
                kernel_size=3,
                padding=1,
                groups=self.hid_channels * 4,
            ),
            nn.InstanceNorm2d(self.hid_channels * 4),
            nn.GELU(),
            nn.Conv2d(
                in_channels=self.hid_channels * 4,
                out_channels=self.hid_channels * 4,
                kernel_size=1,
            ),
            nn.BatchNorm2d(self.hid_channels * 4),
            nn.GELU(),
        )

        self.conv_freq_x = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hid_channels * 4,
                out_channels=self.hid_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(self.hid_channels),
            nn.GELU(),
        )

        self.fusion = MultiScaleAttentionFusion(
            in_channels=self.hid_channels,
            input_resolution_hw=input_resolution,
        )

    def forward(self, x):
        b, t, c, h, w = x.shape

        wavelet = x.reshape(b, t * c, h, w)

        x_orig = wavelet
        ll, lh, hl, hh = self.wt(wavelet)
        y = torch.concat([ll, lh, hl, hh], dim=1)

        freqency_x = self.conv_freq_x(y)
        freqency_x = nn.functional.interpolate(
            freqency_x,
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        y = self.conv(y)
        ll, lh, hl, hh = torch.chunk(y, chunks=4, dim=1)
        wavelet_out = self.iwt([ll, lh, hl, hh])

        freqency_x = freqency_x.reshape(b, t, c, h, w)
        mamba_out = self.hi_mamba(x, freqency_x)

        mamba_out = mamba_out.reshape(b, t * c, h, w)

        out = self.fusion(mamba_out, wavelet_out)
        out = out + x_orig
        out = out.reshape(b, t, c, h, w)
        return out


class MidMetaNet(nn.Module):
    def __init__(
        self,
        t,
        channel_in,
        n2,
        input_resolution=None,
        drop=0.0,
        drop_path=0.1,
    ):
        super(MidMetaNet, self).__init__()
        assert n2 >= 2
        self.n2 = n2
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.n2)]
        enc_layers = [
            MetaBlock(
                t,
                channel_in,
                input_resolution,
                drop,
                drop_path=dpr[0],
                d_state=16,
            )
        ]
        for i in range(1, n2 - 1):
            enc_layers.append(
                MetaBlock(
                    t,
                    channel_in,
                    input_resolution,
                    drop,
                    drop_path=dpr[i],
                )
            )
        enc_layers.append(
            MetaBlock(
                t,
                channel_in,
                input_resolution,
                drop,
                drop_path=drop_path,
            )
        )
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        original = x
        z = x
        for i in range(self.n2):
            z = self.enc[i](z)
        y = z + original
        return y


class FH_Mamba(nn.Module):
    def __init__(
        self,
        t,
        c,
        img_size,
        patch_size,
        hid_s,
        hid_t=16,
        n_s=4,
        n_t=4,
        spatio_kernel_enc=3,
        spatio_kernel_dec=3,
        act_inplace=False,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
    ):
        self.hid_t = hid_t
        super(FH_Mamba, self).__init__()

        num_downsampling_ops = n_s // 2
        h_bottleneck = img_size[0] // (patch_size[0] * (2 ** num_downsampling_ops))
        w_bottleneck = img_size[1] // (patch_size[1] * (2 ** num_downsampling_ops))

        self.hid_s = hid_s
        self.t = t

        self.enc = Encoder(
            c,
            hid_s,
            n_s,
            spatio_kernel_enc,
            act_inplace=act_inplace,
        )
        self.dec = Decoder(
            hid_s,
            c,
            n_s,
            spatio_kernel_dec,
            act_inplace=act_inplace,
        )

        self.reshape_i = nn.Conv3d(in_channels=hid_s, out_channels=hid_t, kernel_size=1)
        self.reshape_o = nn.Conv3d(in_channels=hid_t, out_channels=hid_s, kernel_size=1)
        self.hid = nn.Sequential(
            MetaBlock(
                t,
                hid_t,
                (h_bottleneck, w_bottleneck),
                drop,
                drop_path=drop_path,
            ),
            MetaBlock(
                t,
                hid_t,
                (h_bottleneck, w_bottleneck),
                drop,
                drop_path=drop_path,
            ),
            MetaBlock(
                t,
                hid_t,
                (h_bottleneck, w_bottleneck),
                drop,
                drop_path=drop_path,
            ),
        )

        self.criterion = nn.HuberLoss()

    def forward(self, input_x, targets):
        t_seg = self.t

        b, _, c_frame, h_img, w_img = input_x.shape
        recursion = 1

        full_prediction = torch.zeros_like(targets).to(input_x.device)

        current_input_segment = input_x[:, :t_seg, ...]

        total_loss = 0.0
        num_loss_terms = 0

        for r_idx in range(recursion):
            x_for_enc = current_input_segment.reshape(b * t_seg, c_frame, h_img, w_img)
            embed, skip = self.enc(x_for_enc)

            _, c_bottle, h_bottle, w_bottle = embed.shape

            z_for_hid = embed.view(b, c_bottle, t_seg, h_bottle, w_bottle)
            z_for_hid = self.reshape_i(z_for_hid)
            z_for_hid = z_for_hid.reshape(b, t_seg, self.hid_t, h_bottle, w_bottle)

            fc_time_seq = self.hid(z_for_hid)

            fc_time_seq = fc_time_seq.reshape(b, self.hid_t, t_seg, h_bottle, w_bottle)
            fc_time_seq = self.reshape_o(fc_time_seq)
            fc_flat = fc_time_seq.reshape(b * t_seg, c_bottle, h_bottle, w_bottle)

            next_frames_segment = self.dec(fc_flat, skip)
            next_frames_segment = torch.clamp(next_frames_segment, 0, 1)
            next_frames_segment = next_frames_segment.reshape(
                b, t_seg, c_frame, h_img, w_img
            )

            start_idx = r_idx * t_seg
            end_idx = (r_idx + 1) * t_seg
            full_prediction[:, start_idx:end_idx, ...] = next_frames_segment

            current_target_segment = targets[:, start_idx:end_idx, ...]
            loss_segment = self.criterion(next_frames_segment, current_target_segment)
            total_loss += loss_segment
            num_loss_terms += 1

            if r_idx < recursion - 1:
                current_input_segment = next_frames_segment

        avg_loss = (
            total_loss / num_loss_terms
            if num_loss_terms > 0
            else torch.tensor(0.0).to(input_x.device)
        )

        return full_prediction, avg_loss
