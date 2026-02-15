import torch
from torch import nn
from .HilbertScan3DMambaBlock_HSA import *
import pywt
import torch.nn.functional as F

class WaveletTransform2D(nn.Module):
    def __init__(self, inverse=False, wavelet="haar", mode="constant"):
        super(WaveletTransform2D, self).__init__()
        self.mode = mode
        wavelet = pywt.Wavelet(wavelet)

        if isinstance(wavelet, tuple):
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet
        else:
            dec_lo, dec_hi, rec_lo, rec_hi = wavelet.filter_bank

        self.inverse = inverse
        if inverse is False:
            dec_lo = torch.tensor(dec_lo).flip(-1).unsqueeze(0)
            dec_hi = torch.tensor(dec_hi).flip(-1).unsqueeze(0)
            self.build_filters(dec_lo, dec_hi)
        else:
            rec_lo = torch.tensor(rec_lo).unsqueeze(0)
            rec_hi = torch.tensor(rec_hi).unsqueeze(0)
            self.build_filters(rec_lo, rec_hi)

    def build_filters(self, lo, hi):
        # construct 2d filter
        self.dim_size = lo.shape[-1]
        ll = self.outer(lo, lo)
        lh = self.outer(hi, lo)
        hl = self.outer(lo, hi)
        hh = self.outer(hi, hi)
        filters = torch.stack([ll, lh, hl, hh], dim=0)
        filters = filters.unsqueeze(1)
        self.register_buffer("filters", filters)  # [4, 1, height, width]

    def outer(self, a: torch.Tensor, b: torch.Tensor):
        """Torch implementation of numpy's outer for 1d vectors."""
        a_flat = torch.reshape(a, [-1])
        b_flat = torch.reshape(b, [-1])
        a_mul = torch.unsqueeze(a_flat, dim=-1)
        b_mul = torch.unsqueeze(b_flat, dim=0)
        return a_mul * b_mul

    def get_pad(self, data_len: int, filter_len: int):
        padr = (2 * filter_len - 3) // 2
        padl = (2 * filter_len - 3) // 2
        # pad to even singal length.
        if data_len % 2 != 0:
            padr += 1
        return padr, padl

    def adaptive_pad(self, data):
        padb, padt = self.get_pad(data.shape[-2], self.dim_size)
        padr, padl = self.get_pad(data.shape[-1], self.dim_size)

        data_pad = torch.nn.functional.pad(
            data, [padl, padr, padt, padb], mode=self.mode
        )
        return data_pad

    def forward(self, data):
        if self.inverse is False:
            b, c, h, w = data.shape
            dec_res = []
            data = self.adaptive_pad(data)
            for filter in self.filters:
                dec_res.append(
                    torch.nn.functional.conv2d(
                        data, filter.repeat(c, 1, 1, 1), stride=2, groups=c
                    )
                )
            return dec_res
        else:
            b, c, h, w = data[0].shape
            data = torch.stack(data, dim=2).reshape(b, -1, h, w)
            rec_res = torch.nn.functional.conv_transpose2d(
                data, self.filters.repeat(c, 1, 1, 1), stride=2, groups=c
            )
            return rec_res



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
        # pdb.set_trace()
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.act = nn.SiLU(inplace=act_inplace)

    def forward(self, x):
        # input -> 14, 9, 144, 144 -> 14, 64, 144, 144 -> 14, 64, 72, 72 -> 14, 64, 36, 36
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
        downsampling = False,
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
            groups=out_channels
        )

        self.final_block = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, stride=stride), # stride在这里实现下采样
            nn.InstanceNorm2d(out_channels), 
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)

        out2_temp = self.branch2_conv1(x)
        out2 = self.branch2_dwconv(out2_temp)

        out = torch.cat([out1, out2], dim=1)

        out = self.final_block(out)

        return out


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse:
        return list(reversed(samplings[:N]))
    else:
        return samplings[:N]


class Encoder(nn.Module):
    def __init__(self, C_in, C_hid, N_S):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
            DoubleBranchConv(
                C_in,
                C_hid,
                downsampling=samplings[0],
            ),
            *[
                DoubleBranchConv(
                    C_hid, C_hid, downsampling=s,
                )
                for s in samplings[1:]
            ]
        )

    def forward(self, x):  # B*T, C, H, W
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    def __init__(self, C_hid, C_out, N_S):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[
                BasicConv2d(
                    C_hid, C_hid, 3, upsampling=s
                )
                for s in samplings[:-1]
            ],
            BasicConv2d(
                C_hid,
                C_hid,
                3,
                upsampling=samplings[-1],
            )
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec) - 1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y




class MetaBlock(nn.Module):
    def __init__(
        self,
        T,
        in_channels,
        input_resolution=None,
        drop=0.0,
        drop_path=0.0,
        d_state=16,
    ):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.hid_channels = T * in_channels

        self.hi_mamba = HilbertScan3DMambaBlock(dim = in_channels, hid_dim=int(in_channels / 2), patch_size=(2, 9, 9), mlp_ratio=1.2, expand=1, d_state = d_state, drop = drop, drop_path=drop_path)
        
        self.wt = WaveletTransform2D()
        self.iwt = WaveletTransform2D(inverse=True)
        
        
        self.conv = nn.Sequential(nn.Conv2d(in_channels = self.hid_channels * 4, out_channels = self.hid_channels * 4, kernel_size = 3, padding = 1, groups=self.hid_channels * 4),
                                  nn.InstanceNorm2d(self.hid_channels * 4),
                                  nn.GELU(),
                                  nn.Conv2d(in_channels = self.hid_channels * 4, out_channels = self.hid_channels * 4, kernel_size = 1),
                                  nn.BatchNorm2d(self.hid_channels * 4),
                                  nn.GELU())
        self.final_conv = nn.Sequential(nn.LayerNorm(normalized_shape = input_resolution),
                                        nn.Conv2d(in_channels=self.hid_channels, out_channels=self.hid_channels, kernel_size=15, padding=int(15 / 2)),
                                        nn.InstanceNorm2d(self.hid_channels),
                                        nn.GELU(),
                                        nn.Conv2d(in_channels=self.hid_channels, out_channels=self.hid_channels, kernel_size=15, padding=int(15 / 2)),
                                        nn.InstanceNorm2d(self.hid_channels),
                                        nn.GELU())

    def forward(self, x):
        B, T, C, H, W = x.shape
        
        wavelet = x.reshape(B, T * C, H, W)
        
        x_orig = wavelet
        # B, T * C, H, W
        LL, LH, HL, HH = self.wt(wavelet)
        y = torch.concat([LL, LH, HL, HH], dim = 1)
        
        y = self.conv(y)
        LL, LH, HL, HH = torch.chunk(y, chunks=4, dim = 1)
        wavelet_out = self.iwt([LL, LH, HL, HH]) # (B, T * C, H, W)
        
        
        wavelet_out = wavelet_out.reshape(B, T, C, H, W)
        mamba_out = self.hi_mamba(x, wavelet_out)
        
        mamba_out = mamba_out.reshape(B, T * C, H, W)

        out = self.final_conv(mamba_out) + mamba_out
        out = out.reshape(B, T, C, H, W)
        return out


class MidMetaNet(nn.Module):
    def __init__(
        self,
        T,
        channel_in,
        N2,
        input_resolution=None,
        drop=0.0,
        drop_path=0.1,
    ):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2
        self.N2 = N2
        dpr = [
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)
        ]
        # downsample
        enc_layers = [
            MetaBlock(
                T,
                channel_in,
                input_resolution,
                drop,
                drop_path=dpr[0],
                d_state=16,
            )
        ]
        # middle layers
        for i in range(1, N2 - 1):
            enc_layers.append(
                MetaBlock(
                    T,
                    channel_in,
                    input_resolution,
                    drop,
                    drop_path=dpr[i],
                )
            )
        # upsample
        enc_layers.append(
            MetaBlock(
                T,
                channel_in,
                input_resolution,
                drop,
                drop_path=drop_path,
            )
        )
        self.enc = nn.Sequential(*enc_layers)
    def forward(self, x):
        original = x
        B, T, C, H, W = x.shape
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
        y = z + original
        return y



class VideoSICLoss(nn.Module):
    def __init__(self, lambda_grad=0.1):
        super(VideoSICLoss, self).__init__()
        self.lambda_grad = lambda_grad
        
    def spatial_gradient(self, x):
        grad_h = torch.zeros_like(x)
        grad_h[:, :, :, :-1, :] = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]

        grad_w = torch.zeros_like(x)
        grad_w[:, :, :, :, :-1] = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        
        return grad_h, grad_w
    
    def gradient_loss(self, pred, target):
        pred_grad_h, pred_grad_w = self.spatial_gradient(pred)
        target_grad_h, target_grad_w = self.spatial_gradient(target)
        grad_loss_h = F.l1_loss(pred_grad_h, target_grad_h)
        grad_loss_w = F.l1_loss(pred_grad_w, target_grad_w)
        
        return grad_loss_h + grad_loss_w
    
    def forward(self, pred, target):
        L_rec = F.l1_loss(pred, target)

        L_grad = self.gradient_loss(pred, target)

        L_total = L_rec + self.lambda_grad * L_grad

        loss_dict = {
            'total_loss': L_total,
            'reconstruction_loss': L_rec,
            'gradient_loss': L_grad
        }
        
        return L_total, loss_dict


class FH_Mamba(nn.Module):
    def __init__(
        self,
        T,
        C,  # 3*3=9
        img_size,
        patch_size,
        hid_S,
        hid_T=16,
        N_S=4,
        drop=0.0,
        drop_path=0.0,
    ):
        self.hid_T = hid_T
        super(FH_Mamba, self).__init__()
        H = img_size[0] // patch_size[0]
        W = img_size[1] // patch_size[1]
        H, W = int(H / 2 ** (N_S / 2)), int(
            W / 2 ** (N_S / 2)
        )

        num_downsampling_ops = N_S // 2 
        
        H_bottleneck = img_size[0] // (patch_size[0] * (2 ** num_downsampling_ops))
        W_bottleneck = img_size[1] // (patch_size[1] * (2 ** num_downsampling_ops))
        
        self.hid_S = hid_S
        self.T = T

        self.enc = Encoder(
            C,
            hid_S,
            N_S,
        )
        self.dec = Decoder(
            hid_S,
            C,
            N_S,
        )        
        self.reshape_i = nn.Conv3d(in_channels=hid_S, out_channels=hid_T, kernel_size=1)
        self.reshape_o = nn.Conv3d(in_channels=hid_T, out_channels=hid_S, kernel_size=1)
        self.hid = nn.Sequential(
            MetaBlock(
                    T,
                    hid_T,
                    (H_bottleneck, W_bottleneck),
                    drop,
                    drop_path=drop_path,
                ),
            MetaBlock(
                    T,
                    hid_T,
                    (H_bottleneck, W_bottleneck),
                    drop,
                    drop_path=drop_path,
                ),
            MetaBlock(
                    T,
                    hid_T,
                    (H_bottleneck, W_bottleneck),
                    drop,
                    drop_path=drop_path,
                ),
        )

        self.criterion = VideoSICLoss()

    def forward(self, input_x, targets):
        T_seg = self.T

        B, T_total_input, C_frame, H_img, W_img = input_x.shape
        recursion = 1

        full_prediction = torch.zeros_like(targets).to(input_x.device)

        current_input_segment = input_x[:, :T_seg, ...]

        total_loss = 0.0
        num_loss_terms = 0

        for r_idx in range(recursion):
            # [B*T_seg, C_frame, H_img, W_img]
            x_for_enc = current_input_segment.reshape(B * T_seg, C_frame, H_img, W_img)
            embed, skip = self.enc(x_for_enc) # embed: [B*T_seg, C_bottle, H_bottle, W_bottle]

            _, C_bottle, H_bottle, W_bottle = embed.shape
            
            z_for_hid = embed.view(B, C_bottle,T_seg, H_bottle, W_bottle)
            z_for_hid = self.reshape_i(z_for_hid)
            z_for_hid = z_for_hid.reshape(B, T_seg, self.hid_T, H_bottle, W_bottle)
            
            Fc_time_seq = self.hid(z_for_hid)
            Fc_time_seq = Fc_time_seq.reshape(B, self.hid_T, T_seg, H_bottle, W_bottle)
            Fc_time_seq = self.reshape_o(Fc_time_seq)
            Fc_flat = Fc_time_seq.reshape(B * T_seg, C_bottle, H_bottle, W_bottle)

            next_frames_segment = self.dec(Fc_flat, skip) # [B*T_seg, C_frame, H_img, W_img]
            next_frames_segment = torch.clamp(next_frames_segment, 0, 1)
            next_frames_segment = next_frames_segment.reshape(B, T_seg, C_frame, H_img, W_img)

            start_idx = r_idx * T_seg
            end_idx = (r_idx + 1) * T_seg
            full_prediction[:, start_idx:end_idx, ...] = next_frames_segment

            current_target_segment = targets[:, start_idx:end_idx, ...]
            loss_segment = self.criterion(next_frames_segment, current_target_segment)
            total_loss += loss_segment[0]
            num_loss_terms +=1

            if r_idx < recursion - 1:
                current_input_segment = next_frames_segment 
        
        avg_loss = total_loss / num_loss_terms if num_loss_terms > 0 else torch.tensor(0.0).to(input_x.device)

        return full_prediction, avg_loss
