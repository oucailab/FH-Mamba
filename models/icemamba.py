# icemamba_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


class WaveletTransform2D(nn.Module):
    """Compute a two-dimensional wavelet transform."""

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
        self.dim_size = lo.shape[-1]
        ll = self.outer(lo, lo)
        lh = self.outer(hi, lo)
        hl = self.outer(lo, hi)
        hh = self.outer(hi, hi)
        filters = torch.stack([ll, lh, hl, hh], dim=0)
        filters = filters.unsqueeze(1)
        self.register_buffer("filters", filters)

    def outer(self, a: torch.Tensor, b: torch.Tensor):
        a_flat = torch.reshape(a, [-1])
        b_flat = torch.reshape(b, [-1])
        a_mul = torch.unsqueeze(a_flat, dim=-1)
        b_mul = torch.unsqueeze(b_flat, dim=0)
        return a_mul * b_mul

    def get_pad(self, data_len: int, filter_len: int):
        padr = (2 * filter_len - 3) // 2
        padl = (2 * filter_len - 3) // 2
        if data_len % 2 != 0:
            padr += 1
        return padr, padl

    def adaptive_pad(self, data):
        padb, padt = self.get_pad(data.shape[-2], self.dim_size)
        padr, padl = self.get_pad(data.shape[-1], self.dim_size)

        data_pad = torch.nn.functional.pad(data, [padl, padr, padt, padb], mode=self.mode)
        return data_pad

    def forward(self, data):
        if self.inverse is False:
            _, c, _, _ = data.shape
            dec_res = []
            data = self.adaptive_pad(data)
            for filter_tensor in self.filters:
                dec_res.append(
                    torch.nn.functional.conv2d(
                        data, filter_tensor.repeat(c, 1, 1, 1), stride=2, groups=c
                    )
                )
            return dec_res

        b, c, h, w = data[0].shape
        data = torch.stack(data, dim=2).reshape(b, -1, h, w)
        rec_res = torch.nn.functional.conv_transpose2d(
            data, self.filters.repeat(c, 1, 1, 1), stride=2, groups=c
        )
        return rec_res


class OrthoConvolution(nn.Module):
    def __init__(self, channels, kernel_size_tuple, bias=False):
        super(OrthoConvolution, self).__init__()
        padding_h = (kernel_size_tuple[0] - 1) // 2
        padding_w = (kernel_size_tuple[1] - 1) // 2
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size_tuple,
            padding=(padding_h, padding_w),
            bias=bias,
            groups=channels,
        )

    def forward(self, x):
        return self.conv(x)


class MultiScaleMapping(nn.Module):
    def __init__(self, channels, input_resolution_hw):
        super(MultiScaleMapping, self).__init__()
        self.channels = channels
        h, w = input_resolution_hw
        self.ln = nn.LayerNorm([channels, h, w])

        self.oc_1x13 = OrthoConvolution(channels, (1, 13))
        self.oc_13x1 = OrthoConvolution(channels, (13, 1))
        self.oc_1x17 = OrthoConvolution(channels, (1, 17))
        self.oc_17x1 = OrthoConvolution(channels, (17, 1))
        self.oc_1x21 = OrthoConvolution(channels, (1, 21))
        self.oc_21x1 = OrthoConvolution(channels, (21, 1))

        self.conv_q = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def _ock(self, x_norm, conv1xk, convkx1):
        return conv1xk(x_norm) + convkx1(x_norm)

    def forward(self, x):
        x_norm = self.ln(x)

        oc13_out = self._ock(x_norm, self.oc_1x13, self.oc_13x1)
        oc17_out = self._ock(x_norm, self.oc_1x17, self.oc_17x1)
        oc21_out = self._ock(x_norm, self.oc_1x21, self.oc_21x1)

        sum_oc_outputs = oc13_out + oc17_out + oc21_out

        q = self.conv_q(sum_oc_outputs)
        k = self.conv_k(sum_oc_outputs)
        v = self.conv_v(sum_oc_outputs)

        return q, k, v


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature=None, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        b, c, h, w = q.shape
        q_flat = q.view(b, c, h * w).transpose(1, 2)
        k_flat = k.view(b, c, h * w)
        v_flat = v.view(b, c, h * w).transpose(1, 2)

        if self.temperature is None:
            self.temperature = c ** 0.5

        attn = torch.bmm(q_flat, k_flat) / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.bmm(attn_weights, v_flat)
        output = output.transpose(1, 2).contiguous().view(b, c, h, w)
        return output


class DomainAttentionFusion(nn.Module):
    def __init__(self, channels):
        super(DomainAttentionFusion, self).__init__()
        self.attention = ScaledDotProductAttention()
        self.conv1x1_att1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.conv1x1_att2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, q1, k1, v1, q2, k2, v2):
        att_out_211 = self.attention(q2, k1, v1)
        f_daf1 = self.conv1x1_att1(att_out_211)

        att_out_122 = self.attention(q1, k2, v2)
        f_daf2 = self.conv1x1_att2(att_out_122)

        return f_daf1, f_daf2


class AdaptiveFeaturesFusionBlock(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3):
        super(AdaptiveFeaturesFusionBlock, self).__init__()
        self.W = nn.Parameter(torch.empty(1, in_channels, 1, 1))
        nn.init.zeros_(self.W)

        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

        self.norm = nn.LayerNorm(normalized_shape=in_channels)
        self.silu = nn.SiLU()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        alpha = self.sigmoid(self.W)
        fused_features = alpha * x1 + (1 - alpha) * x2

        x = self.conv(fused_features)
        x_permuted = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_permuted)
        x_restored = x_norm.permute(0, 3, 1, 2)
        output = self.silu(x_restored)
        return output


class MultiScaleAttentionFusion(nn.Module):
    def __init__(self, in_channels, input_resolution_hw):
        super(MultiScaleAttentionFusion, self).__init__()

        self.x1_mm = MultiScaleMapping(in_channels, input_resolution_hw)
        self.x2_mm = MultiScaleMapping(in_channels, input_resolution_hw)
        self.daf = DomainAttentionFusion(in_channels)
        self.affb = AdaptiveFeaturesFusionBlock(in_channels=in_channels)

    def forward(self, x1, x2):
        q1, k1, v1 = self.x1_mm(x1)
        q2, k2, v2 = self.x2_mm(x2)

        f_daf1, f_daf2 = self.daf(q1, k1, v1, q2, k2, v2)
        out = self.affb(f_daf1, f_daf2)
        return out
