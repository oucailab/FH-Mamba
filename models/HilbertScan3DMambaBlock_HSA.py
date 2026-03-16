import math
import torch
import torch.nn as nn
from einops import repeat
from timm.models.layers import DropPath, trunc_normal_
from mamba_ssm import Mamba


def Hilbert3d(width, height, depth):
    yield from generate3d(0, 0, 0, width, 0, 0, 0, height, 0, 0, 0, depth)


def sgn(x):
    return -1 if x < 0 else (1 if x > 0 else 0)


def generate3d(x, y, z, ax, ay, az, bx, by, bz, cx, cy, cz):
    w = abs(ax + ay + az)
    h = abs(bx + by + bz)
    d = abs(cx + cy + cz)

    (dax, day, daz) = (sgn(ax), sgn(ay), sgn(az))
    (dbx, dby, dbz) = (sgn(bx), sgn(by), sgn(bz))
    (dcx, dcy, dcz) = (sgn(cx), sgn(cy), sgn(cz))

    if h == 1 and d == 1:
        for _ in range(0, w):
            yield (x, y, z)
            (x, y, z) = (x + dax, y + day, z + daz)
        return

    if w == 1 and d == 1:
        for _ in range(0, h):
            yield (x, y, z)
            (x, y, z) = (x + dbx, y + dby, z + dbz)
        return

    if w == 1 and h == 1:
        for _ in range(0, d):
            yield (x, y, z)
            (x, y, z) = (x + dcx, y + dcy, z + dcz)
        return

    (ax2, ay2, az2) = (ax // 2, ay // 2, az // 2)
    (bx2, by2, bz2) = (bx // 2, by // 2, bz // 2)
    (cx2, cy2, cz2) = (cx // 2, cy // 2, cz // 2)

    w2 = abs(ax2 + ay2 + az2)
    h2 = abs(bx2 + by2 + bz2)
    d2 = abs(cx2 + cy2 + cz2)

    if (w2 % 2) and (w > 2):
        (ax2, ay2, az2) = (ax2 + dax, ay2 + day, az2 + daz)

    if (h2 % 2) and (h > 2):
        (bx2, by2, bz2) = (bx2 + dbx, by2 + dby, bz2 + dbz)

    if (d2 % 2) and (d > 2):
        (cx2, cy2, cz2) = (cx2 + dcx, cy2 + dcy, cz2 + dcz)

    if (2 * w > 3 * h) and (2 * w > 3 * d):
        yield from generate3d(x, y, z, ax2, ay2, az2, bx, by, bz, cx, cy, cz)

        yield from generate3d(
            x + ax2,
            y + ay2,
            z + az2,
            ax - ax2,
            ay - ay2,
            az - az2,
            bx,
            by,
            bz,
            cx,
            cy,
            cz,
        )

    elif 3 * h > 4 * d:
        yield from generate3d(x, y, z, bx2, by2, bz2, cx, cy, cz, ax2, ay2, az2)

        yield from generate3d(
            x + bx2,
            y + by2,
            z + bz2,
            ax,
            ay,
            az,
            bx - bx2,
            by - by2,
            bz - bz2,
            cx,
            cy,
            cz,
        )

        yield from generate3d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            z + (az - daz) + (bz2 - dbz),
            -bx2,
            -by2,
            -bz2,
            cx,
            cy,
            cz,
            -(ax - ax2),
            -(ay - ay2),
            -(az - az2),
        )

    elif 3 * d > 4 * h:
        yield from generate3d(x, y, z, cx2, cy2, cz2, ax2, ay2, az2, bx, by, bz)

        yield from generate3d(
            x + cx2,
            y + cy2,
            z + cz2,
            ax,
            ay,
            az,
            bx,
            by,
            bz,
            cx - cx2,
            cy - cy2,
            cz - cz2,
        )

        yield from generate3d(
            x + (ax - dax) + (cx2 - dcx),
            y + (ay - day) + (cy2 - dcy),
            z + (az - daz) + (cz2 - dcz),
            -cx2,
            -cy2,
            -cz2,
            -(ax - ax2),
            -(ay - ay2),
            -(az - az2),
            bx,
            by,
            bz,
        )

    else:
        yield from generate3d(x, y, z, bx2, by2, bz2, cx2, cy2, cz2, ax2, ay2, az2)

        yield from generate3d(
            x + bx2,
            y + by2,
            z + bz2,
            cx,
            cy,
            cz,
            ax2,
            ay2,
            az2,
            bx - bx2,
            by - by2,
            bz - bz2,
        )

        yield from generate3d(
            x + (bx2 - dbx) + (cx - dcx),
            y + (by2 - dby) + (cy - dcy),
            z + (bz2 - dbz) + (cz - dcz),
            ax,
            ay,
            az,
            -bx2,
            -by2,
            -bz2,
            -(cx - cx2),
            -(cy - cy2),
            -(cz - cz2),
        )

        yield from generate3d(
            x + (ax - dax) + bx2 + (cx - dcx),
            y + (ay - day) + by2 + (cy - dcy),
            z + (az - daz) + bz2 + (cz - dcz),
            -cx,
            -cy,
            -cz,
            -(ax - ax2),
            -(ay - ay2),
            -(az - az2),
            bx - bx2,
            by - by2,
            bz - bz2,
        )

        yield from generate3d(
            x + (ax - dax) + (bx2 - dbx),
            y + (ay - day) + (by2 - dby),
            z + (az - daz) + (bz2 - dbz),
            -bx2,
            -by2,
            -bz2,
            cx2,
            cy2,
            cz2,
            -(ax - ax2),
            -(ay - ay2),
            -(az - az2),
        )


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, nf, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).contiguous().reshape(b, c, nf, h, w)
        x = self.dwconv(x)
        x = x.contiguous().flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, nf, h, w):
        x = self.fc1(x)
        x = self.dwconv(x, nf, h, w)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SequenceShuffleAttention(nn.Module):
    def __init__(self, dim, num_sequences=3):
        super().__init__()
        self.dim = dim
        self.num_sequences = num_sequences

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.group_conv = nn.Conv1d(
            in_channels=self.num_sequences * self.dim,
            out_channels=self.num_sequences * self.dim,
            kernel_size=1,
            groups=self.dim,
            bias=False,
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, sequences):
        if not isinstance(sequences, list) or len(sequences) != self.num_sequences:
            raise ValueError(f"Input must be a list of {self.num_sequences} tensors.")

        b, _, c = sequences[0].shape
        if c != self.dim:
            raise ValueError(f"Input feature dimension {c} does not match module's dim {self.dim}.")

        pooled_features_list = [self.avg_pool(seq.transpose(1, 2)) for seq in sequences]
        x_tilde_cat = torch.cat([p.squeeze(-1) for p in pooled_features_list], dim=1)

        x_tilde_reshaped = x_tilde_cat.view(b, self.num_sequences, self.dim)
        x_hat_intermediate = x_tilde_reshaped.transpose(1, 2).contiguous()
        x_hat = x_hat_intermediate.view(b, self.num_sequences * self.dim, 1)

        weights_shuffled = self.group_conv(x_hat)
        weights_unshuffled_intermediate = weights_shuffled.view(b, self.dim, self.num_sequences)
        w_tilde_reshaped = weights_unshuffled_intermediate.transpose(1, 2).contiguous()

        attention_weights = self.softmax(w_tilde_reshaped)
        output_sum = torch.zeros_like(sequences[0])

        for i in range(self.num_sequences):
            w_i = attention_weights[:, i, :]
            w_i_broadcast = w_i.unsqueeze(1)
            output_sum += w_i_broadcast * sequences[i]

        return output_sum


class MambaLayerlocal(nn.Module):
    def __init__(
        self,
        dim,
        d_state=16,
        d_conv=4,
        expand=0.5,
        mlp_ratio=4,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.norm1 = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)
        self.ssa = SequenceShuffleAttention(dim=dim, num_sequences=3)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, freq_x, hilbert_curve):
        x = x.permute(0, 2, 1, 3, 4)
        b, c, nf, h, w = x.shape
        assert c == self.dim

        img_dims = x.shape[2:]
        x_hw = x.flatten(2).contiguous()

        x_hil = x_hw.index_select(dim=-1, index=hilbert_curve)
        x_flat = x_hil.transpose(-1, -2).contiguous()

        freq_x = freq_x.permute(0, 2, 1, 3, 4)
        freq_x_hw = freq_x.flatten(2).contiguous()
        freq_x_hil = freq_x_hw.index_select(dim=-1, index=hilbert_curve)
        freq_x_flat_hilbert = freq_x_hil.transpose(-1, -2).contiguous()

        normed_x = self.norm1(x_flat)
        mamba_out_fwd = self.mamba(normed_x)

        x_rev = torch.flip(normed_x, dims=[1])
        mamba_out_rev_temp = self.mamba(x_rev)
        mamba_out_rev = torch.flip(mamba_out_rev_temp, dims=[1])

        mamba_out = self.ssa([mamba_out_fwd, mamba_out_rev, freq_x_flat_hilbert])
        x_mamba = x_flat + self.drop_path(mamba_out)

        x_mamba_out = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), nf, h, w))

        outmamba = x_mamba_out.transpose(-1, -2)

        unscattered_out = torch.zeros_like(outmamba)
        hilbert_curve_re = repeat(hilbert_curve, "hw -> b c hw", b=outmamba.shape[0], c=outmamba.shape[1])
        unscattered_out.scatter_(dim=-1, index=hilbert_curve_re, src=outmamba)

        sum_out = unscattered_out.reshape(b, c, *img_dims).contiguous()
        out = sum_out.permute(0, 2, 1, 3, 4)

        return out


class HilbertScan3DMambaBlock(nn.Module):
    def __init__(
        self,
        dim,
        hid_dim,
        patch_size,
        d_state=16,
        d_conv=4,
        expand=0.5,
        mlp_ratio=4.0,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.pT, self.pH, self.pW = patch_size
        self.mamba_feat_dim = self.pT * self.pH * self.pW * hid_dim

        self.mamba_layer = MambaLayerlocal(
            dim=self.mamba_feat_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mlp_ratio=mlp_ratio,
            drop=drop,
            drop_path=drop_path,
            act_layer=act_layer,
        )
        self.hilbert_curve_cache = {}
        self.conv3d_i = nn.Conv3d(in_channels=dim, out_channels=hid_dim, kernel_size=1)
        self.conv3d_o = nn.Conv3d(in_channels=hid_dim, out_channels=dim, kernel_size=1)

    def _get_hilbert_curve(self, n_pt, n_ph, n_pw, device):
        cache_key = (n_pt, n_ph, n_pw)
        if cache_key in self.hilbert_curve_cache:
            return self.hilbert_curve_cache[cache_key].to(device)
        coords_3d = list(Hilbert3d(n_pw, n_ph, n_pt))
        hilbert_indices = torch.tensor(
            [
                t_coord * (n_ph * n_pw) + h_coord * n_pw + w_coord
                for w_coord, h_coord, t_coord in coords_3d
            ],
            dtype=torch.long,
        )
        self.hilbert_curve_cache[cache_key] = hilbert_indices
        return hilbert_indices.to(device)

    def _prepare_input(self, tensor, b, t_hid, c_hid, h_hid, w_hid, n_pt, n_ph, n_pw):
        x_reshaped = tensor.reshape(b, n_pt, self.pT, c_hid, n_ph, self.pH, n_pw, self.pW)
        x_permuted_patches = x_reshaped.permute(0, 1, 4, 6, 2, 5, 7, 3).contiguous()
        x_flat_content = x_permuted_patches.reshape(b, n_pt * n_ph * n_pw, self.mamba_feat_dim)

        x_grid_patches = x_flat_content.reshape(b, n_pt, n_ph, n_pw, self.mamba_feat_dim)
        x_to_mamba_layer = x_grid_patches.permute(0, 1, 4, 2, 3).contiguous()
        return x_to_mamba_layer

    def forward(self, x, freqency_x):
        assert x.shape == freqency_x.shape, "Input x and freqency_x must have the same shape."
        b, t_orig, c_orig, h_orig, w_orig = x.shape
        assert c_orig == self.dim, "Input channel dimension mismatch"

        x = x.reshape(b, c_orig, t_orig, h_orig, w_orig)
        x = self.conv3d_i(x)
        x = x.reshape(b, t_orig, -1, h_orig, w_orig)

        freqency_x = freqency_x.reshape(b, c_orig, t_orig, h_orig, w_orig)
        freqency_x = self.conv3d_i(freqency_x)
        freqency_x = freqency_x.reshape(b, t_orig, -1, h_orig, w_orig)

        b, t_hid, c_hid, h_hid, w_hid = x.shape
        assert t_hid % self.pT == 0 and h_hid % self.pH == 0 and w_hid % self.pW == 0

        n_pt = t_hid // self.pT
        n_ph = h_hid // self.pH
        n_pw = w_hid // self.pW

        x_to_mamba_layer = self._prepare_input(x, b, t_hid, c_hid, h_hid, w_hid, n_pt, n_ph, n_pw)
        freq_to_mamba_layer = self._prepare_input(
            freqency_x, b, t_hid, c_hid, h_hid, w_hid, n_pt, n_ph, n_pw
        )

        hilbert_curve_patch_grid = self._get_hilbert_curve(n_pt, n_ph, n_pw, x.device)

        output_from_mamba_layer = self.mamba_layer(
            x_to_mamba_layer,
            freq_x=freq_to_mamba_layer,
            hilbert_curve=hilbert_curve_patch_grid,
        )

        out_permuted_d = output_from_mamba_layer.permute(0, 1, 3, 4, 2).contiguous()
        out_unflat_patch_content = out_permuted_d.reshape(
            b, n_pt, n_ph, n_pw, self.pT, self.pH, self.pW, c_hid
        )
        out_gathered_parts = out_unflat_patch_content.permute(0, 1, 4, 7, 2, 5, 3, 6).contiguous()
        final_output = out_gathered_parts.reshape(b, t_hid, c_hid, h_hid, w_hid)

        final_output = final_output.reshape(b, c_hid, t_hid, h_hid, w_hid)
        final_output = self.conv3d_o(final_output)
        final_output = final_output.reshape(b, t_orig, c_orig, h_orig, w_orig)

        return final_output
