import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from dateutil.relativedelta import relativedelta
import datetime
import xarray as xr
import netCDF4 as nc


def cal_time_length(start_time, end_time):
    """
    Calculate the number of days between two dates.

    Args:
    - start_time (str): Start date in the format YYYYMMDD.
    - end_time (str): End date in the format YYYYMMDD.

    Returns:
    - int: Number of days between the two dates.
    """
    assert len(str(start_time)) == 8 & len(str(end_time)) == 8
    start = datetime.datetime.strptime(str(start_time), "%Y%m%d")
    end = datetime.datetime.strptime(str(end_time), "%Y%m%d")
    delta = end - start
    length = delta.days + 1
    return length


def GenTimeList(start_time, end_time):
    """
    生成日期列表
    Args:
        start_time: 开始日期，格式为YYYYMMDD
        end_time: 结束日期，格式为YYYYMMDD
    Returns:
        日期列表，格式为list[YYYYMMDD]
    """
    Times = []
    current = datetime.datetime.strptime(str(start_time), "%Y%m%d")
    end = datetime.datetime.strptime(str(end_time), "%Y%m%d")
    # current = datetime.datetime.strptime(str(start_time), "%Y%m")
    # end = datetime.datetime.strptime(str(end_time), "%Y%m")

    while current <= end:
        Times.append(current.strftime("%Y%m%d"))
        # Times.append(current.strftime("%Y%m"))
        current += relativedelta(days=1)
        # current += relativedelta(months=1)

    return [int(x) for x in Times]


def read_from_file(data_file):
    """
    从文件中读取数据作为图像
    数据包含 448 行和 304 列
    Args:
        data_file: 数据文件路径
    Returns:
        图像数据的NumPy数组
    """
    with nc.Dataset(data_file) as img:
        return np.array(img.variables[list(img.variables.keys())[7]][:][0])


def read_process_data(txt, start_time, end_time):
    """
    读取和处理所有数据
    Args:
        start_time/end_time: YYYYMMDD
        txt: 包含要读取为图像的文件名列表的 .txt 文件
    Returns:
        imgs: 经过填充处理和陆地标记的 SIC 数据
    """
    assert len(str(start_time)) == 8 and len(str(end_time)) == 8
    with open(txt, "r") as f:
        img_names = np.array(f.read().split())

    times = np.array([int(x.split("_")[5][0:8]) for x in img_names])
    img_names = img_names[(times >= start_time) & (times <= end_time)]

    imgs = []
    for img_name in img_names:
        img = read_from_file(img_name)
        img = post_process_data(img)
        imgs.append(img)
    assert len(imgs) == cal_time_length(start_time, end_time)
    return imgs


def write_netcdf(data_paths_file, start_time, end_time, out_path):
    """
    将处理过的数据写入 .nc 文件以便下次读取
    Args:
        data_paths_file: 包含要处理的所有数据的文件路径的文本文件的路径，这个文本文件可以通过'gen_data_text.sh'脚本生成，前提是文件都按照规则存放好，可以使用'data/move_dataset.py'对文件进行排列
        start_time/end_time: 数据的时间范围（YYYYMMDD格式，如20080101）
        out_path: 输出的 .nc 文件的路径
    Returns:
        ds: 写入的 xarray.Dataset 对象
    """
    # 读取并处理数据
    imgs = read_process_data(data_paths_file, start_time, end_time)
    # 创建 xarray 数据集并写入 .nc 文件
    time_coords = GenTimeList(start_time, end_time)
    x_coords = range(432)
    y_coords = range(432)

    ds = xr.Dataset(
        {"imgs": (["time", "y", "x"], imgs)},
        coords={"time": time_coords, "x": x_coords, "y": y_coords},
    )

    ds.to_netcdf(out_path)
    return ds


def post_process_data(img):
    """
    0 - 1 Sea ice concentration
    -32767 Land

    处理数据，包括归一化、处理缺失数据、陆地屏蔽等
    Args:
        img: 输入的图像数据
    Returns:
        处理后的图像数据
    """

    # 处理陆地
    img[img == -32767] = 0

    # 归一化
    img = img / 100

    # # 断言确保没有超出范围的值
    assert not np.any(img > 1)

    return img


def unfold_StackOverChannel(img, patch_size):
    """
    将图像切分成多个小块并沿着通道堆叠
    Args:
        img (N, *, C, H, W): 最后两个维度必须是空间维度
        patch_size(k_h,k_w): 长度为2的元组，就是configs.patch_size
    Returns:
        output (N, *, C*k_h*k_w, H/k_h, W/k_w)
    """
    n_dim = len(img.size())
    assert n_dim == 4 or n_dim == 5
    if patch_size[0] == 1 and patch_size[1] == 1:
        return img

    pt = img.unfold(-2, size=patch_size[0], step=patch_size[0])
    pt = pt.unfold(-2, size=patch_size[1], step=patch_size[1]).flatten(-2)
    if n_dim == 4:  # (N, C, H, W)
        pt = pt.permute(0, 1, 4, 2, 3).flatten(1, 2)
    elif n_dim == 5:  # (N, T, C, H, W)
        pt = pt.permute(0, 1, 2, 5, 3, 4).flatten(2, 3)
    assert pt.size(-3) == img.size(-3) * patch_size[0] * patch_size[1]
    return pt


def fold_tensor(tensor, output_size, patch_size):
    """
    用non-overlapping的块重建图像
    Args:
        input tensor shape (N, *, C*k_h*k_w, h, w)
        output_size: (H, W)，要重建的原始图像的大小
        patch_size: (k_h, k_w)
        请注意，对于non-overlapping的滑动窗口，通常stride等于patch_size
    Returns:
        output (N, *, C, H=n_h*k_h, W=n_w*k_w)
    """
    n_dim = len(tensor.shape)
    assert n_dim == 4 or n_dim == 5

    if patch_size[0] == 1 and patch_size[1] == 1:
        return tensor

    # 展平输入
    f = tensor.flatten(0, 1) if n_dim == 5 else tensor

    # 使用 F.fold 函数进行重建
    folded = F.fold(
        f.flatten(-2),
        output_size=output_size,
        kernel_size=patch_size,
        stride=patch_size,
    )

    if n_dim == 5:
        folded = folded.view(tensor.size(0), tensor.size(1), *folded.shape[1:])

    return folded


def prepare_inputs_targets(
    len_time, input_gap, input_length, pred_shift, pred_gap, pred_length, samples_gap
):
    """
    Args:
        input_gap: 两个连续输入帧之间的时间间隔
        input_length: 输入帧的数量
        pred_shift: 最后一个目标预测的前导时间
        pred_gap: 两个连续输出帧之间的时间间隔
        pred_length: 输出帧的数量
        samples_gap: 两个检索样本的起始时间之间的间隔
    Returns:
        idx_inputs: 指向输入样本位置的索引
        idx_targets: 指向目标样本位置的索引
    """
    assert pred_shift >= pred_length
    input_span = input_gap * (input_length - 1) + 1
    input_ind = np.arange(0, input_span, input_gap)
    target_ind = np.arange(0, pred_shift, pred_gap) + input_span + pred_gap - 1
    ind = np.concatenate([input_ind, target_ind]).reshape(1, input_length + pred_length)
    max_n_sample = len_time - (input_span + pred_shift - 1)
    ind = ind + np.arange(max_n_sample)[:, np.newaxis] @ np.ones(
        (1, input_length + pred_length), dtype=int
    )
    idx_inputs = ind[::samples_gap, :input_length]
    idx_targets = ind[::samples_gap, input_length:]
    assert len(idx_inputs) == len(idx_targets)
    return idx_inputs, idx_targets


class SIC_dataset(Dataset):
    def __init__(
        self,
        full_data_path,
        start_time,
        end_time,
        input_gap,
        input_length,
        pred_shift,
        pred_gap,
        pred_length,
        samples_gap,
    ):
        """
        Args:
            full_data_path: the path specifying where the processed data file is located
            start_time/end_time: used to specify the dataset (train/eval/test) period
            input_gap: 两个连续输入帧之间的时间间隔
            input_length: 输入帧的数量
            pred_shift: 最后一个目标预测的前导时间
            pred_gap: 两个连续输出帧之间的时间间隔
            pred_length: 输出帧的数量
            samples_gap: 两个检索样本的起始时间之间的间隔
        """
        super().__init__()

        with xr.open_dataset(full_data_path) as full_data:
            self.times = GenTimeList(start_time, end_time)
            self.data = full_data.imgs.sel(time=self.times).values

        self.idx_inputs, self.idx_targets = prepare_inputs_targets(
            self.data.shape[0],
            input_gap=input_gap,
            input_length=input_length,
            pred_shift=pred_shift,
            pred_gap=pred_gap,
            pred_length=pred_length,
            samples_gap=samples_gap,
        )

    def __len__(self):
        return len(self.idx_inputs)

    def __getitem__(self, index):
        return (
            self.data[self.idx_inputs[index]][:, None],
            self.data[self.idx_targets[index]][:, None],
        )

    def GetInputs(self):
        return self.data[self.idx_inputs][:, :, None]

    def GetTargets(self):
        return self.data[self.idx_targets][:, :, None]

    def GetTimes(self):
        return np.array(self.times)[
            np.concatenate([self.idx_inputs, self.idx_targets], axis=1)
        ]

    def GetDataSetShape(self):
        inputs_B = self.idx_inputs.shape[0]
        inputs_T, inputs_C, inputs_H, inputs_W = self.data[self.idx_inputs[0]][
            :, None
        ].shape

        targets_B = self.idx_targets.shape[0]
        targets_T, targets_C, targets_H, targets_W = self.data[self.idx_targets[0]][
            :, None
        ].shape

        return {
            "inputs(B, T, C, H, W)": (inputs_B, inputs_T, inputs_C, inputs_H, inputs_W),
            "targets(B, T, C, H, W)": (
                targets_B,
                targets_T,
                targets_C,
                targets_H,
                targets_W,
            ),
        }
