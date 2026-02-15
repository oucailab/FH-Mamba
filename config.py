import torch

class Configs:
    def __init__(self):
        pass

configs = Configs()
configs.model = "FH_Mamba"

# 训练参数相关
configs.batch_size_vali = 1
configs.batch_size = 1
configs.lr = 0.001
configs.weight_decay = 1e-4
configs.num_epochs = 100
configs.early_stopping = True
configs.patience = 10
configs.gradient_clipping = True
configs.clipping_threshold = 1.0
configs.layer_norm = False
configs.display_interval = 50
# 数据集相关
configs.img_size = (432, 432)
configs.input_dim = 1
configs.output_dim = 1
configs.input_length = 14
configs.pred_length = 14
configs.input_gap = 1
configs.pred_gap = 1
configs.pred_shift = configs.pred_gap * configs.pred_length
configs.train_period = (19910101, 20101231)
configs.eval_period = (20110101, 20151231)

# FH_Mamba相关
configs.patch_size = (2, 2)
configs.hid_S = 32
configs.hid_T_channels = 8
configs.N_S = 2
configs.drop = 0.1
configs.drop_path = 0.05

# paths
configs.full_data_path = "data/full_sic.nc"
configs.train_log_path = "train_logs"
configs.test_results_path = "test_results"