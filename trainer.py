import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
import pickle
import math
import logging
from tqdm import tqdm
import pdb
from thop import profile
from torchinfo import summary
from thop import profile
from thop import clever_format # 用于格式化输出
from model_factory import IceNet
from utils import SIC_dataset
from metrics import *
from fvcore.nn import FlopCountAnalysis, parameter_count_table

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger()


class Trainer:
    """
    Trainer of IceNet.

    Args:
    - configs: configs of Model.
    """

    def __init__(self, configs):
        torch.set_float32_matmul_precision('high')
        self.configs = configs
        self.device = "cuda:0"
        self.arctic_mask = torch.from_numpy(np.load("arctic_mask.npy"))
        self._get_data()
        self._build_network()

    def _get_data(self):
        dataset_train = SIC_dataset(
            self.configs.full_data_path,
            self.configs.train_period[0],
            self.configs.train_period[1],
            self.configs.input_gap,
            self.configs.input_length,
            self.configs.pred_shift,
            self.configs.pred_gap,
            self.configs.pred_length,
            samples_gap=1,
        )

        dataset_vali = SIC_dataset(
            self.configs.full_data_path,
            self.configs.eval_period[0],
            self.configs.eval_period[1],
            self.configs.input_gap,
            self.configs.input_length,
            self.configs.pred_shift,
            self.configs.pred_gap,
            self.configs.pred_length,
            samples_gap=1,
        )

        self.dataloader_train = DataLoader(
            dataset_train, batch_size=self.configs.batch_size, shuffle=True
        )

        self.dataloader_vali = DataLoader(
            dataset_vali, batch_size=self.configs.batch_size_vali, shuffle=False
        )

    def _build_network(self):
        #self.network = torch.compile(IceNet(self.configs)).to(self.device)
        self.network = IceNet(self.configs).to(self.device)
        #print(self.network)
        self.optimizer = AdamW(
            self.network.parameters(),
            lr=self.configs.lr,
            weight_decay=self.configs.weight_decay,
        )

        self.lr_scheduler = OneCycleLR(
            optimizer=self.optimizer,
            epochs=self.configs.num_epochs,
            steps_per_epoch=len(self.dataloader_train),
            max_lr=self.configs.lr,
        )

    def _save_configs(self, config_path):
        with open(config_path, "wb") as path:
            pickle.dump(self.configs, path)

    def _save_model(self, path):
        torch.save(
            {
                "net": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def train_once(self, inputs, targets, mask):
        self.optimizer.zero_grad()

        inputs = inputs.float().to(self.device)
        targets = targets.float().to(self.device)
        mask = mask.float().to(self.device)
        sic_pred, loss = self.network(inputs, targets)

        total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        

        mse = mse_func(sic_pred, targets, mask)
        rmse = rmse_func(sic_pred, targets, mask)
        mae = mae_func(sic_pred, targets, mask)
        nse = nse_func(sic_pred, targets, mask)
        psnr = PSNR(sic_pred, targets, mask)
        bacc = BACC_func(sic_pred, targets, mask)

        loss.backward()

        if self.configs.gradient_clipping:
            nn.utils.clip_grad_norm_(
                self.network.parameters(), self.configs.clipping_threshold
            )

        self.optimizer.step()
        self.lr_scheduler.step()

        return (
            mse,
            rmse,
            mae,
            nse,
            psnr,
            loss.item(),
            bacc,
            total_params,
            # params,
            # flops
        )

    def vali(self, dataloader, mask):
        """
        evaluation part.

        Args:
        - dataloader: dataloader of evaluation dataset.
        """
        mask = mask.float().to(self.device)
        total_loss = 0
        total_mse = 0
        total_rmse = 0
        total_mae = 0
        total_nse = 0
        total_psnr = 0
        total_bacc = 0
        num_batches = len(dataloader)  # 获取数据加载器中的批次数
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(dataloader):
                inputs = inputs.float().to(self.device)
                targets = targets.float().to(self.device)
                sic_pred, loss = self.network(inputs, targets)

                total_loss += loss.item()
                total_mse += mse_func(sic_pred, targets, mask)
                total_rmse += rmse_func(sic_pred, targets, mask)
                total_mae += mae_func(sic_pred, targets, mask)
                total_nse += nse_func(sic_pred, targets, mask)
                total_psnr += PSNR(sic_pred, targets, mask)
                
                total_bacc += BACC_func(sic_pred, targets, mask)
                B, num_frames, C, H, W = sic_pred.shape
                baccs = []
                for frames_idx in range(num_frames):
                    pred_pic = sic_pred[:, frames_idx]
                    target_pic = targets[:, frames_idx]

                    bacc = BACC_pic_func(pred_pic, target_pic, mask)
                    baccs.append(bacc.item())
                print(baccs)
                
        assert num_batches > 0
        # pdb.set_trace()
        return (
            total_mse / num_batches,
            total_rmse / num_batches,
            total_mae / num_batches,
            total_nse / num_batches,
            total_psnr / num_batches,
            total_loss / num_batches,
            total_bacc / num_batches,
        )

    def test(self, dataloader):
        sic_pred_list = []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.float().to(self.device)
                targets = targets.float().to(self.device)
                sic_pred, _ = self.network(inputs, targets)

                sic_pred_list.append(sic_pred)

        return torch.cat(sic_pred_list, dim=0)

    def train(self, chk_path):

        log_file = f"{self.configs.train_log_path}/train_{self.configs.model}.log"
        self.logger = setup_logging(log_file)

        self.logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.logger.info("###### Training begins! ########\n")
        self.logger.info(self.configs.__dict__)
        self.network.eval()
            # 注意：这里计算的是 MACs (Multiply-Accumulate operations)
            #         通常认为 1 MAC ≈ 2 FLOPs，但 thop 的文档和社区习惯有时会混用
        input_tensor_a = torch.randn(1, 14, 1, 432, 432).to(self.device)                 #    profile 函数默认返回 MACs，可以通过 flops_units 参数调整
        input_tensor_b = torch.randn(1,14,1,432,432).to(self.device)
        inputs = (input_tensor_a, input_tensor_b)
        flops = FlopCountAnalysis(self.network, inputs)

        #4. 获取总 FLOPs (通常以 GFLOPs 显示)
        total_flops = flops.total()
        print(f"Total FLOPs: {total_flops}")
        #fvcore 通常以 GigaFLOPs (1e9) 为单位报告，可以转换
        print(f"Total GFLOPs: {total_flops / 1e9:.2f} GFLOPs")

        count = 0
        best = math.inf
        total_start_time = time.time()
        
        for i in range(self.configs.num_epochs):
            # train
            self.network.train()
            loop = tqdm(
                (self.dataloader_train), total=len(self.dataloader_train), leave=True
            )
            for inputs, targets in loop:
                mse, rmse, mae, nse, psnr, loss, bacc, total_params = self.train_once(
                    inputs, targets, self.arctic_mask
                )
                loop.set_description(f"epoch [{i+1}/{self.configs.num_epochs}]")
                loop.set_postfix(
                    mse="{:.4f}".format(mse),
                    rmse="{:.4f}".format(rmse),
                    mae="{:.4f}".format(mae),
                    nse="{:.4f}".format(nse),
                    psnr="{:.4f}".format(psnr),
                    loss="{:.4f}".format(loss),
                    bacc="{:.4f}".format(bacc),
                    total_params="{:.1f}".format(total_params),
                    # params="{:.1f}".format(params),
                    # flops="{:.1f}".format(flops),
                )
            
            # evaluation
            self.network.eval()
            (mse_eval, rmse_eval, mae_eval, nse_eval, psnr_eval, loss_eval, bacc_eval) = self.vali(
                self.dataloader_vali, self.arctic_mask
            )
            self.logger.info(
                "\nepoch {} vali loss: {:.4f} mse: {:.4f}, rmse: {:.4f}, mae: {:.4f}, nse: {:.4f}, psnr: {:.4f}, bacc: {:.4f}".format(
                    i + 1, loss_eval, mse_eval, rmse_eval, mae_eval, nse_eval, psnr_eval, bacc_eval
                )
            )

            # # 使用更好的模型保存方式，例如只保存最好的模型
            if mae_eval < best:
                count = 0
                self.logger.info(
                    "\nvali score is improved from {:.5f} to {:.5f}, saving model\n".format(
                        best, mae_eval
                    )
                )
                self._save_model(chk_path)
                best = mae_eval
            else:
                count += 1
                self.logger.info(
                    "\nvali score is not improved for {} epoch\n".format(count)
                )

            if count == self.configs.patience and self.configs.early_stopping:
                self.logger.info(
                    "early stopping reached, best score is {:5f}\n".format(best)
                )
                break

        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        self.logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.logger.info(f"Total training time: {total_time:.2f} seconds")
        self.logger.info("\n###### Training complete! ########\n")
