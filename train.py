
import torch

from config import configs
from trainer import Trainer
import numpy as np
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(42)  # 可以使用任何整数作为种子

    trainer = Trainer(configs)
    trainer._save_configs(f"pkls/train_config_{configs.model}.pkl")
    trainer.train(f"checkpoints/checkpoint_{configs.model}.chk")
    print("###########Finish###########")


if __name__ == "__main__":
    main()
