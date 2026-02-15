import argparse
import time
import logging
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import SIC_dataset
from trainer import Trainer
from config import configs


def setup_logging(log_file):
    """
    设置log
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger()


def load_model_configs(config_file):
    """
    加载pkl文件参数
    """
    with open(config_file, "rb") as config_file:
        return pickle.load(config_file)


def create_parser():
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument(
        "-ts",
        "--start_time",
        type=int,
        required=True,
        help="Starting time (six digits, YYYYMMDD)",
    )
    parser.add_argument(
        "-te",
        "--end_time",
        type=int,
        required=True,
        help="Ending time (six digits, YYYYMMDD)",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    log_file = f"{configs.test_results_path}/test_{configs.model}.log"
    logger = setup_logging(log_file)
    logger.info(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logger.info("###### Start testing! ########\n")
    logger.info(
        f"Arguments:\n\
        start time: {args.start_time}\n\
        end time: {args.end_time}\n\
        output_dir: {configs.test_results_path}\n\
        full_data_path: {configs.full_data_path}\n"
    )
    model_configs = load_model_configs(f"pkls/train_config_{configs.model}.pkl")
    model_configs.model = configs.model
    logger.info(model_configs.__dict__)

    dataset_test = SIC_dataset(
        configs.full_data_path,
        args.start_time,
        args.end_time,
        model_configs.input_gap,
        model_configs.input_length,
        model_configs.pred_shift,
        model_configs.pred_gap,
        model_configs.pred_length,
        samples_gap=1,
    )

    dataloader_test = DataLoader(dataset_test, shuffle=False)

    logger.info("\ntesting...")
    tester = Trainer(model_configs)
    try:
        tester.network.load_state_dict(
            torch.load(f"checkpoints/checkpoint_{configs.model}.chk")["net"]
        )
    except FileNotFoundError:
        logger.error(
            "\nModel file 'checkpoint.chk' not found. Please make sure the file exists."
        )
        return

    mse, rmse, mae, nse, psnr, loss, bacc = tester.vali(
        dataloader_test, torch.from_numpy(np.load("arctic_mask.npy"))
    )
    logger.info(
        f"\nmse: {mse:.5f}, rmse: {rmse:.5f}, mae: {mae:.5f}, nse: {nse:.5f} psnr: {psnr:.5f}, loss: {loss:.5f}, bacc: {bacc:.5f}"
    )

    sic_pred = tester.test(dataloader_test)

    logger.info(f"\nsaving output to {configs.test_results_path}")
    np.save(
        f"{configs.test_results_path}/sic_pred_{configs.model}.npy",
        sic_pred.cpu().numpy(),
    )
    np.save(f"{configs.test_results_path}/inputs.npy", dataset_test.GetInputs())
    np.save(f"{configs.test_results_path}/targets.npy", dataset_test.GetTargets())
    np.save(f"{configs.test_results_path}/times.npy", dataset_test.GetTimes())
    logger.info("\n###### End of test! ########\n")


if __name__ == "__main__":
    main()

