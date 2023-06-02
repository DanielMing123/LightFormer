# Copyright (c) 2021 Li Auto Company. All rights reserved.

import argparse
import os
import shutil
import time
import warnings
import sys
sys.path.append('.')
import pytorch_lightning as pl
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from models.light_former import LightFormerPredictor
warnings.filterwarnings("ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*")


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='eval intention network.')

    parser.add_argument('-cfg', '--config', type=str, default='', required=True, help='config file')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('-log', '--log_dir', type=str, default='./log', help='log directory')
    parser.add_argument('-g', '--gpu', type=int, default=1, help='0: cpu, n: n num of gpus, -1: all gpus')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Must set seed everything in multi node multi gpus training
    seed: int = 42  # Magic number for deep leanrning
    seed_everything(seed)

    args: argparse.ArgumentParser = parse_args()

    # load config
    config_file = args.config
    print(f'Using config: {config_file}')
    with open(config_file, 'rb') as f:
        config = yaml.safe_load(f)

    # set GPU device
    gpu_num = args.gpu
    if gpu_num == 0:
        print('Gpu not specified, exit normally')
        exit(0)

    # create logger
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_logger = TensorBoardLogger(log_dir, name=config['model_name'])

    # create test result dir
    if 'test_result_pkl_dir' in config['test']:
        test_result_pkl_dir = config['test']['test_result_pkl_dir']
    if test_result_pkl_dir is None:
        print('Test result dir not specified, exit normally')
    if os.path.exists(test_result_pkl_dir):
        output_database_folder_tmp = test_result_pkl_dir + "_" + time.strftime("%Y-%m-%d_%H:%M:%S",
                                                                               time.localtime(time.time()))
        print(f'Result folder already exists, Moving to {output_database_folder_tmp}')
        shutil.move(test_result_pkl_dir, output_database_folder_tmp)
    if not os.path.exists(test_result_pkl_dir):
        os.makedirs(test_result_pkl_dir)

 
    predictor = LightFormerPredictor(config=config)

    # set checkpoint path
    checkpoint_file = args.checkpoint
    if checkpoint_file is None:
        print('Not checkpoint exit eval')
        exit(0)
    print(f'Using checkpoint: {checkpoint_file}')

    trainer = pl.Trainer(
        gpus=gpu_num,
        logger=tb_logger,
        sync_batchnorm=True,
        accelerator="gpu",
        strategy=DDPPlugin(find_unused_parameters=False),
        replace_sampler_ddp=True,
        deterministic=True,
    )

    trainer.test(predictor, ckpt_path=checkpoint_file)
