import argparse
import os
import json
import sys
sys.path.append('.')
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.plugins import DDPPlugin
from models.light_former import LightFormerPredictor

def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='train intention network.')

    parser.add_argument('-cfg', '--config', type=str, default='', required=True, help='config file')
    parser.add_argument('-ckpt', '--checkpoint', type=str, default=None, help='checkpoint file')
    parser.add_argument('-save', '--saved_model', type=str, default='./result', help='path to save model')
    parser.add_argument('-log', '--log_dir', type=str, default='./log', help='log directory')
    parser.add_argument('-g', '--gpu', type=int, default=1, help='0: cpu, n: n num of gpus, -1: all gpus')
    parser.add_argument('-n', '--node', type=int, default=1, help='num of nodes across multi machines')
    parser.add_argument('--resume_weight_only',
                        dest='resume_weight_only',
                        action='store_true',
                        default=False, # False
                        help='resume only weights from chekpoint file')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Must set seed everything in multi node multi gpus training
    seed: int = 42
    seed_everything(seed)

    args: argparse.ArgumentParser = parse_args()
    print("\nargs:", args)

    # load config
    config = None
    config_file = args.config
    print(f'\nUsing config: {config_file}')
    with open(config_file, 'r') as f:
        config = json.load(f)
    print("\nconfig:", config)

    # set GPU device
    gpu_num = args.gpu
    if gpu_num == 0:
        print('Gpu not specified, exit normally')
        exit(0)

    # set node
    node_num = args.node
    if node_num < 1:
        print('Node num must be greater than 0')
        exit(0)
    
    # create logger
    log_dir = args.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    tb_logger = TensorBoardLogger(log_dir, name=config['model_name'])

    # set checkpoint callback to save best val_error_rate and last epoch
    saved_path = args.saved_model
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=saved_path,
                                          filename="checkpoint-{epoch:04d}-{val_loss:.5f}",
                                          save_weights_only=False,
                                          mode='min',
                                          save_top_k=10)
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True)

    # set checkpoint path
    checkpoint_file = args.checkpoint
    if checkpoint_file is not None:
        print(f'Using checkpoint: {checkpoint_file}')

    resume_weight_only = args.resume_weight_only
    if resume_weight_only:
        predictor = LightFormerPredictor.load_from_checkpoint(config=config,
                                                                   checkpoint_path=checkpoint_file,
                                                                   strict=True)
    else:
        predictor = LightFormerPredictor(config=config)

    
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback, lr_monitor],
        gpus=gpu_num,
        auto_select_gpus=True,
        num_nodes=node_num,
        max_epochs=config['training']['epoch'],
        # val_check_interval=config['validation']['check_interval'],
        # limit_val_batches=config['validation']['limit_batches'],
        logger=tb_logger,
        # log_every_n_steps=config['log_every_n_steps'],
        gradient_clip_val=config['optim']['gradient_clip_val'],
        gradient_clip_algorithm=config['optim']['gradient_clip_algorithm'],
        sync_batchnorm=True,
        accelerator="gpu",
        strategy=DDPPlugin(find_unused_parameters=False),
        replace_sampler_ddp=True,
        deterministic=False, 
        auto_lr_find=True
    )

    if resume_weight_only:
        trainer.fit(predictor)
    else:
        trainer.fit(predictor, ckpt_path=checkpoint_file)