# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

import multiprocessing as mp

import pprint
import yaml

from src.utils.distributed import init_distributed
from src.train import main as app_main

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
    '--mock', action='store_true',
    help='Use mock random tensors for training')
parser.add_argument(
    '--mock-epochs', type=int, default=None,
    help='Number of epochs to run in mock mode')
parser.add_argument(
    '--mock-iters', type=int, default=10,
    help='Number of training iterations/batches per epoch in mock mode')
parser.add_argument(
    '--optimized_code', action='store_true',
    help='Enable all optimized performance features')
parser.add_argument(
    '--opt_sdpa', action='store_true',
    help='Enable FlashAttention (SDPA)')
parser.add_argument(
    '--opt_compile', action='store_true',
    help='Enable torch.compile model compilation')
parser.add_argument(
    '--opt_fused_adamw', action='store_true',
    help='Enable Fused AdamW optimizer')
parser.add_argument(
    '--opt_dataloader', action='store_true',
    help='Enable optimized dataloader and fast mask generation')


def process_main(rank, fname, world_size, devices, mock=False, mock_epochs=None, mock_iters=10, 
                 optimized_code=False, opt_sdpa=False, opt_compile=False, opt_fused_adamw=False, opt_dataloader=False):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank].split(':')[-1])

    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    # -- Inject mock options
    if mock:
        if 'meta' not in params:
            params['meta'] = {}
        params['meta']['mock'] = True
        params['meta']['mock_epochs'] = mock_epochs
        params['meta']['mock_iters'] = mock_iters

    if 'meta' not in params:
        params['meta'] = {}
    
    # If general optimized_code flag is set, enable all individual optimizations
    if optimized_code:
        opt_sdpa = True
        opt_compile = True
        opt_fused_adamw = True
        opt_dataloader = True
        params['meta']['optimized_code'] = True

    params['meta']['opt_sdpa'] = opt_sdpa
    params['meta']['opt_compile'] = opt_compile
    params['meta']['opt_fused_adamw'] = opt_fused_adamw
    params['meta']['opt_dataloader'] = opt_dataloader

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')
    app_main(args=params)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.mock and args.mock_epochs is None:
        parser.error('--mock requires --mock-epochs to be specified')

    num_gpus = len(args.devices)
    mp.set_start_method('spawn')

    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, args.mock, args.mock_epochs, args.mock_iters, 
                  args.optimized_code, args.opt_sdpa, args.opt_compile, args.opt_fused_adamw, args.opt_dataloader)
        ).start()
