# Copyright (c) 2024 Alibaba Inc
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function
import argparse
import datetime
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from copy import deepcopy
import torch
import torch.distributed as dist
import deepspeed
import glob
import os
from hyperpyyaml import load_hyperpyyaml
from torch.cuda.amp import GradScaler, autocast
from torch.distributed.elastic.multiprocessing.errors import record
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from inspiremusic.utils.executor import Executor
from inspiremusic.utils.train_utils import (
    init_distributed,
    init_dataset_and_dataloader,
    init_optimizer_and_scheduler,
    init_summarywriter, save_model,
    wrap_cuda_model, check_modify_and_save_config)


def get_args():
    parser = argparse.ArgumentParser(description='training your network')
    parser.add_argument('--train_engine',
                        default='torch_ddp',
                        choices=['torch_ddp', 'deepspeed'],
                        help='Engine for paralleled training')
    parser.add_argument('--model', required=True, help='model which will be trained')
    parser.add_argument('--config', required=True, help='config file')
    parser.add_argument('--train_data', required=True, help='train data file')
    parser.add_argument('--cv_data', required=True, help='cv data file')
    parser.add_argument('--checkpoint', help='checkpoint model')
    parser.add_argument('--model_dir', required=True, help='save model dir')
    parser.add_argument('--tensorboard_dir',
                        default='tensorboard',
                        help='tensorboard log dir')
    parser.add_argument('--ddp.dist_backend',
                        dest='dist_backend',
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='distributed backend')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='number of subprocess workers for reading')
    parser.add_argument('--prefetch',
                        default=100,
                        type=int,
                        help='prefetch number')
    parser.add_argument('--pin_memory',
                        action='store_true',
                        default=False,
                        help='Use pinned memory buffers used for reading')
    parser.add_argument('--deepspeed.save_states',
                        dest='save_states',
                        default='model_only',
                        choices=['model_only', 'model+optimizer'],
                        help='save model/optimizer states')
    parser.add_argument('--timeout',
                        default=30,
                        type=int,
                        help='timeout (in seconds) of inspiremusic_join.')
    parser.add_argument('--fp16',
                          action='store_true',
                          default=False,      
                          help='Enable fp16 mixed precision training')
    parser.add_argument('--lora',
                          action='store_true',
                          default=False,      
                          help='Enable LoRA training')
    parser.add_argument('--lora_rank',
                          default=4,
                          type=int,
                          help='LoRA rank')   
    parser.add_argument('--lora_alpha',     
                          default=16,
                          type=int,
                          help='LoRA alpha')  
    parser.add_argument('--lora_dropout',   
                          default=0.1,
                          type=float,
                          help='LoRA dropout rate')
    parser.add_argument('--lora_target_modules',
                          nargs='+',
                          default=["k_proj","v_proj"],
                          help='Target modules to apply LoRA (e.g., ["q_proj", "v_proj"])')

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args


@record
def main():
    args = get_args()
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')

    override_dict = {k: None for k in ['llm', 'flow', 'hift'] if k != args.model}
    with open(args.config, 'r') as f:
        configs = load_hyperpyyaml(f, overrides=override_dict)
    configs['train_conf'].update(vars(args))

    # Init env for ddp
    init_distributed(args)

    # Get dataset & dataloader
    train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
        init_dataset_and_dataloader(args, configs)

    # Do some sanity checks and save config to arsg.model_dir
    configs = check_modify_and_save_config(args, configs)

    # Tensorboard summary
    writer = init_summarywriter(args)

    # load checkpoint
    model = configs[args.model]

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    else:
        # Find and load the latest checkpoint
        checkpoint_files = glob.glob(os.path.join(args.model_dir, '*.pt'))

        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            logging.info(f"Loaded latest checkpoint from {latest_checkpoint}")

            model.load_state_dict(torch.load(latest_checkpoint, map_location='cpu'))

    if args.lora:
        logging.info("Applying LoRA to the model...")
        if not args.lora_target_modules:
            raise ValueError("No target modules specified for LoRA. Please provide --lora_target_modules.")
        lora_config = LoraConfig(
            task_type="CAUSAL_LM",  # Change to appropriate task type
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.lora_target_modules
        )
        model.llm.model = get_peft_model(model.llm.model, lora_config)
        # Optionally freeze the base model
    else:
        logging.info("LoRA is not enabled. Training the full model.")

    # Dispatch model from cpu to gpu
    model = wrap_cuda_model(args, model)

    # Get optimizer & scheduler
    model, optimizer, scheduler = init_optimizer_and_scheduler(args, configs, model)

    # Initialize AMP for torch_ddp if fp16 is enabled
    scaler = None
    if args.fp16:
        scaler = GradScaler()
        logging.info("Initialized AMP GradScaler for mixed precision training.")

    # Save init checkpoints
    info_dict = deepcopy(configs['train_conf'])

    # Get executor
    executor = Executor()

    # Start training loop
    for epoch in range(info_dict['max_epoch']):
        executor.epoch = epoch
        train_dataset.set_epoch(epoch)
        dist.barrier()
        group_join = dist.new_group(backend="gloo", timeout=datetime.timedelta(seconds=args.timeout))
        executor.train_one_epoch(model, optimizer, scheduler, train_data_loader, cv_data_loader, writer, info_dict, group_join, scaler=scaler)
        dist.destroy_process_group(group_join)

if __name__ == '__main__':
    main()
