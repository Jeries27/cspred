
import os
import argparse
import pickle
import numpy as np
import torch
import descriptors
import yaml
from types import SimpleNamespace

from experiment import Experiment
# from models import RegressorTrainer

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{key: dict_to_namespace(value) for key, value in d.items()})
    return d

def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config_dict = yaml.safe_load(file)
    return dict_to_namespace(config_dict)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="mace_config.yaml", help='path to YAML config file')
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get("LOCAL_RANK", 0)),
                        help='Local rank for distributed training')
    return parser.parse_args()


def main():
    args = parse_args()

    distributed = False
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        distributed = True
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        print(f"Distributed training enabled on local rank {args.local_rank}")

    config = load_config(f"configs/{args.config}")

    descriptor_config = config.descriptor
    descriptor = getattr(descriptors, descriptor_config.name)()
    descriptor.generate_descriptors(**vars(descriptor_config.generate_descriptors_params))


if __name__ == "__main__":
    main()
