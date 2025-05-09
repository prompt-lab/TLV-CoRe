import os
import time
from dataclasses import dataclass
from multiprocessing import Value

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.base_datasets import touch_dataset
from open_clip import get_tokenizer
from open_clip.factory import HF_HUB_PREFIX


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value

@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def get_touch_dataset(args):
    dataset = touch_dataset(args)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed else None
    shuffle = sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        # prefetch_factor=2,
        # persistent_workers=True,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=False,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)

def get_data(args, epoch=0):
    data = {}
    # get train data
    if args.do_train:
        if args.train_data.endswith(".json"):
            data[f"{args.clip_type}_pt"] = get_touch_dataset(args)
        else:
            raise NameError

    # get valid data
    if args.do_eval:
        from zero_shot.datasets import get_tag_dataset, get_feeling_dataset
        
        temp_batch_size = args.batch_size
        temp_val_t_cls_data = args.val_t_cls_data

        data_root = "dataset/downstream/"
        data["t_cls"] = []
        cls_mode = ["material", "hard", "rough"]
        if temp_val_t_cls_data == ['Touch_and_Go']:
            for val_t_cls_data in temp_val_t_cls_data:
                for mode in cls_mode:              
                    args.cls_mode = mode
                    args.val_t_cls_data = val_t_cls_data
                    args.touch_data_path = os.path.join(data_root, f'{val_t_cls_data}/{args.cls_mode}') 
                    data['t_cls'].append({val_t_cls_data: get_tag_dataset(args)})
        elif temp_val_t_cls_data == ['feeling']:
            for val_t_cls_data in temp_val_t_cls_data:
                args.val_t_cls_data = val_t_cls_data
                args.touch_data_path = os.path.join(data_root, f'{val_t_cls_data}/data') 
                data['t_cls'].append({val_t_cls_data: get_feeling_dataset(args)})
        
        args.val_t_cls_data = temp_val_t_cls_data
        args.batch_size = temp_batch_size

    return data



