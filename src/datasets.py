
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from cprint import pprint_color
from build_graph import TargetSubseqs
from param import args
from typing import List, Dict, Tuple
from data_aug import Crop, Mask, Reorder, Insert, Substitute, SubsetSplit
class SRDataset(Dataset):
    def __init__(
        self,
        user_seq: List[List[int]],
        data_type: str = "train",
    ) -> None:

        self.user_seq =  user_seq
        self.id_seq = [seq[0] for seq in user_seq]
        self.data_type = data_type
        self.max_len: int = args.max_seq_length
        # create target item sets
        target_item_subseq = TargetSubseqs(args.subseqs_path, args.target_subseqs_path, args.subseqs_target_path)
        self.train_tag: Dict[int, Dict[Tuple[int, ...], Tuple[int, List[int]]]] = target_item_subseq.load_target_subseqs_dict(args.subseqs_path, args.target_subseqs_path, args.time_subseqs_path)

        self.get_pad_user_seq()

    def __getitem__(self, index: int):

        user_id = index
        # * new loader_type: 1. use global pad sequence 2. drop target_pos sample 3. remove test noise interactions
        pad_user_seq = self.pad_user_seq[index]
        if self.data_type in ["train", "graph"]:
            user_seq = self.user_seq[index]
            user_ids = self.id_seq[index]
            input_ids = pad_user_seq[:-3]
            target_pos = pad_user_seq[1:-2]
            answer = target_pos[-1]

            subseqs_id = (
                args.subseq_id_map[self.pad_origin_map[pad_user_seq][:-3]] if self.data_type == "graph" else []
            )

            return (
                torch.tensor(subseqs_id, dtype=torch.long),
                torch.tensor(user_ids, dtype=torch.long),
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )
        elif self.data_type == "valid":
            # 保证一共是50个
            input_ids = pad_user_seq[1:-2]
            target_pos = pad_user_seq[2:-1]
            answer = [pad_user_seq[-2]]

        else:
            input_ids = pad_user_seq[2:-1]
            target_pos = pad_user_seq[3:]
            answer = [pad_user_seq[-1]]
        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target_pos, dtype=torch.long),
            torch.tensor(answer, dtype=torch.long),
        )


    def __len__(self):
        """consider n_view of a single sequence as one sample"""
        return len(self.user_seq)

    def get_pad_user_seq(self):
        """Prepare the padding in advance, so there's no need to do it again during each __getitem__()  of the Dataloader."""
        max_len = self.max_len + 3
        padded_user_seq = np.zeros((len(self.user_seq), max_len), dtype=int)

        for i, seq in enumerate(self.user_seq):
            padded_user_seq[i, -min(len(seq), max_len):] = seq[-max_len:]

        self.pad_user_seq = tuple(map(tuple, padded_user_seq))
        user_seq = tuple(map(tuple, self.user_seq))

        self.origin_pad_map = dict(zip(user_seq, self.pad_user_seq))
        self.pad_origin_map = dict(zip(self.pad_user_seq, user_seq))

def build_dataloader(user_seq, loader_type):
    sampler = RandomSampler if loader_type == "train" else SequentialSampler
    pprint_color(f">>> Building {loader_type} Dataloader")
    dataset = SRDataset(user_seq, data_type=loader_type)
    return DataLoader(
        dataset,
        sampler=sampler(dataset),
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=True,
        # persistent_workers=True
    )
