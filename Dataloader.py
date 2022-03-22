import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
import numpy as np
import pandas as pd
from utils import adjust_seq_length
from tqdm import tqdm
import copy


class Dataset(Dataset):
    def __init__(self, data, neg):
        data = torch.tensor(data, dtype=torch.int64)
        self.seq = copy.deepcopy(data[:, :-1])
        self.pos = copy.deepcopy(data[:, 1:])
        self.neg = torch.tensor(neg, dtype=torch.int64)

    def __getitem__(self, idx):

        seq = self.seq[idx]
        pos = self.pos[idx]
        neg = self.neg[idx]

        return seq, pos, neg

    def __len__(self):
        return len(self.seq)


def Data_load(data, neg, BATCH_SIZE):

    train_dataset = Dataset(data, neg)
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=BATCH_SIZE,  num_workers=8, persistent_workers=True)
    print("Data_loading : Complete!")
    return train_loader


def collect_data(PATH, filename):
    df = pd.read_csv(PATH + filename + ".txt", sep="\s",
                     engine="python", header=None)
    seq_idx = df.iloc[:, 0].unique()
    seqs = []
    for idx in tqdm(seq_idx):
        temp = df[df[0] == idx]
        templist = temp[1].to_list()
        seq = adjust_seq_length(templist)
        seqs.append(seq)
    print("Done : Data collection")
    return seqs
