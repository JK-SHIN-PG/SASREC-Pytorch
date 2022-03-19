import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
import numpy as np
import pandas as pd
from utils import adjust_seq_length
from tqdm import tqdm


class Dataset(Dataset):
    def __init__(self, labels):
        self.labels = torch.tensor(labels, dtype=torch.int64)

    def __getitem__(self, idx):
        #x = torch.tensor(self.data[idx], dtype = torch.float32)
        label = self.labels[idx]
        #label = torch.tensor(self.labels[idx], dtype = torch.int64)
        return label

    def __len__(self):
        return len(self.labels)


def Data_load(index_label, BATCH_SIZE):

    train_dataset = Dataset(index_label)
    train_loader = DataLoader(train_dataset, shuffle=True,
                              batch_size=BATCH_SIZE,  num_workers=8, persistent_workers=True)
    print("Data_loading : Complete!")
    return index_label.shape, train_loader


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
