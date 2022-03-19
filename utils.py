import os
import numpy as np
import torch
import shutil
import os.path as osp
from tqdm import tqdm
import random


def ensure_path(path):
    if osp.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def save_model(model, name, path):
    torch.save(model.state_dict(), osp.join(path, name + '.pth'))


def set_seed(seed, cuda=False):
    np.random.seed(seed)
    random.seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    else:
        torch.manual_seed(seed)


def adjust_seq_length(templist):
    if len(templist) == 20:
        pass
    else:
        if len(templist) < 20:
            templist = templist + [3]*(20-len(templist))
        else:
            templist = templist[:20]
    return templist


def get_label_index(label):
    unique_label_list, unique_dic = load_unique_label(
        "./Data/fixed_unique_label.csv")
    L_shape = label.shape
    label_array = np.empty(L_shape)
    for i in tqdm(range(len(label)), desc="Label indexing: "):
        for j in range(len(label.columns)):
            label_array[i][j] = unique_dic[label.iloc[i, j]]
    return label_array, unique_label_list, unique_dic


def load_unique_label(path):
    import pandas as pd
    df = pd.read_csv(path)
    unique_list = df["0"].tolist()
    df['index'] = [i for i in range(len(df))]
    unique_dic = df.set_index("0").to_dict()["index"]
    return unique_list, unique_dic
