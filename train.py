# %%
from model import *
from utils import *
from Dataloader import *
import torch
import os

if __name__ == "__main__":
    DATA_SOURCE_PATH = "./Data/retail/"
    MODEL_STORAGE_PATH = "./Saved_file/"
    GPU_NUM = "0"
    save = "example"
    #ensure_path(MODEL_STORAGE_PATH + save)
    seed = 2022
    set_seed(seed, cuda=True)

    device = torch.device(
        f"cuda:{GPU_NUM}") if torch.cuda.is_available() else torch.device("cpu")

    seqs = collect_data(DATA_SOURCE_PATH, "train")
# %%
    # Hyper-parameter setting
    num_item
    model = SASREC(num_item, hidden_dim, seq_len,
                   dropout_rate, num_blocks, num_heads)
