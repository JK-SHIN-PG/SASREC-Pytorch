# %%
from model import *
from utils import *
from Dataloader import *
import torch
import torch.nn as nn
import torch.optim as optim
import os


def train(model, iterator, criterion, optimizer, device):
    epoch_loss = 0
    model.train()
    for batch_index, (x, pos, neg) in enumerate(iterator):
        optimizer.zero_grad()
        pos_logit, neg_logit = model(x, pos, neg)
        pos_label = torch.ones(pos_logit.shape, device=device)
        neg_label = torch.zeros(neg_logit.shape, device=device)
        index = np.where(pos != 3)
        loss = criterion(pos_logit[index], pos_label[index]) + \
            criterion(neg_logit[index], neg_label[index])
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

    return epoch_loss / len(iterator)


def eval(model, iterator, criterion, num_item, device):
    model.eval()
    epoch_acc = 0
    with torch.no_grad():
        for batch_index, (x, pos, _) in enumerate(iterator):
            batch_acc = 0
            item_candidate = torch.LongTensor(
                torch.arange(0, num_item).repeat(x.shape[0], 1))
            for s in range(3, seq_len-1):
                input = x[:, :s]
                target = pos[:, s-1]
                pred_logit, pred_label = model.predict(
                    input, item_candidate.to(device))
                index = np.where(target != 3)
                #loss = criterion(pred_logit[index], target[index])
                prediction = torch.argmax(pred_label, dim=1)
                correct = prediction[index] == target[index].to(device)
                accuracy = torch.sum(correct) / len(correct)
                batch_acc += accuracy / len(index[0])
            epoch_acc += batch_acc

    return epoch_acc / len(iterator)


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
    test_seqs = collect_data(DATA_SOURCE_PATH, "test")
    unique_list, unique_dic = load_unique_label(
        "./Data/fixed_unique_label.csv")

    # Hyper-parameter setting
    num_item = len(unique_list)
    embedding_dim = 256
    seq_len = 20
    dropout_rate = 0.2
    num_blocks = 6
    num_heads = 8  # embedding_dim must be divisible by num_heads
    BATCH_SIZE = 256
    lr = 0.0005
    epoch = 10
    # positive answer sequence

    # negative answer sequence # number of sequence is seq_len -1
    neg_sample = negative_sampler(len(seqs), seq_len-1, num_item)

    train_loader = Data_load(seqs, neg_sample, BATCH_SIZE)
    test_loader = Data_load(test_seqs, neg_sample, BATCH_SIZE)

    model = SASREC(num_item, embedding_dim, seq_len-1,
                   dropout_rate, num_blocks, num_heads, device).to(device)

    criterion = nn.BCEWithLogitsLoss()
    criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    item_candidate = torch.LongTensor(
        torch.arange(0, num_item).repeat(BATCH_SIZE, 1))
# %%
    for ep in range(1000):
        loss = train(model, train_loader, criterion, optimizer, device)
        if ep % 10 == 0:
            acc = eval(model, train_loader, criterion, num_item, device)
            print("epoch : {} \t loss(train) : {:.5f} \t acc(train) : {:.5f} \n".format(
                ep, loss, acc))


#    def train(model, train_loader, optimizer):
# %%


# %%
loss, acc = eval(model, train_loader, criterion, num_item, device)


# %%
item_candidate = torch.LongTensor(
    torch.arange(0, num_item).repeat(96, 1))
logit, pred_label = model.predict(x[:, :2], item_candidate.to(device))
target = pos[:, 1]
index = np.where(target != 3)
prediction = torch.argmax(pred_label, dim=1)

correct = prediction[index] == target[index].to(device)
accuracy = torch.sum(correct) / len(correct)
# %%

# %%
# %%
