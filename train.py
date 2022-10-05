import torch
import numpy as np
import glob
from torch.utils.data import DataLoader
from dataload import SequenceMIDI
from model import VQVAE
from tqdm import tqdm
from util import *
# from preprocess_midi import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# filenames = glob.glob(BASE_PATH)
# max_file_num = len(filenames)

# trainning_data = SequenceMIDI(
#     BASE_PATH, sequence_lenth, max_file_num=max_file_num)
trainning_data = SequenceMIDI(
    BASE_PATH, sequence_lenth, max_file_num)
print(f"Read {len(trainning_data)} sequences.")
loader = DataLoader(trainning_data, batch_size=batch_size)
# print("loader length: ", len(loader))
# print("loader ", loader)

# for X, y in loader:
#     print(f"X: {X.shape} {X.dtype}")
#     print(f"y: {y}")
#     break

model = VQVAE(in_channels, embedding_dim, num_embeddings).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model)

print("Start trainning...")
best_loss = 0.0
last_sum_loss = 0.0
size = len(loader.dataset)
map = {}
for t in range(epochs):
    model.train()
    sum_loss = 0.0
    print(f"Epoch {t+1}\n-----------------")
    for batch,(X, y) in enumerate(tqdm(loader)):
        X = torch.tensor(X)
        X= X.to(device)
        for feat in y.keys():
            y[feat]=y[feat].to(device)
        # print("train X ", X)
        # print("X.shape", X.shape)
        index, pred, input, loss = model(X)
        sum_loss += loss.item()
        # index = torch.tensor( [item.cpu().detach().numpy() for item in index] )
        index = index.cpu().numpy()
        for i in range(len(index)):
            if (map.__contains__(index[i])):
                map[index[i]] += 1
            else:
                map[index[i]] = 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # avg_loss /= len(loader)
        # avg_loss /= len(loader)
    if (sum_loss < last_sum_loss):
        print(f"best loss = {best_loss}, last loss = {last_sum_loss}, loss = {sum_loss}, diff loss = {best_loss - sum_loss}, per = {((best_loss - sum_loss) / sum_loss * 100)}")
        last_sum_loss = sum_loss
    else:
        best_loss = sum_loss
        last_sum_loss = sum_loss
        print(f"loss = {sum_loss}")
    if (t+1) % epochs == 0:
        torch.save(model.state_dict, "./model/model_vae%d.pth" % (t+1))
print(sorted(map.items(),key=lambda s:s[1]))
print("Done!")

# torch.save(model.state_dict(), SAVE_MODEL_NAME)
print(f"Saved PyTorch Model State to {SAVE_MODEL_NAME}")
