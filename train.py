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
print("loader length: ", len(loader))
print("loader ", loader)

for X, y in loader:
    print(f"X: {X.shape} {X.dtype}")
    print(f"y: {y}")
    break

model = VQVAE(in_channels, embedding_dim, num_embeddings).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model)

print("Start trainning...")
size = len(loader.dataset)
for t in range(epochs):
    model.train()
    avg_loss = 0.0
    print(f"Epoch {t+1}\n-----------------")
    for batch,(X, y) in enumerate(tqdm(loader)):
        X = torch.tensor(X)
        X= X.to(device)
        for feat in y.keys():
            y[feat]=y[feat].to(device)
        # print("train X ", X)
        print("X.shape", X.shape)
        pred, input, loss = model(X)
        avg_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss /= len(loader)
    print(f"average loss = {avg_loss}")
    if (t+1) % 200 == 0:
        torch.save(model, "./model/model%d.pth" % (t+1))
print("Done!")

torch.save(model.state_dict(), SAVE_MODEL_NAME)
print(f"Saved PyTorch Model State to {SAVE_MODEL_NAME}")
