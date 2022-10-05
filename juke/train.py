import sys
sys.path.append("..")
# from juke.vqvae import VQVAE
from vqvae import VQVAE
from juke.juke_params import *
import torch
from torch.utils.data import DataLoader
from juke.dataload import SequenceMIDI
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train(model, X):
    model.train()

    # print("Start trainning...")
    model.train()
    X = torch.tensor(X)
    X= X.to(device)
    index, x_out, loss, _metrics = model(X)
    return index, loss, _metrics


trainning_data = SequenceMIDI(
    BASE_PATH, sequence_lenth, max_file_num)

print(f"Read {len(trainning_data)} sequences.")
loader = DataLoader(trainning_data, batch_size=batch_size)
print("loader length: ", len(loader))
print("loader ", loader)

block_kwargs = dict(width=width, depth=depth, m_conv=m_conv, dilation_growth_rate=dilation_growth_rate, \
     dilation_cycle=dilation_cycle, reverse_decoder_dilation=reverse_decoder_dilation)

model = VQVAE(input_shape, levels, downs_t, strides_t, emb_width, l_bins, mu, commit, \
              spectral, multispectral, multipliers, use_bottleneck, **block_kwargs)

# model.state_dict = torch.load("../model/juke_vae20.pth")
optimizer = torch.optim.SGD(model.parameters(), lr)

model = model.to(device)

# print("Start trainning...")
size = len(loader.dataset)
last_sum_loss = 0.0
best_loss = 0.0
map = {}
for epoch in range(epochs):
    model.train()
    sum_loss = 0.0
    print(f"Epoch {epoch+1}\n-----------------")
    for batch,(X, y) in enumerate(tqdm(loader)):
        X = torch.tensor(X)
        X= X.to(device)
        for feat in y.keys():
            y[feat]=y[feat].to(device)
        # print("X.shape", X.shape)
        # pred, input, loss = model(X)
        index, loss, _metrics = train(model, X)
        sum_loss += loss.item()
        index = index[0].cpu().numpy()
        for i in range(len(index)):
            if (map.__contains__(index[i])):
                map[index[i]] += 1
            else:
                map[index[i]] = 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # avg_loss /= len(loader)
    if (sum_loss < last_sum_loss):
        print(f"best loss = {best_loss}, last loss = {last_sum_loss}, loss = {sum_loss}, diff loss = {best_loss - sum_loss}, per = {(best_loss - sum_loss) / sum_loss * 100}")
        last_sum_loss = sum_loss
    else:
        best_loss = sum_loss
        last_sum_loss = sum_loss
        print(f"loss = {sum_loss}")
    # print(f"average loss = {sum_loss / len(loader)}")
    if (epoch+1) % epochs == 0:
        torch.save(model.state_dict, "../model/juke_vae%d.pth" % (epoch+1))
print("Done!")
print(sorted(map.items(),key=lambda s:s[1]))
# print("map", map)