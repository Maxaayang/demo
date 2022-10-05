import torch
import numpy as np
from torch.utils.data import DataLoader
from loaddata import SequenceMIDI
from model import MyModel, MyLoss
from tqdm import tqdm


# note feature: pitch, step, duration
batch_size = 64
sequence_lenth = 25
max_file_num = 1200
epochs = 200
learning_rate = 0.005

loss_weight = [0.1, 20.0, 1.0]

save_model_name = "model.pth"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

trainning_data = SequenceMIDI(
    "maestro-v2.0.0\\*\\*.midi", sequence_lenth, max_file_num=max_file_num)
print(f"Read {len(trainning_data)} sequences.")
loader = DataLoader(trainning_data, batch_size=batch_size)

for X, y in loader:
    print(f"X: {X.shape} {X.dtype}")
    print(f"y: {y}")
    break

model = MyModel().to(device)
loss_fn = MyLoss(loss_weight).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
print(model)
print(loss_fn)

print("Start trainning...")
size = len(loader.dataset)
for t in range(epochs):
    model.train()
    avg_loss = 0.0
    print(f"Epoch {t+1}\n-----------------")
    for batch,(X, y) in enumerate(tqdm(loader)):
        X= X.to(device)
        for feat in y.keys():
            y[feat]=y[feat].to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        avg_loss = avg_loss + loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    avg_loss /= len(loader)
    print(f"average loss = {avg_loss}")
    if (t+1) % 10 == 0:
        torch.save(model.state_dict(), "model%d.pth" % (t+1))
print("Done!")

torch.save(model.state_dict(), save_model_name)
print(f"Saved PyTorch Model State to {save_model_name}")
