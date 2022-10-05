import torch
import random
import numpy as np
from loaddata import SequenceMIDI
from model import MyModel
from utils import GetNoteSequence, CreateMIDIInstrumennt
from pretty_midi import PrettyMIDI
from tqdm import tqdm


sample_file_name = "sample.mid"
output_file_name = "output10.mid"
save_model_name = "model10.pth"
predict_length = 128
sequence_lenth = 25


def WeightedRandom(weight, k=100000) -> int:
    sum = int(0)
    for w in weight:
        sum += int(k*w)
    x = random.randint(1, sum)
    sum = 0
    for id, w in enumerate(weight):
        sum += int(k*w)
        if sum >= x:
            return id
    return


def PredictNextNote(model: MyModel(), input: np.ndarray):
    model.eval()
    with torch.no_grad():
        input = torch.tensor(input, dtype=torch.float32).unsqueeze(0)
        pred = model(input)
        pitch = WeightedRandom(np.squeeze(pred['pitch'], axis=0))
        step = np.maximum(np.squeeze(pred['step'], axis=0), 0)
        duration = np.maximum(np.squeeze(pred['duration'], axis=0), 0)
    return pitch, float(step), float(duration)


model = MyModel()
model.load_state_dict(torch.load(save_model_name))

sample_data = SequenceMIDI(sample_file_name, sequence_lenth)

cur = sample_data.getendseq()
res = []
prev_start = 0
for i in tqdm(range(predict_length)):
    pitch, step, duration = PredictNextNote(model, cur)
    res.append([pitch, step, duration])
    cur = cur[1:]
    cur = np.append(cur, [[pitch, step, duration]], axis=0)
    prev_start += step

pm_output = PrettyMIDI()
pm_output.instruments.append(
    CreateMIDIInstrumennt(res, "Acoustic Grand Piano"))
pm_output.write(output_file_name)
