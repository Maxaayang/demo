import torch
import numpy as np
import glob
from torch.utils.data import DataLoader
from dataload import SequenceMIDI
from model import VQVAE
from tqdm import tqdm
from util import *
import pickle
# from preprocess_midi import *

trainning_data = SequenceMIDI(
    BASE_PATH, sequence_lenth, max_file_num)
pickle.dump(trainning_data.notes, open('../leave_notes', 'wb'))
pickle.dump(trainning_data.seq_len, open('../seq_len', 'wb'))
