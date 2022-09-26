import glob
import numpy as np
from pretty_midi import PrettyMIDI
from torch.utils.data import Dataset
from tqdm import tqdm
# from utils import GetNoteSequence
import sys
sys.path.append("..")
from demo.preprocess import *
from demo.util import *


class SequenceMIDI(Dataset):
    def __init__(self, files, seq_len, max_file_num=None):
        notes = None
        filenames = glob.glob(files)
        print(f"Find {len(filenames)} files.")
        if max_file_num is None:
            max_file_num = len(filenames)
        print(f"Reading {max_file_num} files...")
        # train_len = int(len(max_file_num) * 0.8)
        # train_dataset = max_file_num[: train_len]
        # test_dataset = max_file_num[train_len: ]
        for f in tqdm(filenames[:max_file_num]):
            # pm = PrettyMIDI(f)
            # instrument = pm.instruments[0]  # 乐器列表
            # 读入音符列表, 将其处理为需要的三个特征pitch, step, duration, 并转换为numpy数组
            # instrument 音轨, notes 取所有音符元素
            # print("instrument ", instrument.notes)
            # # 每个音符的音高, 距离上一个音符开始的时间, 持续时间
            # new_notes = GetNoteSequence(instrument)
            # new_notes /= [128.0, 1.0, 1.0]
            # if notes is not None:
            #     notes = np.append(notes, new_notes, axis=0)
            # else:
            #     notes = new_notes
            print("f ", f)
            if preprocess_midi(f) == None:
                continue
            piano_roll, bar_indices, pm_old = preprocess_midi(f)
            if piano_roll.shape == (0,):
                continue
            print("piano_roll shape: ", piano_roll.shape)
            piano_roll_new = np.reshape(piano_roll,(-1,piano_roll.shape[-1]))
            if notes is not None:
                notes = np.append(notes, piano_roll_new, axis=0)
            else:
                notes = piano_roll_new
            # pm_new = util.roll_to_pretty_midi(piano_roll_new,pm_old)

        self.seq_len = seq_len
        self.notes = np.array(notes, dtype=np.float32)

    def __len__(self):
        # print("note length: ", len(str(self.notes)))
        # print("__len__ ", len(str(self.notes))-self.seq_len)
        return len(str(self.notes))-self.seq_len

    def __getitem__(self, idx) -> (np.ndarray, dict):
        label_note = self.notes[idx+self.seq_len]
        label = {
            'pitch': (label_note[0]*128).astype(np.int64), 'step': label_note[1], 'duration': label_note[2]}
        return self.notes[idx:idx+self.seq_len], label

    def getendseq(self) -> np.ndarray:
        print("getendseq ", self.notes[-self.seq_len:])
        return self.notes[-self.seq_len:]
