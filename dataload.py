import glob
import numpy as np
from pretty_midi import PrettyMIDI
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import GetNoteSequence
from preprocess_midi import *
from util import *
import pickle


class SequenceMIDI(Dataset):
    def __init__(self, files, seq_len, max_file_num=None):
        notes = None
        filenames = glob.glob(files)
        print(f"Find {len(filenames)} files.")
        if max_file_num is None:
            max_file_num = len(filenames)
        print(f"Reading {max_file_num} files...")
        # for f in tqdm(filenames[:max_file_num]):
        #     # pm = PrettyMIDI(f)
        #     # instrument = pm.instruments[0]  # 乐器列表
        #     # 读入音符列表, 将其处理为需要的三个特征pitch, step, duration, 并转换为numpy数组
        #     # instrument 音轨, notes 取所有音符元素
        #     # print("instrument ", instrument.notes)
        #     # # 每个音符的音高, 距离上一个音符开始的时间, 持续时间
        #     # new_notes = GetNoteSequence(instrument)
        #     # new_notes /= [128.0, 1.0, 1.0]
        #     # if notes is not None:
        #     #     notes = np.append(notes, new_notes, axis=0)
        #     # else:
        #     #     notes = new_notes
        #     print("f ", f)
        #     if preprocess_midi(f) == None:
        #         continue
        #     piano_roll, bar_indices, pm_old = preprocess_midi(f)
        #     # (16, 64, 89)
        #     if piano_roll.shape == (0,):
        #         continue
        #     # print("piano_roll shape: ", piano_roll.shape)
        #     # (1024, 89)
        #     # piano_roll_new = np.reshape(piano_roll,(-1,piano_roll.shape[-1]))
        #     piano_roll_new = piano_roll
        #     if notes is not None:
        #         notes = np.append(notes, piano_roll_new, axis=0)
        #     else:
        #         notes = piano_roll_new
        #     seq_len = np.append(seq_len, piano_roll_new.shape[0])
        #     # pm_new = util.roll_to_pretty_midi(piano_roll_new,pm_old)

        # # seq_len = pickle.load(open('../seq_lenth', 'rb'))
        # seq_len = seq_len.astype(np.int64)
        self.seq_len = pickle.load(open('../seq_len', 'rb'))
        self.notes = pickle.load(open('../leave_notes', 'rb'))

        # 有声音的, 空的, start, 有声音的, 空的, start, 第一个有声音的是 0~72, 第二个是压缩到了 12., melody, bass
        # self.notes = np.array(notes, dtype=np.float32)
        # pickle.dump()

    def __len__(self):
        # print("note length: ", len(str(self.notes)))
        # print("__len__ ", len(str(self.notes))-self.seq_len)
        # return len(str(self.notes))-self.seq_len
        return len(str(self.notes))

    def __getitem__(self, idx) -> (np.ndarray, dict):
        # label_note = self.notes[idx+self.seq_len]
        # TODO 这里的label需要修改
        # label = {
        #     'pitch': (label_note[0]*128).astype(np.int64), 'step': label_note[1], 'duration': label_note[2]}
        # label = {
        #     'melody': (label_note[:73]).astype(np.int64), 'melody_empty': label_note[73], 'melody_start': label_note[74], \
        #         'bass': (label_note[75:87]).astype(np.int64), 'bass_empty': label_note[87], 'bass_start': label_note[88]}
        return self.notes[self.seq_len[idx]:self.seq_len[idx + 1]]

    def getendseq(self) -> np.ndarray:
        print("getendseq ", self.notes[-self.seq_len:])
        return self.notes[-self.seq_len:]
