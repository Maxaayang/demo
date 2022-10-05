import glob
import numpy as np
from pretty_midi import PrettyMIDI
from torch.utils.data import Dataset
from tqdm import tqdm
from utils import GetNoteSequence


class SequenceMIDI(Dataset):
    def __init__(self, files, seq_len, max_file_num=None):
        notes = None
        filenames = glob.glob(files)
        print(f"Find {len(filenames)} files.")
        if max_file_num is None:
            max_file_num = len(filenames)
        print(f"Reading {max_file_num} files...")
        for f in tqdm(filenames[:max_file_num]):
            pm = PrettyMIDI(f)
            instrument = pm.instruments[0]
            new_notes = GetNoteSequence(instrument)
            new_notes /= [128.0, 1.0, 1.0]
            if notes is not None:
                notes = np.append(notes, new_notes, axis=0)
            else:
                notes = new_notes

        self.seq_len = seq_len
        self.notes = np.array(notes, dtype=np.float32)

    def __len__(self):
        return len(self.notes)-self.seq_len

    def __getitem__(self, idx) -> (np.ndarray, dict):
        label_note = self.notes[idx+self.seq_len]
        label = {
            'pitch': (label_note[0]*128).astype(np.int64), 'step': label_note[1], 'duration': label_note[2]}
        return self.notes[idx:idx+self.seq_len], label

    def getendseq(self) -> np.ndarray:
        return self.notes[-self.seq_len:]
