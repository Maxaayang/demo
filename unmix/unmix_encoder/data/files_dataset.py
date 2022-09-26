import librosa
import math
import numpy as np
import unmix_encoder.utils.dist_adapter as dist
from torch.utils.data import Dataset
from unmix_encoder.utils.dist_utils import print_all
from unmix_encoder.utils.io import get_duration_sec, load_audio
import torch as t


class FilesAudioDataset(Dataset):
    def __init__(self, hps):
        super().__init__()
        self.sr = hps.sr
        self.channels = hps.channels
        self.min_duration = hps.min_duration or math.ceil(
            hps.sample_length / hps.sr)
        self.max_duration = hps.max_duration or math.inf
        self.sample_length = hps.sample_length
        assert hps.sample_length / \
            hps.sr < self.min_duration, f'Sample length {hps.sample_length} per sr {hps.sr} ({hps.sample_length / hps.sr:.2f}) should be shorter than min duration {self.min_duration}'
        self.aug_shift = hps.aug_shift
        self.encoder = hps.encoder
        self.channel = hps.channel
        self.init_dataset(hps)

    def filter(self, files, durations):
        # Remove files too short or too long
        keep = []
        for i in range(len(files)):
            if durations[i] / self.sr < self.min_duration:
                continue
            if durations[i] / self.sr >= self.max_duration:
                continue
            keep.append(i)
        print_all(
            f'self.sr={self.sr}, min: {self.min_duration}, max: {self.max_duration}')
        print_all(f"Keeping {len(keep)} of {len(files)} files")
        self.files = [files[i] for i in keep]
        self.durations = [int(durations[i]) for i in keep]
        self.cumsum = np.cumsum(self.durations)

    def init_dataset(self, hps):
        # Load list of files and starts/durations
        files = librosa.util.find_files(f'{hps.audio_files_dir}', [
                                        'mp3', 'opus', 'm4a', 'aac', 'wav'])
        print_all(f"Found {len(files)} files. Getting durations")
        cache = dist.get_rank() % 8 == 0 if dist.is_available() else True
        durations = np.array([get_duration_sec(
            file, cache=cache) * self.sr for file in files])  # Could be approximate
        self.filter(files, durations)

    def get_index_offset(self, item):
        # For a given dataset item and shift, return song index and offset within song
        half_interval = self.sample_length//2
        shift = np.random.randint(-half_interval,
                                  half_interval) if self.aug_shift else 0
        # Note we centred shifts, so adding now
        offset = item * self.sample_length + shift
        midpoint = offset + half_interval
        assert 0 <= midpoint < self.cumsum[-1], f'Midpoint {midpoint} of item beyond total length {self.cumsum[-1]}'
        # index <-> midpoint of interval lies in this song
        index = np.searchsorted(self.cumsum, midpoint)
        # start and end of current song
        start, end = self.cumsum[index -
                                 1] if index > 0 else 0.0, self.cumsum[index]
        assert start <= midpoint <= end, f"Midpoint {midpoint} not inside interval [{start}, {end}] for index {index}"
        if offset > end - self.sample_length:  # Going over song
            offset = max(start, offset - half_interval)  # Now should fit
        elif offset < start:  # Going under song
            offset = min(end - self.sample_length, offset +
                         half_interval)  # Now should fit
        assert start <= offset <= end - \
            self.sample_length, f"Offset {offset} not in [{start}, {end - self.sample_length}]. End: {end}, SL: {self.sample_length}, Index: {index}"
        offset = offset - start
        return index, offset

    """
    def audio_preprocess(self, x):
        # Extra layer in case we want to experiment with different preprocessing
        # For two channel, blend randomly into mono (standard is .5 left, .5 right)

        # x: NTC
        x = x.float()
        if x.shape[-1] == 2:
            # if hps.aug_blend:
            #    mix = t.rand((x.shape[0], 1), device=x.device)  # np.random.rand()
            # else:
            mix = 0.5
            x = (mix*x[:, :, 0]+(1-mix)*x[:, :, 1])
        elif x.shape[-1] == 1:
            x = x[:, :, 0]
        else:
            assert False, f'Expected channels . Got unknown {x.shape[-1]} channels'

        # x: NT -> NTC
        x = x.unsqueeze(2)
        return x

    
    def get_vectors(self, filename, sr, offset, duration):

        filename_encoder = filename.replace("_0", "_2")
        data_encoder, sr = load_audio(filename_encoder, sr=sr,
                                      offset=offset, duration=duration)
        data_encoder = t.tensor(data_encoder.T)
        #data_encoder = data_encoder.to('cuda', non_blocking=True)
        # x_in = data_encoder  # self.audio_preprocess(data_encoder)

        _, _, _, vectors = self.vqvae(x_in, **self.forw_kwargs)

        return vectors
    """

    def get_song_chunk(self, index, offset, test=False):
        filename, total_length = self.files[index], self.durations[index]
        data, sr = load_audio(filename, sr=self.sr,
                              offset=offset, duration=self.sample_length)
        assert data.shape == (
            self.channels, self.sample_length), f'Expected {(self.channels, self.sample_length)}, got {data.shape}'
        if self.encoder:
            filename_encoder = filename.replace("_0", self.channel)
            data_encoder, sr = load_audio(filename_encoder, sr=self.sr,
                                          offset=offset, duration=self.sample_length)
            return data.T, data_encoder.T
        else:
            return data.T

    def get_item(self, item, test=False):
        index, offset = self.get_index_offset(item)
        return self.get_song_chunk(index, offset, test)

    def __len__(self):
        return int(np.floor(self.cumsum[-1] / self.sample_length))

    def __getitem__(self, item):
        return self.get_item(item)
