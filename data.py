import numpy as np
import os
import tensorflow as tf
import librosa
import IPython.display as ipd
import librosa.display
import matplotlib.pyplot as plt
from util import *

AUTOTUNE = tf.data.experimental.AUTOTUNE

def DatasetLoader(class_):
    music_list = np.array(sorted(os.listdir(BASE_PATH+'/'+class_)))
    train_music_1 = list(music_list[[0,52,19,39,71,12,75,85,3,45,24,46,88]]) #99,10,66,76,41
    train_music_2 = list(music_list[[4,43,56,55,45,31,11,13,70,37,21,78]]) #65,32,53,22,19,80,89,
    TrackSet_1 = [(BASE_PATH)+'/'+class_+'/%s'%(x) for x in train_music_1]
    TrackSet_2 = [(BASE_PATH)+'/'+class_+'/%s'%(x) for x in train_music_2]

    return TrackSet_1, TrackSet_2

def load(file_):
    data_, sampling_rate = librosa.load(file_,sr=3000, offset=0.0, duration=30)
    data_ = data_[:9]
    data_ = data_.reshape(1,9)
    return data_
map_data = lambda filename: tf.compat.v1.py_func(load, [filename], [tf.float32])

TrackSet_1, TrackSet_2 = DatasetLoader(DATA_PATH)

sample = TrackSet_1[1]
sample_, sampling_rate = librosa.load(sample,sr=3000, offset=0.0, duration=30)
ipd.Audio(sample_,rate=3000)

# plt.figure(figsize=(18,15))
# for i in range(4):
#     plt.subplot(4, 4, i + 1)
#     j = load(TrackSet_1[i])
#     librosa.display.waveshow(j[0], sr=3000)

train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((TrackSet_1))
    .map(map_data, num_parallel_calls=AUTOTUNE)
    .shuffle(3)
    .batch(BATCH_SIZE)
)
test_dataset = (
    tf.data.Dataset
    .from_tensor_slices((TrackSet_2))
    .map(map_data, num_parallel_calls=AUTOTUNE)
    .shuffle(3)
    .batch(BATCH_SIZE)
)