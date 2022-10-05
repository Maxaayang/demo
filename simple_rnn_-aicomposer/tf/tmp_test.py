import pretty_midi
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from midi_utils import midi_to_notes, notes_to_midi
from plt_utils import plot_piano_roll, plot_distributions

'''
ds = tf.data.Dataset.from_tensor_slices([[1,2,3],[4,5,6],[7,8,9]])
ds = ds.flat_map(lambda x : tf.data.Dataset.from_tensor_slices(x+1))
print(list(ds.as_numpy_iterator()))
'''

filenames = glob.glob("GiantMIDI-Piano\\midis\\*.mid")
print('Number of files:', len(filenames))

sample_file = filenames[0]
print(sample_file)

pm = pretty_midi.PrettyMIDI(sample_file)
print('Number of instruments:', len(pm.instruments))
instrument = pm.instruments[0]
print(instrument)

for i, note in enumerate(instrument.notes[:10]):
    note_name = pretty_midi.note_number_to_name(note.pitch)
    duration = note.end - note.start
    print(f'{i}: pitch={note.pitch}, note_name={note_name}, duration={duration:.4f}')

raw_notes = midi_to_notes(sample_file)
print(raw_notes.head())

# plot_piano_roll(raw_notes)
# plot_distributions(raw_notes)

example_file = 'example.mid'
example_pm = notes_to_midi(
    raw_notes, out_file=example_file, instrument_name="Helicopter")
