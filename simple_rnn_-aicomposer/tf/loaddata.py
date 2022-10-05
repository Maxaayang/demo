import pretty_midi
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import hyperpara as hp

from midi_utils import midi_to_notes, notes_to_midi


def load() -> tf.data.Dataset:
    filenames = glob.glob("GiantMIDI-Piano\\midis\\*.mid")
    print('Number of files:', len(filenames))

    num_files = 100
    all_notes = []
    for f in filenames[:num_files]:
        notes = midi_to_notes(f)
    all_notes.append(notes)

    all_notes = pd.concat(all_notes)
    n_notes = len(all_notes)
    print('Number of notes parsed:', n_notes)

    key_order = ['pitch', 'step', 'duration']
    train_notes = np.stack([all_notes[key] for key in hp.key_order], axis=1)
    notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)
    print(notes_ds.element_spec)

    return notes_ds


def create_sequences(
    dataset: tf.data.Dataset,
    seq_length: int,
    vocab_size=hp.vocab_size,
) -> tf.data.Dataset:
    """Returns TF Dataset of sequence and label examples."""
    seq_length = seq_length+1

    # Take 1 extra for the labels
    windows = dataset.window(seq_length, shift=1, stride=1,
                             drop_remainder=True)

    # `flat_map` flattens the" dataset of datasets" into a dataset of tensors
    def flatten(x): return x.batch(seq_length, drop_remainder=True)
    sequences = windows.flat_map(flatten)

    # Normalize note pitch
    def scale_pitch(x):
        x = x/[vocab_size, 1.0, 1.0]
        return x

    # Split the labels
    def split_labels(sequences):
        inputs = sequences[:-1]
        labels_dense = sequences[-1]
        labels = {key: labels_dense[i] for i, key in enumerate(hp.key_order)}

        return scale_pitch(inputs), labels

    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)


if __name__ == '__main__':
    load()
