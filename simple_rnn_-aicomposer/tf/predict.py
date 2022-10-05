import tensorflow as tf
import numpy as np
import pandas as pd
import hyperpara as hp
from model import MyModel
from midi_utils import midi_to_notes, notes_to_midi


def predict_next_note(
        notes: np.ndarray,
        keras_model: MyModel,
        temperature: float = 1.0):
    """Generates a note IDs using a trained sequence model."""

    # Add batch dimension
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples=1)
    pitch = tf.squeeze(pitch, axis=-1)
    duration = tf.squeeze(duration, axis=-1)
    step = tf.squeeze(step, axis=-1)

    step = tf.maximum(0, step)
    duration = tf.maximum(0, duration)

    return int(pitch), float(step), float(duration)


def generate(sample_midi_file: str, output_midi_file: str, model, num_predictions):
    raw_notes = midi_to_notes(sample_midi_file)
    sample_notes = np.stack([raw_notes[key] for key in hp.key_order], axis=1)

    input_notes = (
        sample_notes[:hp.seq_length] / np.array([hp.vocab_size, 1, 1]))

    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(
            input_notes, model, hp.temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(
            input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*hp.key_order, 'start', 'end'))
    print(generated_notes.head(10))

    out_pm = notes_to_midi(
        generated_notes, out_file=output_midi_file, instrument_name="Acoustic Grand Piano")

if __name__ == '__main__':
    model = MyModel()
    model.load_weights("training_checkpoints\\ckpt_50")
    generate("sample.mid", "output.mid", model, 120)
