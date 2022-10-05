import tensorflow as tf
import numpy as np
import hyperpara as hp
from loaddata import load, create_sequences
from model import MyModel
import matplotlib.pyplot as plt


def create_train_data(notes_ds: tf.data.Dataset):
    buffersize = len(notes_ds)-hp.seq_length
    seq_ds = create_sequences(notes_ds, hp.seq_length, hp.vocab_size)
    return seq_ds.shuffle(buffersize).batch(hp.batch_size, drop_remainder=True)


def mse_with_positive_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true - y_pred) ** 2
    positive_pressure = 10 * tf.maximum(-y_pred, 0.0)
    return tf.reduce_mean(mse + positive_pressure)


def train():
    notes_ds = load()
    train_ds = create_train_data(notes_ds)

    print(train_ds.element_spec)

    model = MyModel()
    loss = {
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True),
        'step': mse_with_positive_pressure,
        'duration': mse_with_positive_pressure,
    }
    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.learning_rate)
    model.compile(loss=loss, loss_weights={
        'pitch': 1.0,
        'step': 100.0,
        'duration': 100.0,
    }, optimizer=optimizer)
    model.build(input_shape=(None, hp.seq_length, 3))
    model.summary()

    losses = model.evaluate(train_ds, return_dict=True)
    print(losses)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='./training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
    ]

    history = model.fit(
        train_ds,
        epochs=hp.epochs,
        callbacks=callbacks,
    )

    plt.plot(history.epoch, history.history['loss'], label='total loss')
    plt.show()


if __name__ == '__main__':
    train()
