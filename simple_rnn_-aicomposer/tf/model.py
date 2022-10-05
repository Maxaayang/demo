import tensorflow as tf


class MyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.lstm = tf.keras.layers.LSTM(units=128)
        self.pitch_dense = tf.keras.layers.Dense(units=128, name='pitch')
        self.step_dense = tf.keras.layers.Dense(units=1, name='step')
        self.duration_dense = tf.keras.layers.Dense(units=1, name='duration')

    def call(self, x):
        x = self.lstm(x)
        output = {'pitch': self.pitch_dense(x),
                  'step': self.step_dense(x),
                  'duration': self.duration_dense(x)}
        return output
    