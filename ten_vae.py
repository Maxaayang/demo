import torch
import numpy as np
from dataload import SequenceMIDI
from model import VQVAE
from tqdm import tqdm
from util import *
import tensorflow as tf
# from preprocess_midi import *


import glob
import numpy as np
from tqdm import tqdm
from preprocess_midi import *
from util import *
import pickle


from tensorflow.keras.layers import Dense, \
    GRU, Input, Bidirectional, RepeatVector, \
    TimeDistributed, Lambda
from tensorflow.keras import Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import backend as K

from util import *
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# Dense 全连接层



encoder_input = Input(shape=(time_step, input_dim), name='encoder_input')

rnn1 = Bidirectional(GRU(rnn_dim, return_sequences=True), name='rnn1')(encoder_input)
rnn2 = Bidirectional(GRU(rnn_dim), name='rnn2')(rnn1)


# 进行线性变换, AW + b, 可训练的变量是 W 和 b
z_mean = Dense(z_dim, name='z_mean')(rnn2)
z_log_var = Dense(z_dim, name='z_log_var')(rnn2)

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling, output_shape=(z_dim,), name='z')([z_mean, z_log_var])
class kl_beta(tf.keras.layers.Layer):
    def __init__(self):
        super(kl_beta, self).__init__()

        # your variable goes here
        self.beta = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    def call(self, inputs, **kwargs):
        # your mul operation goes here
        return -self.beta *inputs

beta = kl_beta()
encoder = Model(encoder_input, z, name='encoder')

# decoder

decoder_latent_input = Input(shape=z_dim, name='z_sampling')

repeated_z = RepeatVector(time_step, name='repeated_z_tension')(decoder_latent_input)



rnn1_output = GRU(rnn_dim, name='decoder_rnn1', return_sequences=True)(repeated_z)

rnn2_output = GRU(rnn_dim, name='decoder_rnn2', return_sequences=True)(rnn1_output)

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss = tf.reduce_mean(kl_loss)

kl_loss = 0.5 *kl_loss

kl_loss = beta(kl_loss)
# https://www.jianshu.com/p/223e13ce35a2?u_atoken=f201905d-fb1b-484a-827d-859f4bb214a0&u_asession=01nrQ17WepaF_J1TCv8PZaIZuYLGnYBcZ4DWE4D2l50S8LqgXbEz0N3riCtz8oJlOjX0KNBwm7Lovlpxjd_P_q4JsKWYrT3W_NKPr8w6oU7K9yYtUSKuWfYomXFit3_UC5Pn5sJEo90JdruCukG2OVYmBkFo3NEHBv0PZUm6pbxQU&u_asig=05oEbVwSzUZF-mtwraded9ERi4CxOzuiYzDi_BCEXmJ0ZeSiisN_lmpKpA-Cs80UqeAfEodcMgcM6a9teedKr-Mz1j5iaLBJA85LvpCbO-AXAkVswsfppHbDydXikdA8-ZCsjrOX3Y0Vjb13QzwWEvTQYoes4QlPPYJ4TSoSYZtLH9JS7q8ZD7Xtz2Ly-b0kmuyAKRFSVJkkdwVUnyHAIJzXpfbz50YM0RMA3barHc_9AO14Q2vEGaAqC9fgdU__7NLoLNx2oKfcIBo1yjPF0Bne3h9VXwMyh6PgyDIVSG1W-zQZ_qalgy3JzZcmP3eZL_D1Woc7UlI2Op9PXm8noS-cYWnfUV0kOqH32rf9D7hrsMRBkGxRRre9kPLZgPaxWImWspDxyAEEo4kbsryBKb9Q&u_aref=CmY01xtvkd52QEUGVpOtdz0vh6U%3D
tensile_middle_output = TimeDistributed(Dense(tension_middle_dim, activation='elu'),
                                        name='tensile_strain_dense1')(rnn2_output)

tensile_output = TimeDistributed(Dense(tension_output_dim, activation='elu'),
                                    name='tensile_strain_dense2')(tensile_middle_output)

diameter_middle_output = TimeDistributed(Dense(tension_middle_dim, activation='elu'),
                                            name='diameter_strain_dense1')(rnn2_output)

diameter_output = TimeDistributed(Dense(tension_output_dim, activation='elu'),
                                    name='diameter_strain_dense2')(diameter_middle_output)

melody_rhythm_1 = TimeDistributed(Dense(start_middle_dim, activation='elu'),
                                    name='melody_start_dense1')(rnn2_output)
melody_rhythm_output = TimeDistributed(Dense(melody_note_start_dim, activation='sigmoid'),
                                        name='melody_start_dense2')(
    melody_rhythm_1)

melody_pitch_1 = TimeDistributed(Dense(melody_bass_dense_1_dim, activation='elu'),
                                    name='melody_pitch_dense1')(rnn2_output)

melody_pitch_output = TimeDistributed(Dense(melody_output_dim, activation='softmax'),
                                        name='melody_pitch_dense2')(melody_pitch_1)

bass_rhythm_1 = TimeDistributed(Dense(start_middle_dim, activation='elu'),
                                name='bass_start_dense1')(rnn2_output)

bass_rhythm_output = TimeDistributed(Dense(bass_note_start_dim, activation='sigmoid'),
                                        name='bass_start_dense2')(
    bass_rhythm_1)

bass_pitch_1 = TimeDistributed(Dense(melody_bass_dense_1_dim, activation='elu'),
                                name='bass_pitch_dense1')(rnn2_output)
bass_pitch_output = TimeDistributed(Dense(bass_output_dim, activation='softmax'),
                                    name='bass_pitch_dense2')(bass_pitch_1)

decoder_output = [melody_pitch_output, melody_rhythm_output, bass_pitch_output, bass_rhythm_output,
                    tensile_output, diameter_output
                    ]

decoder = Model(decoder_latent_input, decoder_output, name='decoder')

model_input = encoder_input

vae = Model(model_input, decoder(encoder(model_input)), name='encoder_decoder')

vae.add_loss(kl_loss)

vae.add_metric(kl_loss, name='kl_loss', aggregation='mean')

optimizer = keras.optimizers.Adam()


# metrics 模型的评估
vae.compile(optimizer=optimizer,
            loss=['categorical_crossentropy', 'binary_crossentropy',
                    'categorical_crossentropy', 'binary_crossentropy',
                    'mse', 'mse'
                    ],
            metrics=[[keras.metrics.CategoricalAccuracy()],
                        [keras.metrics.BinaryAccuracy()],
                        [keras.metrics.CategoricalAccuracy()],
                        [keras.metrics.BinaryAccuracy()],
                        [keras.metrics.MeanSquaredError()],
                        [keras.metrics.MeanSquaredError()]
                        ]
            )

# return vae

seq_len = pickle.load(open('../seq_len', 'rb'))
notes = pickle.load(open('../leave_notes', 'rb'))

vae.fit(notes, notes, batch_size = 5, epochs = 16, validation_split = 0.2, validation_freq = 20)
