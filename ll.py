# import pickle

# seq_len = pickle.load(open('../seq_len', 'rb'))

# size = seq_len.shape[0]
# for i in range(size):
#     if (i > 0):
#         seq_len[i] += seq_len[i - 1]

# pickle.dump(seq_len, open('../seq_len', 'wb'))

import torch.nn as nn
import torch
from math import sqrt
from dataload import SequenceMIDI
from model import VQVAE
from tqdm import tqdm
from util import *
import tensorflow as tf
from tensorflow.keras.layers import Dense, \
    GRU, Input, Bidirectional, RepeatVector, \
    TimeDistributed, Lambda
from tensorflow.keras import Model
from tensorflow.python.keras import backend as K
# import tfpyth

# class GRUCell(nn.Module):
#     """自定义GRUCell"""
#     def __init__(self, input_size, hidden_size):
#         super(GRUCell, self).__init__()
#         # 输入变量的线性变换过程是 x @ W.T + b (@代表矩阵乘法， .T代表矩阵转置) 
#         # in2hid_w 的原始形状应是 (hidden_size, input_size), 为了编程的方便, 这里改成(input_size, hidden_size)
#         lb, ub = -sqrt(1/hidden_size), sqrt(1/hidden_size)
#         self.in2hid_w = nn.ParameterList([self.__init(lb, ub, input_size, hidden_size) for _ in range(3)])
#         self.hid2hid_w = nn.ParameterList([self.__init(lb, ub, hidden_size, hidden_size) for _ in range(3)])
#         self.in2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])
#         self.hid2hid_b = nn.ParameterList([self.__init(lb, ub, hidden_size) for _ in range(3)])

#     @staticmethod
#     def __init(low, upper, dim1, dim2=None):
#         if dim2 is None:
#             return nn.Parameter(torch.rand(dim1) * (upper - low) + low)  # 按照官方的初始化方法来初始化网络参数
#         else:
#             return nn.Parameter(torch.rand(dim1, dim2) * (upper - low) + low)

#     def forward(self, x, hid):
#         r = torch.sigmoid(torch.mm(x, self.in2hid_w[0]) + self.in2hid_b[0] +
#                           torch.mm(hid, self.hid2hid_w[0]) + self.hid2hid_b[0])
#         z = torch.sigmoid(torch.mm(x, self.in2hid_w[1]) + self.in2hid_b[1] +
#                           torch.mm(hid, self.hid2hid_w[1]) + self.hid2hid_b[1])
#         n = torch.tanh(torch.mm(x, self.in2hid_w[2]) + self.in2hid_b[2] +
#                        torch.mul(r, (torch.mm(hid, self.hid2hid_w[2]) + self.hid2hid_b[2])))
#         next_hid = torch.mul((1 - z), n) + torch.mul(z, hid)
#         return next_hid

# class GRUModel(nn.Module):
    
#     def __init__(self, input_num, hidden_num, output_num):
#         super(GRUModel, self).__init__()
#         self.hidden_size = hidden_num
#         self.grucell = nn.GRUCell(input_num, hidden_num)
#         self.out_linear = nn.Linear(hidden_num, output_num)

#     def forward(self, x, hid = None):
#         if hid is None:
#             hid = torch.randn(x.shape[0], self.hidden_size)
#         next_hid = self.grucell(x, hid)  # 需要传入隐藏层状态
#         y = self.out_linear(next_hid)
#         return y, next_hid.detach()  # detach()和detach_()都可以使用

trainning_data = SequenceMIDI(
    BASE_PATH, sequence_lenth, max_file_num)

X = trainning_data.__getitem__(0)

# model = VQVAE(in_channels, embedding_dim, num_embeddings).to('cuda')

# gru = GRUModel(16, 64, 89)

# session = tf.Session()
def gru1():
    encoder_input = Input(shape=(time_step, input_dim), name='encoder_input')
    rnn1 = Bidirectional(GRU(rnn_dim, return_sequences=True), name='rnn1')(encoder_input)   # (1, 64, 192)
    rnn2 = Bidirectional(GRU(rnn_dim), name='rnn2')(rnn1)   # (1, 192)

    z_mean = Dense(z_dim, name='z_mean')(rnn2)  # (1, 96)
    z_log_var = Dense(z_dim, name='z_log_var')(rnn2)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(z_dim,), name='z')([z_mean, z_log_var])

    enco = Model(encoder_input, rnn1, name='encoder')
    return enco

def gru2():
    encoder_input = Input(shape=(64, 192), name='encoder_input')
    # rnn1 = Bidirectional(GRU(rnn_dim, return_sequences=True), name='rnn1')(encoder_input)   # (1, 64, 192)
    rnn2 = Bidirectional(GRU(rnn_dim), name='rnn2')(encoder_input)   # (1, 192)

    z_mean = Dense(z_dim, name='z_mean')(rnn2)  # (1, 96)
    z_log_var = Dense(z_dim, name='z_log_var')(rnn2)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(z_dim,), name='z')([z_mean, z_log_var])

    enco = Model(encoder_input, z, name='encoder')
    return enco

def gru3():
    # encoder_input = Input(shape=(96), name='encoder_input')
    # rnn1 = Bidirectional(GRU(rnn_dim, return_sequences=True), name='rnn1')(encoder_input)   # (1, 64, 192)
    decoder_latent_input = Input(shape=z_dim, name='z_sampling')
    repeated_z = RepeatVector(time_step, name='repeated_z_tension')(decoder_latent_input)
    rnn1_output = GRU(rnn_dim, name='decoder_rnn1', return_sequences=True)(repeated_z)

    rnn2_output = GRU(rnn_dim, name='decoder_rnn2', return_sequences=True)(
        rnn1_output)
    rnn2 = Bidirectional(GRU(rnn_dim), name='rnn2')(repeated_z)   # (1, 192)

    z_mean = Dense(z_dim, name='z_mean')(rnn2)  # (1, 96)
    z_log_var = Dense(z_dim, name='z_log_var')(rnn2)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = Lambda(sampling, output_shape=(z_dim,), name='z')([z_mean, z_log_var])

    enco = Model(decoder_latent_input, rnn2_output, name='encoder')
    return enco

gru11 = gru1()
gru12 = gru2()
gru13 = gru3()
ss = X[:1, :, :]    # (1, 64, 89)
aa = gru11(ss)
bb = gru12(aa)
cc = gru13(bb)
