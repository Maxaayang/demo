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
import tfpyth

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

model = VQVAE(in_channels, embedding_dim, num_embeddings).to('cuda')

# gru = GRUModel(16, 64, 89)

# session = tf.Session()
def gru():
    encoder_input = Input(shape=(time_step, input_dim), name='encoder_input')
    rnn1 = Bidirectional(GRU(rnn_dim, return_sequences=True), name='rnn1')(encoder_input)
    enco = Model(encoder_input, rnn1, name='encoder')
    # f = tfpyth.torch_from_tensorflow(session, )

gru1 = gru()

aa = gru.layers[1].predict
