from lib2to3.pgen2.token import GREATER
import torch as t
import torch.nn as nn
import sys
sys.path.append("..")
# from vqvae.resnet import Resnet, Resnet1D
# from utils.torch_utils import assert_shape

from juke.juke_params import *
from unmix.unmix.vqvae.resnet import Resnet, Resnet1D
from unmix.unmix.utils.torch_utils import assert_shape


class EncoderConvBlock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv,
                 dilation_growth_rate=1, dilation_cycle=None, zero_out=False,
                 res_scale=False):
        super().__init__()
        print("input_emb_width: ", input_emb_width)
        print("width: ", width)
        print("output_emb_width: ", output_emb_width)
        blocks = []
        filter_t, pad_t = 3, stride_t // 2
        if down_t > 0:
            # for i in range(1):
                # block = nn.Sequential(
                    # nn.Conv1d(input_emb_width if i == 0 else width,
                    #           width, filter_t, stride_t, pad_t),
                    # Resnet1D(width, depth, m_conv, dilation_growth_rate,
                    #          dilation_cycle, zero_out, res_scale),
                self.first_enc = nn.GRU(input_dim, rnn_dim)
                self.second_enc =  nn.GRU(rnn_dim, embedding_dim)
                # )
                # blocks.append(block)
            # block = nn.Conv1d(width, output_emb_width, 3, 1, 1)
            # blocks.append(block)
        # self.model = nn.Sequential(*blocks)

    def forward(self, x):
        #print("encoder")
        output1, state1 = self.first_enc(x)
        output2, state2 = self.second_enc(output1)
        return output2


class DecoderConvBock(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, down_t,
                 stride_t, width, depth, m_conv, dilation_growth_rate=1, dilation_cycle=None, zero_out=False, res_scale=False, reverse_decoder_dilation=False, checkpoint_res=False):
        super().__init__()
        blocks = []
        if down_t > 0:
            filter_t, pad_t = stride_t * 2, stride_t // 2
            # block = nn.Conv1d(output_emb_width, width, 3, 1, 1)
            # blocks.append(block)
            # for i in range(down_t):
            #     block = nn.Sequential(
                    # Resnet1D(width, depth, m_conv, dilation_growth_rate, dilation_cycle, zero_out=zero_out,
                    #          res_scale=res_scale, reverse_dilation=reverse_decoder_dilation, checkpoint_res=checkpoint_res),
                    # nn.ConvTranspose1d(width, input_emb_width if i == (
                    #     down_t - 1) else width, filter_t, stride_t, pad_t)
            self.fir_dec = nn.GRU(embedding_dim, embedding_dim)
            self.sec_dec = nn.GRU(embedding_dim, embedding_dim)
                # )
        #         blocks.append(block)
        # self.model = nn.Sequential(*blocks)

    def forward(self, x):
        #print("decoder")
        output1, state1 = self.fir_dec(x)
        output2, state2 = self.sec_dec(output1)
        return output2


class Encoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels
        self.downs_t = downs_t
        self.strides_t = strides_t

        block_kwargs_copy = dict(**block_kwargs)
        if 'reverse_decoder_dilation' in block_kwargs_copy:
            del block_kwargs_copy['reverse_decoder_dilation']

        def level_block(level, down_t, stride_t): return EncoderConvBlock(input_emb_width if level == 0 else output_emb_width,
                                                                          output_emb_width,
                                                                          down_t, stride_t,
                                                                          **block_kwargs_copy)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

    def forward(self, x):
        N, T = x.shape[0], x.shape[-1]
        emb = self.input_emb_width
        # assert_shape(x, (N, emb, T))
        xs = []

        # 64, 32, ...
        # print("levels ", self.levels)
        # print("input_x.shape ", x.shape)
        iterator = zip(list(range(self.levels)), self.downs_t, self.strides_t)
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T // (stride_t ** down_t)
            emb = self.output_emb_width
            # TODO
            # print(x.shape, " ", (N, emb, T))
            # assert_shape(x, (N, emb, T))
            xs.append(x)

        return xs


class Decoder(nn.Module):
    def __init__(self, input_emb_width, output_emb_width, levels, downs_t,
                 strides_t, **block_kwargs):
        super().__init__()
        self.input_emb_width = input_emb_width
        self.output_emb_width = output_emb_width
        self.levels = levels

        self.downs_t = downs_t

        self.strides_t = strides_t

        def level_block(level, down_t, stride_t): return DecoderConvBock(output_emb_width,
                                                                         output_emb_width,
                                                                         down_t, stride_t,
                                                                         **block_kwargs)
        self.level_blocks = nn.ModuleList()
        iterator = zip(list(range(self.levels)), downs_t, strides_t)
        for level, down_t, stride_t in iterator:
            self.level_blocks.append(level_block(level, down_t, stride_t))

        self.out = nn.Conv1d(output_emb_width, input_emb_width, 3, 1, 1)

    def forward(self, xs, all_levels=True):
        if all_levels:
            assert len(xs) == self.levels
        else:
            assert len(xs) == 1
        x = xs[-1]
        N, T = x.shape[0], x.shape[-1]
        emb = self.output_emb_width
        # assert_shape(x, (N, emb, T))

        # 32, 64 ...
        iterator = reversed(
            list(zip(list(range(self.levels)), self.downs_t, self.strides_t)))
        for level, down_t, stride_t in iterator:
            level_block = self.level_blocks[level]
            x = level_block(x)
            emb, T = self.output_emb_width, T * (stride_t ** down_t)
            # assert_shape(x, (N, emb, T))
            if level != 0 and all_levels:
                x = x + xs[level - 1]

        # x = self.out(x)
        return x