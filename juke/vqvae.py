import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("..")
# from vqvae.encdec import Encoder, Decoder, assert_shape
# from vqvae.bottleneck import NoBottleneck, Bottleneck
# from utils.logger import average_metrics
# from utils.audio_utils import spectral_convergence, spectral_loss, multispectral_loss, audio_postprocess

from unmix.unmix.vqvae.encdec import Encoder, Decoder, assert_shape
from unmix.unmix.vqvae.bottleneck import NoBottleneck, Bottleneck
from unmix.unmix.utils.logger import average_metrics
from unmix.unmix.utils.audio_utils import spectral_convergence, spectral_loss, multispectral_loss, audio_postprocess


def dont_update(params):
    for param in params:
        param.requires_grad = False


def update(params):
    for param in params:
        param.requires_grad = True


def calculate_strides(strides, downs):
    return [stride ** down for stride, down in zip(strides, downs)]


# def _loss_fn(loss_fn, x_target, x_pred, hps):
def _loss_fn(loss_fn, x_target, x_pred):
    if loss_fn == 'l1':
        # return t.mean(t.abs(x_pred - x_target)) / hps.bandwidth['l1']
        return t.mean(t.abs(x_pred - x_target))
    # elif loss_fn == 'l2':
    #     return t.mean((x_pred - x_target) ** 2) / hps.bandwidth['l2']
    # elif loss_fn == 'linf':
    #     residual = ((x_pred - x_target) ** 2).reshape(x_target.shape[0], -1)
    #     values, _ = t.topk(residual, hps.linf_k, dim=1)
    #     return t.mean(values) / hps.bandwidth['l2']
    # elif loss_fn == 'lmix':
    #     loss = 0.0
    #     if hps.lmix_l1:
    #         loss += hps.lmix_l1 * _loss_fn('l1', x_target, x_pred, hps)
    #     if hps.lmix_l2:
    #         loss += hps.lmix_l2 * _loss_fn('l2', x_target, x_pred, hps)
    #     if hps.lmix_linf:
    #         loss += hps.lmix_linf * _loss_fn('linf', x_target, x_pred, hps)
    #     return loss
    # else:
    #     assert False, f"Unknown loss_fn {loss_fn}"


class VQVAE(nn.Module):
    def __init__(self, input_shape, levels, downs_t, strides_t,
                 emb_width, l_bins, mu, commit, spectral, multispectral,
                 multipliers=None, use_bottleneck=True, **block_kwargs):
        super().__init__()

        self.sample_length = input_shape[0]
        x_shape, x_channels = input_shape[:-1], input_shape[-1]
        self.x_shape = x_shape

        self.downsamples = calculate_strides(strides_t, downs_t)
        self.hop_lengths = np.cumprod(self.downsamples)
        self.z_shapes = z_shapes = [
            (x_shape[0] // self.hop_lengths[level],) for level in range(levels)]
        self.levels = levels

        if multipliers is None:
            self.multipliers = [1] * levels
        else:
            assert len(multipliers) == levels, "Invalid number of multipliers"
            self.multipliers = multipliers

        def _block_kwargs(level):
            this_block_kwargs = dict(block_kwargs)
            this_block_kwargs["width"] *= self.multipliers[level]
            this_block_kwargs["depth"] *= self.multipliers[level]
            return this_block_kwargs

        def encoder(level): return Encoder(64, emb_width, level + 1,
                                           downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
        def decoder(level): return Decoder(x_channels, emb_width, level + 1,
                                           downs_t[:level+1], strides_t[:level+1], **_block_kwargs(level))
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for level in range(levels):
            self.encoders.append(encoder(level))
            self.decoders.append(decoder(level))

        if use_bottleneck:
            self.bottleneck = Bottleneck(l_bins, emb_width, mu, levels)
        else:
            self.bottleneck = NoBottleneck(levels)

        self.downs_t = downs_t
        self.strides_t = strides_t
        self.l_bins = l_bins
        self.commit = commit
        self.spectral = spectral
        self.multispectral = multispectral

    def preprocess(self, x):
        # x: NTC [-1,1] -> NCT [-1,1]
        assert len(x.shape) == 3
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # x: NTC [-1,1] <- NCT [-1,1]
        x = x.permute(0, 2, 1)
        return x

    def _decode(self, zs, start_level=0, end_level=None):
        # Decode
        if end_level is None:
            end_level = self.levels
        assert len(zs) == end_level - start_level
        xs_quantised = self.bottleneck.decode(
            zs, start_level=start_level, end_level=end_level)
        assert len(xs_quantised) == end_level - start_level

        # Use only lowest level
        decoder, x_quantised = self.decoders[start_level], xs_quantised[0:1]
        x_out = decoder(x_quantised, all_levels=False)
        x_out = self.postprocess(x_out)
        return x_out

    def decode(self, zs, start_level=0, end_level=None, bs_chunks=1):
        z_chunks = [t.chunk(z, bs_chunks, dim=0) for z in zs]
        x_outs = []
        for i in range(bs_chunks):
            zs_i = [z_chunk[i] for z_chunk in z_chunks]
            x_out = self._decode(
                zs_i, start_level=start_level, end_level=end_level)
            x_outs.append(x_out)
        return t.cat(x_outs, dim=0)

    def _encode(self, x, start_level=0, end_level=None):
        # Encode
        if end_level is None:
            end_level = self.levels
        x_in = self.preprocess(x)
        xs = []
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        zs = self.bottleneck.encode(xs)
        return zs[start_level:end_level]

    # def encode(self, x, start_level=0, end_level=None, bs_chunks=1):
    def encode(self, x):
        # x_chunks = t.chunk(x, bs_chunks, dim=0)
        zs_list = []
        for x_i in x:
            zs_i = self._encode(
                x_i, start_level=0, end_level=1)
            zs_list.append(zs_i)
        zs = [t.cat(zs_level_list, dim=0) for zs_level_list in zip(*zs_list)]
        return zs

    def sample(self, n_samples):
        zs = [t.randint(0, self.l_bins, size=(n_samples, *z_shape),
                        device='cuda') for z_shape in self.z_shapes]
        return self.decode(zs)

    def forward(self, x, loss_fn='l1'):
        metrics = {}

        N = x.shape[0]

        # Encode/Decode
        # x_in = self.preprocess(x)
        x_in = x
        xs = []

        #print("encoder input: ")
        # print(x_in.shape)
        for level in range(self.levels):
            encoder = self.encoders[level]
            x_out = encoder(x_in)
            xs.append(x_out[-1])
        #print("encoder output: ")
        # print(xs[0].shape)
        xs = t.tensor( [item.cpu().detach().numpy() for item in xs]).squeeze(axis = 0).to('cuda')
        zs, xs_quantised, commit_losses, quantiser_metrics = self.bottleneck(
            xs)
        #print("bottelneck output: ")
        # print(xs_quantised[0].shape)

        x_outs = []
        for level in range(self.levels):
            decoder = self.decoders[level]
            x_out = decoder(xs_quantised[level:level+1], all_levels=False)

            # happens when deploying
            if (x_out.shape != x_in.shape):
                # 这里把输入和输出的维度不一样处理了一下
                x_out = F.pad(input=x_out, pad=(
                    0, x_in.shape[-1]-x_out.shape[-1]), mode='constant', value=0)

            assert_shape(x_out, x_in.shape)
            x_outs.append(x_out)
        #print("decoder output: ")
        # print(x_outs[0].shape)
        # Loss

        recons_loss = t.zeros(()).to(x.device)  # 0.0900
        x_target = x.float()

        for level in reversed(range(self.levels)):
            x_out = self.postprocess(x_outs[level])
            x_out = x_out
            this_recons_loss = _loss_fn(loss_fn, x_target, x_out)
            metrics[f'recons_loss_l{level + 1}'] = this_recons_loss
            recons_loss += this_recons_loss

        commit_loss = sum(commit_losses)

        loss = recons_loss + self.commit * commit_loss

        # TODO
        with t.no_grad():
            # sc = t.mean(spectral_convergence(x_target, x_out, hps))
            # l2_loss = _loss_fn("l2", x_target, x_out, hps)
            l1_loss = _loss_fn("l1", x_target, x_out)
            # print("l1_loss ", l1_loss)
            # linf_loss = _loss_fn("linf", x_target, x_out, hps)

        quantiser_metrics = average_metrics(quantiser_metrics)

        metrics.update(dict(
            recons_loss=recons_loss,
            # spectral_loss=spec_loss,
            # multispectral_loss=multispec_loss,
            # spectral_convergence=sc,
            # l2_loss=l2_loss,
            l1_loss=l1_loss,
            # linf_loss=linf_loss,
            commit_loss=commit_loss,
            **quantiser_metrics))

        for key, val in metrics.items():
            metrics[key] = val.detach()

        # print("metrics ", metrics)
        return x_out, loss, metrics
