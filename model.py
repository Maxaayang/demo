from email.header import decode_header
from mimetypes import init
import torch
from base import BaseVAE
from torch import nn
from torch.nn import functional as F
# from .types_ import *
from base import *
from util import *
from typing import List, Callable, Union, Any, TypeVar, Tuple
import matplotlib.pyplot as plt
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')

class VectorQuantizer1(nn.Module):
    """
    Reference:
    [1] https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
    """
    def __init__(self, axis=-1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # self._use_codebook_loss = use_codebook_loss
        # self._cfg['init'].bind(nn.init.kaiming_uniform_)(self.embedding.weight)
        self._axis = axis
        self.beta = beta

    def forward(self, input):
        input = np.array(input)
        if self._axis != -1:
            input = input.transpose(self._axis, -1)

        latents_shape = input.shape
        # Compute L2 distance between latents and embedding weights
        input = torch.from_numpy(input).cuda()
        distances = (torch.sum(input ** 2, axis=-1, keepdim=True)
                     - 2 * torch.matmul(input, self.embedding.weight.T)
                     + torch.sum(self.embedding.weight ** 2, axis=-1))

        # Get the encoding that has the min distance
        # 获取距离最近的向量及其索引
        ids = torch.argmin(distances, axis=-1)
        quantized = self.embedding(ids)

        # Convert to one-hot encodings
        device = input.device
        encoding_one_hot = torch.zeros(ids.size(0), num_embeddings, device=device)
        # print("ids ", ids)
        # print("ids.shape ", ids.shape)
        # print("distances.shape", distances.shape)
        # print("encoding_one_hot.shape ", encoding_one_hot.shape)
        encoding_one_hot.scatter_(1, ids, 1)  # [BHW x K]

        # Quantize the latents
        quantized_latents = torch.matmul(encoding_one_hot, self.embedding.weight)  # [BHW, D]
        quantized_latents = quantized_latents.view(latents_shape)  # [B x H x W x D]

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(quantized_latents.detach(), input)
        embedding_loss = F.mse_loss(quantized_latents, input.detach())

        vq_loss = commitment_loss * self.beta + embedding_loss

        # Add the residue back to the latents
        quantized_latents = input + (quantized_latents - input).detach()

        return quantized, vq_loss

class VectorQuantizer(nn.Module):
    """
    VQ-VAE layer: Input any tensor to be quantized. 
    Args:
        embedding_dim (int): the dimensionality of the tensors in the
          quantized space. Inputs to the modules must be in this format as well.
        num_embeddings (int): the number of vectors in the quantized space.
        commitment_cost (float): scalar which controls the weighting of the loss terms (see
          equation 4 in the paper - this variable is Beta).
    """
    def __init__(self):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = beta
        self.init = False
        
        # initialize embeddings
        # self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim).cuda()

    def _tile(self, x):
        d, ew = x.shape
        if d < self.num_embeddings:
            n_repeats = (self.num_embeddings + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def init_emb(self, x):
        self.init = True
        y = self._tile(x)
        _k_rand = y[torch.randperm(y.shape[0])][:]
        self.embeddings = _k_rand
        
    # TODO 这里除了问题, 编码之后应该是一维的
    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        x = torch.squeeze(x, dim=0)

        if not self.init:
            self.init_emb(x)

        # x = x.permute(0, 2, 1).contiguous()
        # [B, H, W, C] -> [BHW, C]
        # x = x[:, :96, :]
        flat_x = x.reshape(-1, self.embedding_dim)
        # flat_x = flat_x.to(device)
        
        # encoding_indices = self.get_code_indices(flat_x)
        min_distance, x_l = self.get_code_indices(flat_x)
        quantized = self.quantize(x_l)
        quantized = quantized.view_as(x) # [B, H, W, C]
        
        # if not self.training:
        #     quantized = quantized.permute(0, 3, 1, 2).contiguous()
        #     return quantized
        
        # embedding loss: move the embeddings towards the encoder's output
        # quantized_data = quantized.data
        # x_data = x.data
        # q_latent_loss = F.mse_loss(quantized, x.detach())
        # # commitment loss
        # e_latent_loss = F.mse_loss(x_data, quantized_data.detach())
        # loss = q_latent_loss + self.commitment_cost * e_latent_loss
        loss = torch.norm(quantized.detach() - x) ** 2 / np.prod(x.shape) * self.commitment_cost

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        
        # quantized = quantized.permute(0, 3, 1, 2).contiguous()
        # quantized = quantized.permute(0, 2, 1).contiguous()
        # quantized = torch.squeeze(quantized)
        return x_l, quantized, loss
    
    def get_code_indices(self, flat_x):
        # compute L2 distance
        # print("self.embeddings.weight ", self.embeddings.weight)
        # distances = (
        #     torch.sum(flat_x ** 2, dim=1, keepdim=True) +
        #     torch.sum(self.embeddings.weight ** 2, dim=1) -
        #     2. * torch.matmul(flat_x, self.embeddings.weight.t())
        # ) # [N, M]

        # TODO

        distances = (
            torch.sum(flat_x ** 2, dim=-1, keepdim=True) +
            torch.sum(self.embeddings.t().to('cuda') ** 2, dim=0, keepdim=True) -
            2. * torch.matmul(flat_x, self.embeddings.t().to('cuda'))
        ) # [N, M]

        min_distance, x_l = torch.min(distances, dim=-1) # (min, min_indices)
        # encoding_indices = torch.argmin(distances, dim=1) # [N,]
        return min_distance, x_l
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        # TODO
        x = F.embedding(encoding_indices, self.embeddings)
        return x
        # return self.embeddings(encoding_indices) 

class ResidualLayer(nn.Module):

    def __init__(self,
                 in_channels: int,
                 out_channels: int):
        super(ResidualLayer, self).__init__()
        self.resblock = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                                kernel_size=3, padding=1, bias=False),
                                      nn.ReLU(True),
                                      nn.Conv2d(out_channels, out_channels,
                                                kernel_size=1, bias=False))

    def forward(self, input: Tensor) -> Tensor:
        return input + self.resblock(input)


class VQVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 embedding_dim: int,
                 num_embeddings: int,
                #  hidden_dims: List = None,
                 beta: float = 0.25,
                 img_size: int = 64,
                 **kwargs) -> None:
        super(VQVAE, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.img_size = img_size
        self.beta = beta
        self.egru = nn.GRU(input_dim, rnn_dim)
        self.gru = nn.GRU(rnn_dim, rnn_dim)
        self.relu = nn.LeakyReLU()
        self.dgru = nn.GRU(embedding_dim, rnn_dim)
        self.linear = nn.Linear(rnn_dim, rnn_dim)

        self.vq_layer = VectorQuantizer()

    def encode_(self, input1: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        output1, states = self.egru(input1)
        output2, states1 = self.gru(output1)
        result = self.linear(output2)

        return [result]

    def decode_(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        output1, states = self.dgru(z)
        output2, states1 = self.gru(output1)

        return output2

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode_(input)[0]
        encoding = torch.squeeze(encoding, dim = 1)
        index, quantized_inputs, vq_loss = self.vq_layer(encoding)
        quantized_inputs = quantized_inputs.repeat(time_step)
        decode_value = self.decode_(quantized_inputs)

        x = input
        if (decode_value.shape != x.shape):
            # 这里把输入和输出的维度不一样处理了一下
            decode_value = F.pad(input=decode_value, pad=(
                0, x.shape[-1]-decode_value.shape[-1]), mode='constant', value=0)


        recon_loss = torch.mean(torch.abs(decode_value - input))
        loss = vq_loss + recon_loss
        print("vq_loss: ", vq_loss, " recon_loss: ", recon_loss)
        return [index, decode_value, input, loss]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        vq_loss = args[2]

        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss + vq_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'VQ_Loss':vq_loss}

    def sample(self,
               num_samples: int,
               current_device: Union[int, str], **kwargs) -> Tensor:
        raise Warning('VQVAE sampler is not implemented.')

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

class MyLoss(torch.nn.Module):
    def __init__(self, weight):
        super(MyLoss, self).__init__()
        self.weight = torch.Tensor(weight)
        self.pitch_loss=torch.nn.CrossEntropyLoss()
        self.step_loss=mse_with_positive_pressure
        self.duration_loss=mse_with_positive_pressure

    def forward(self, pred, y):
        a = self.pitch_loss(pred['pitch'], y['pitch'])
        b = self.step_loss(pred['step'], y['step'])
        c = self.duration_loss(pred['duration'], y['duration'])
        return a*self.weight[0]+b*self.weight[1]+c*self.weight[2]

def draw_two_figure(tensile_strain, diameter, first_name='tensile strain',
                    second_name='diameter',
                    file_name='default.png', y_label='tension',
                    title='tension figure',
                    save=False):
    if tensile_strain.shape[0] == 64:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        # tensile_strain = tensile_strain
        # diameter = diameter.detach().cpu().numpy()
        ax.plot(tensile_strain, label=first_name)
        ax.plot(diameter, label=second_name)
        ax.legend()
        ax.set_ylabel(y_label)
        ax.set_xlabel('timestep')
        ax.set_title(title)

    if save is True:
        plt.savefig(file_name)

    plt.show()
    plt.close('all')

def manipuate_latent_space(piano_roll, vector_up_t, vector_high_d, vector_up_down_t,
                                                 vae,t_up_factor,d_high_factor,t_up_down_factor,
                                                   change_t=True,change_d=False,change_t_up_down=False,
                                                   with_input=True,draw_tension=True):

    if with_input and piano_roll is not None:
        piano_roll = np.expand_dims(piano_roll, 0)
        piano_roll = torch.from_numpy(piano_roll).float().to('cuda:0')
        z = vae.encode_(piano_roll)
    else:
        z = np.random.normal(size=(1,z_dim))

    z = torch.tensor( [item.cpu().detach().numpy() for item in z] )
    z = torch.squeeze(z, dim = 0).to('cuda')
    reconstruction = vae.decode_(z)
    reconstruction = reconstruction.to('cuda')

    tensile_output_function = nn.Linear(rnn_dim, tension_output_dim).to('cuda')
    diameter_output_function = nn.Linear(rnn_dim, tension_output_dim).to('cuda')
    melody_rhythm_output_function = nn.Linear(rnn_dim, melody_note_start_dim).to('cuda')
    melody_pitch_output_function = nn.Linear(rnn_dim, melody_output_dim).to('cuda')
    bass_rhythm_output_function = nn.Linear(rnn_dim, bass_note_start_dim).to('cuda')
    bass_pitch_output_function = nn.Linear(rnn_dim, bass_output_dim).to('cuda')

    melody_pitch_output = melody_pitch_output_function(reconstruction)
    melody_rhythm_output = melody_rhythm_output_function(reconstruction)
    bass_pitch_output = bass_pitch_output_function(reconstruction)
    bass_rhythm_output = bass_rhythm_output_function(reconstruction)
    tensile_output = tensile_output_function(reconstruction)
    diameter_output = diameter_output_function(reconstruction)
    reconstruction = [melody_pitch_output, melody_rhythm_output, bass_pitch_output, bass_rhythm_output,
                tensile_output, diameter_output
                ]

    # TODO
    tensile_reconstruction = np.squeeze(reconstruction[-2])
    tensile_reconstruction = tensile_reconstruction.cpu().detach()
    diameter_reconstruction = np.squeeze(reconstruction[-1])
    diameter_reconstruction = diameter_reconstruction.cpu().detach()

    # recon_result = result_sampling(np.concatenate(list(reconstruction), axis=-1))[0]
    changed_z = z
    changed_z = changed_z.to('cpu')
    if change_t:
        changed_z += t_up_factor * vector_up_t

    if change_d:
        # changed_z = torch.tensor( [item.cpu().detach().numpy() for item in changed_z] )
        changed_z += d_high_factor * vector_high_d

    if change_t_up_down:
        changed_z += t_up_down_factor * vector_up_down_t

    changed_z = changed_z.to('cuda:0')
    changed_reconstruction = vae.decode_(changed_z)

    changed_reconstruction = torch.tensor([item.cpu().detach().numpy() for item in changed_reconstruction])
    changed_reconstruction = torch.squeeze(changed_reconstruction, dim = 1)
    changed_reconstruction = changed_reconstruction.to('cuda')
    
    melody_pitch_output = melody_pitch_output_function(changed_reconstruction)
    melody_rhythm_output = melody_rhythm_output_function(changed_reconstruction)
    bass_pitch_output = bass_pitch_output_function(changed_reconstruction)
    bass_rhythm_output = bass_rhythm_output_function(changed_reconstruction)
    tensile_output = tensile_output_function(changed_reconstruction)
    diameter_output = diameter_output_function(changed_reconstruction)
    changed_reconstruction = [melody_pitch_output, melody_rhythm_output, bass_pitch_output, bass_rhythm_output,
                tensile_output, diameter_output
                ]

    changed_reconstruction = [item.cpu().detach().numpy() for item in changed_reconstruction]
    changed_recon_result = result_sampling(np.concatenate(list(changed_reconstruction), axis=-1))[0]
    # changed_recon_result = changed_reconstruction.detach().cpu().numpy()

    # TODO
    # changed_tensile_reconstruction = np.squeeze(changed_reconstruction[-2])
    changed_tensile_reconstruction = np.squeeze(changed_reconstruction[-2])

    changed_diameter_reconstruction = np.squeeze(changed_reconstruction[-1])

    if draw_tension:
        draw_two_figure(tensile_reconstruction,diameter_reconstruction,title='original tension')
        draw_two_figure(changed_tensile_reconstruction,changed_diameter_reconstruction,title='changed tension')


    return piano_roll, changed_recon_result
