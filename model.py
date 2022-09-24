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
        
        # initialize embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim).cuda()
        
    def forward(self, x):
        # [B, C, H, W] -> [B, H, W, C]
        # print("x.shape ", x.shape)
        
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # TODO 这里训练时要注释掉
        # device = torch.device('cuda:0')
        # x = x.to(device)
        # x = x.squeeze(dim=1)

        # x = x.permute(0, 2, 3, 1).contiguous()
        # print("x", x)
        x = x.permute(0, 2, 1).contiguous()
        # [B, H, W, C] -> [BHW, C]
        x = x[:, :96, :]
        flat_x = x.reshape(-1, self.embedding_dim)
        # flat_x = flat_x.to(device)
        
        encoding_indices = self.get_code_indices(flat_x)
        quantized = self.quantize(encoding_indices)
        quantized = quantized.view_as(x) # [B, H, W, C]
        
        if not self.training:
            quantized = quantized.permute(0, 3, 1, 2).contiguous()
            return quantized
        
        # embedding loss: move the embeddings towards the encoder's output
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = x + (quantized - x).detach()
        
        # quantized = quantized.permute(0, 3, 1, 2).contiguous()
        quantized = quantized.permute(0, 2, 1).contiguous()
        # quantized = torch.squeeze(quantized)
        return quantized, loss
    
    def get_code_indices(self, flat_x):
        # compute L2 distance
        # print("self.embeddings.weight ", self.embeddings.weight)
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2. * torch.matmul(flat_x, self.embeddings.weight.t())
        ) # [N, M]
        encoding_indices = torch.argmin(distances, dim=1) # [N,]
        return encoding_indices
    
    def quantize(self, encoding_indices):
        """Returns embedding tensor for a batch of indices."""
        return self.embeddings(encoding_indices) 

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
        self.tensile_output = nn.Linear(rnn_dim, tension_output_dim)
        self.diameter_output = nn.Linear(rnn_dim, tension_output_dim)
        self.melody_rhythm_output = nn.Linear(rnn_dim, melody_note_start_dim)
        self.melody_pitch_output = nn.Linear(rnn_dim, melody_output_dim)
        self.bass_rhythm_output = nn.Linear(rnn_dim, bass_note_start_dim)
        self.bass_pitch_output = nn.Linear(rnn_dim, bass_output_dim)

        # modules = []
        # # if hidden_dims is None:
        # #     hidden_dims = [128, 256]

        # # Build Encoder
        # # for h_dim in hidden_dims:
        # modules.append(
        #     nn.Sequential(
        #         nn.GRU(input_dim, rnn_dim),
        #         # nn.LeakyReLU())
        #     )
        # )
        #     # in_channels = h_dim

        # modules.append(
        #     nn.Sequential(
        #         nn.GRU(rnn_dim, rnn_dim),
        #         # nn.LeakyReLU())
        #     )
        # )

        # for _ in range(6):
        #     modules.append(ResidualLayer(in_channels, in_channels))
        # modules.append(nn.LeakyReLU())

        # modules.append(
        #     nn.Sequential(
        #         nn.GRU(rnn_dim, rnn_dim),
        #         # nn.LeakyReLU())
        #     )
        # )

        # self.encoder = nn.Sequential(*modules)

        self.vq_layer = VectorQuantizer()

        # Build Decoder
        # modules = []
        # modules.append(
        #     nn.Sequential(
        #         nn.GRU(embedding_dim, rnn_dim),
        #         # nn.LeakyReLU(),
        #         nn.GRU(rnn_dim, rnn_dim),
        #         # nn.LeakyReLU())
        #     )
        # )

        # for _ in range(6):
        #     modules.append(ResidualLayer(hidden_dims[-1], hidden_dims[-1]))

        # modules.append(nn.LeakyReLU())

        # hidden_dims.reverse()

        # for i in range(len(hidden_dims) - 1):
        # modules.append(
        #     nn.Sequential(
        #         nn.ConvTranspose2d(hidden_dims[i],
        #                             hidden_dims[i + 1],
        #                             kernel_size=4,
        #                             stride=2,
        #                             padding=1),
        #         nn.LeakyReLU())
        #     )

        # modules.append(
        #     nn.Sequential(
        #         nn.ConvTranspose2d(hidden_dims[-1],
        #                            out_channels=3,
        #                            kernel_size=4,
        #                            stride=2, padding=1),
        #         nn.Tanh()))

        # self.decoder = nn.Sequential(*modules)

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

        # result = self.decoder(z)
        # print("z.shape", z.shape)
        # TODO
        # z = torch.tensor(z)
        # z = torch.tensor( [item.cpu().detach().numpy() for item in z] ).to('cuda:0')
        # z = z.squeeze(dim=1)
        # TODO 这里需要6个全连接层
        output1, states = self.dgru(z)
        output2, states1 = self.gru(output1)
        melody_pitch_output = self.melody_pitch_output(output2)
        melody_rhythm_output = self.melody_rhythm_output(output2)
        bass_pitch_output = self.bass_pitch_output(output2)
        bass_rhythm_output = self.bass_rhythm_output(output2)
        tensile_output = self.tensile_output(output2)
        diameter_output = self.diameter_output(output2)
        result = [melody_pitch_output, melody_rhythm_output, bass_pitch_output, bass_rhythm_output,
                    tensile_output, diameter_output
                    ]
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        encoding = self.encode_(input)[0]
        quantized_inputs, vq_loss = self.vq_layer(encoding)
        return [self.decode_(quantized_inputs), input, vq_loss]

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
        tensile_strain = tensile_strain.detach().cpu().numpy()
        diameter = diameter.detach().cpu().numpy()
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

    reconstruction = vae.decode_(z)

    # TODO
    tensile_reconstruction = np.squeeze(reconstruction[-2])
    diameter_reconstruction = np.squeeze(reconstruction[-1])

    # recon_result = result_sampling(np.concatenate(list(reconstruction), axis=-1))[0]
    changed_z = z
    if change_t:
        changed_z += t_up_factor * vector_up_t

    if change_d:
        # changed_z = torch.tensor( [item.cpu().detach().numpy() for item in changed_z] )
        changed_z += d_high_factor * vector_high_d

    if change_t_up_down:
        changed_z += t_up_down_factor * vector_up_down_t

    changed_reconstruction = vae.decode_(changed_z)

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
