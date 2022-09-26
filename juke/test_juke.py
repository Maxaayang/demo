import sys
sys.path.append("..")

import juke_model as model
import util
from juke.preprocess_midi import *
from juke.vqvae import VQVAE
from juke.juke_params import *

import pickle
import torch
import IPython.display as ipd
import os
import sys
import pretty_midi
import numpy as np

block_kwargs = dict(width=width, depth=depth, m_conv=m_conv, dilation_growth_rate=dilation_growth_rate, \
     dilation_cycle=dilation_cycle, reverse_decoder_dilation=reverse_decoder_dilation)

vqvae = VQVAE(input_shape, levels, downs_t, strides_t, emb_width, l_bins, mu, commit, \
              spectral, multispectral, multipliers, use_bottleneck, **block_kwargs)

vqvae.state_dict = torch.load("../model/juke_vae20.pth")
vqvae = vqvae.to('cuda:0')

melody_file = '../r_data/crying_sand.mid'

piano_roll, bar_indices,pm_old = preprocess_midi(melody_file)
piano_roll_new = np.reshape(piano_roll,(-1,piano_roll.shape[-1]))
pm_new = util.roll_to_pretty_midi(piano_roll_new,pm_old)

tensile_strain_direction = "yes" #@param ["yes", "no"]
diameter_direction = "no" #@param ["yes", "no"] {type:"string"}

tensile_strain_level = "no" #@param ["yes", "no"]
diameter_level = "yes" #@param ["yes", "no"]

tensile_strain_up_down = "yes" #@param ["yes", "no"]

#@markdown select first 4 bar tension change factor sign for every 8 bar

first_change = "positive" #@param ["positive", "negative"]

#@markdown select the tension strain change factor
tensile_strain_factor = 6 #@param {type:"slider", min:1, max:7, step:0.5}

#@markdown select the diameter change factor
diameter_factor = 3.5 #@param {type:"slider", min:0.5, max:4, step:0.5}


feature_vectors = []


if tensile_strain_direction == 'yes':
    feature_vectors.append(tensile_up_feature_vector)

if diameter_direction == 'yes':
    feature_vectors.append(diameter_up_feature_vector)

if tensile_strain_level == 'yes':
    feature_vectors.append(tensile_high_feature_vector)

if diameter_level == 'yes':
    feature_vectors.append(diameter_high_feature_vector)

if tensile_strain_up_down == 'yes':
    feature_vectors.append(tensile_up_down_feature_vector)

result_roll, tensile_strain, diameter = four_bar_iterate(piano_roll_new,vqvae,
                                                          feature_vectors,
                                                         tensile_strain_factor,
                                                         diameter_factor,
                                                         first_change)

pm_result = util.roll_to_pretty_midi(result_roll,pm_old)

start_section = 9 #@param {type:"integer"}
end_section = 17 #@param {type:"integer"}

tensile_strain_direction1 = False #@param ["False", "True"] {type:"raw"}
tensile_strain_factor1 = 5.5 #@param {type:"slider", min:-8, max:8, step:0.5}

diameter_level1 = True #@param ["False", "True"] {type:"raw"}
diameter_factor1 = 4 #@param {type:"slider", min:-4, max:4, step:0.5}

#@markdown second 4 bar (optional)

tensile_strain_direction2 = True #@param ["False", "True"] {type:"raw"}
tensile_strain_factor2 = -6 #@param {type:"slider", min:-8, max:8, step:0.5}

diameter_level2 = False #@param ["False", "True"] {type:"raw"}
diameter_factor2 = -4 #@param {type:"slider", min:-4, max:4, step:0.5}

# import pdb; pdb.set_trace()
selected_roll1 = piano_roll_new[16*(start_section-1):16*(start_section + 4-1)]
print('first four bar tension')
_,result_roll1 = model.manipuate_latent_space(selected_roll1,tensile_up_feature_vector,
                                    diameter_high_feature_vector,
                                    tensile_up_down_feature_vector,
                                    vqvae,tensile_strain_factor1,
                                    diameter_factor1,0,
                                    tensile_strain_direction1,diameter_level1,False,
                                    True,True)



if end_section-start_section>4:
    selected_roll2 = piano_roll_new[16*(start_section + 3):16*(end_section-1)]
    print('second four bar tension')
    _,result_roll2 = model.manipuate_latent_space(selected_roll2,tensile_up_feature_vector,
                                    diameter_high_feature_vector,
                                    tensile_up_down_feature_vector,
                                    vqvae,tensile_strain_factor2,
                                    diameter_factor2,0,
                                    tensile_strain_direction2,diameter_level2,False,
                                    True,True)


    result_roll = np.vstack([result_roll1,result_roll2])
    original_roll = piano_roll_new[16*(start_section-1):16*(end_section-1)]
else:
    result_roll = result_roll1
    original_roll = selected_roll1



pm_original = util.roll_to_pretty_midi(original_roll,pm_old)
pm_new = util.roll_to_pretty_midi(result_roll,pm_old)


