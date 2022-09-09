import build_model
import util
from preprocess_midi import *

import pickle
import IPython.display as ipd
import os
import pretty_midi
import numpy as np

model = build_model.build_model()
# model.load_weights('model/vae.h5')

#@title # choose a midi file
#@markdown You can choose one of the examples provided here or upload your own midi file. If you upload your own file please make sure that the first track is melody and second track is bass, and the midi should be in C major or A minor.


# def upload_midi():
#     data = files.upload()
#     if len(list(data.keys())) > 1:
#         print('Multiple files uploaded; using only one.')
      
#     return list(data.keys())[0]


# melody = 'example4'  #@param ['example1','example2', 'example3','example4','Upload your midi']

# if melody == 'Upload your midi':
#     melody_file = upload_midi()
# elif melody == 'example1':
#     melody_file = 'data/041ea9c1df8b4163256c8a8a3ffb04dd.mid'
# elif melody == 'example2':
#     melody_file = 'data/631133fa2ae7095bb9113087af86744e.mid'
# elif melody == 'example3':
#     melody_file = 'data/d0264d60827aa635c5bdf44627f4577a.mid'
# elif melody == 'example4':
#     melody_file = 'data/crying_sand.mid'
# else:
#     melody_file = 'data/e6a4afe05f022c891bfe081d4be261db.mid'

melody_file = 'data/e6a4afe05f022c891bfe081d4be261db.mid'

# (12, 64, 89), 12, 
# (8, 64, 89), 8
# (11, 64, 89), 11
piano_roll, bar_indices,pm_old = preprocess_midi(melody_file)
piano_roll_new = np.reshape(piano_roll,(-1,piano_roll.shape[-1]))
pm_new = util.roll_to_pretty_midi(piano_roll_new,pm_old)

print(piano_roll.shape, piano_roll_new.shape)
print(bar_indices)
print(pm_old, pm_new)

# the original file music
ipd.Audio(pm_old.fluidsynth(fs=16000), rate=16000)

# the filtered new file music
# it concatenates 4 bar sections with both melody and bass tracks
ipd.Audio(pm_new.fluidsynth(fs=16000), rate=16000)

# the new file score
# only the first a few bars are showed here
# for full score please download midi files and use other program to view
util.show_score(pm_new)
pm_new.write('./filtered_4bar_concat.mid')
files.download('./filtered_4bar_concat.mid')

#@title select the tension manipulations included

#@markdown the manipulations selected will be randomly applied to every 8 bars
#@markdown 



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

result_roll, tensile_strain, diameter = four_bar_iterate(piano_roll_new,model,
                                                          feature_vectors,
                                                         tensile_strain_factor,
                                                         diameter_factor,
                                                         first_change)


# pm_new = util.roll_to_pretty_midi(piano_roll_new,pm_old)
# print('original score')
# util.show_score(pm_new)
pm_result = util.roll_to_pretty_midi(result_roll,pm_old)
ipd.Audio(pm_result.fluidsynth(fs=16000), rate=16000)

print('new score')
util.show_score(pm_result)

# the changed music
pm_result.write('./all_changed.mid')
files.download('./all_changed.mid')



#@title Please select start and end bar
#@markdown the selected bar length should be 4 bar or 8 bar

start_section = 9 #@param {type:"integer"}
end_section = 17 #@param {type:"integer"}



#@title select the tension manipulation
#@markdown first 4 bar

tensile_strain_direction1 = False #@param ["False", "True"] {type:"raw"}
tensile_strain_factor1 = 5.5 #@param {type:"slider", min:-8, max:8, step:0.5}

diameter_level1 = True #@param ["False", "True"] {type:"raw"}
diameter_factor1 = 4 #@param {type:"slider", min:-4, max:4, step:0.5}

#@markdown second 4 bar (optional)

tensile_strain_direction2 = True #@param ["False", "True"] {type:"raw"}
tensile_strain_factor2 = -6 #@param {type:"slider", min:-8, max:8, step:0.5}

diameter_level2 = False #@param ["False", "True"] {type:"raw"}
diameter_factor2 = -4 #@param {type:"slider", min:-4, max:4, step:0.5}



selected_roll1 = piano_roll_new[16*(start_section-1):16*(start_section + 4-1)]
print('first four bar tension')
_,result_roll1 = build_model.manipuate_latent_space(selected_roll1,tensile_up_feature_vector,
                                    diameter_high_feature_vector,
                                    tensile_up_down_feature_vector,
                                    model,tensile_strain_factor1,
                                    diameter_factor1,0,
                                    tensile_strain_direction1,diameter_level1,False,
                                    True,True)



if end_section-start_section>4:
    selected_roll2 = piano_roll_new[16*(start_section + 3):16*(end_section-1)]
    print('second four bar tension')
    _,result_roll2 = build_model.manipuate_latent_space(selected_roll2,tensile_up_feature_vector,
                                    diameter_high_feature_vector,
                                    tensile_up_down_feature_vector,
                                    model,tensile_strain_factor2,
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



ipd.Audio(pm_new.fluidsynth(fs=16000), rate=16000)



ipd.Audio(pm_original.fluidsynth(fs=16000), rate=16000)



print('original score')
util.show_score(pm_original)
print('new score')
util.show_score(pm_new)
