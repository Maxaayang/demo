import build_model
import util
from preprocess_midi import *

import pickle
import IPython.display as ipd
import os
import pretty_midi
import numpy as np
# import sonnet as snt
from sonnet.src.nets import vqvae

util.setup_musescore()
model = build_model.build_model()
model.load_weights('model/vae.h5')

# snt.src.nets.vae