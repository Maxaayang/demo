from re import A


rnn_dim = 96
input_dim = 89

time_step = 64

start_middle_dim = 64
melody_bass_dense_1_dim = 128

melody_output_dim = 74  # 63
melody_note_start_dim = 1
bass_output_dim = 13
bass_note_start_dim = 1

tension_middle_dim = 128
tension_output_dim = 1

z_dim = 96

TEMPO = 90
melody_dim = melody_output_dim
bass_dim = bass_output_dim
velocity = 100

SAMPLES_PER_BAR = 16
SEGMENT_BAR_LENGTH = 4
SLIDING_WINDOW=SEGMENT_BAR_LENGTH

train_size = 60000
test_size = 10000
# epochs = 20000
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2
num_examples_to_generate = 10

# BASE_PATH = 'lmd_matched\\A\\A\\A\\TRAAAZF12903CCCF6B\\*.mid'
# BASE_PATH="./data/r_data/*.mid"
# BASE_PATH="./data/m_data/data/mid/*.mid"
BASE_PATH="../data/lmd/*.mid"
DATA_PATH = 'jazz'
SAVE_MODEL_NAME =  "model.pth"

DEBUG = True

# note feature: pitch, step, duration
batch_size = 64
sequence_lenth = 1
max_file_num = None
epochs = 20000
learning_rate = 0.005

loss_weight = [0.1, 20.0, 1.0]

# VectorQuantizer
in_channels = 5
num_embeddings = 2000
embedding_dim = 96
use_codebook_loss = True
beta = 0.02 # 1