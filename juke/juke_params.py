input_shape = (96, 89)
levels = 1
downs_t = (3, 2, 2)
strides_t = (2, 2, 2)
emb_width = 96
l_bins = 2048
mu = 0.99
commit = 0.02
spectral = 0.0
multispectral = 1.0
multipliers = (2,)
use_bottleneck = True
width = 32
depth = 4
m_conv = 1.0
dilation_growth_rate = 3
dilation_cycle = None
reverse_decoder_dilation = True

lr = 0.05
epochs = 20000
max_file_num = None
BASE_PATH = '../../data/lmd/**.mid'
sequence_lenth = 1
batch_size = 64

SAMPLES_PER_BAR = 16
SEGMENT_BAR_LENGTH = 4
SLIDING_WINDOW=SEGMENT_BAR_LENGTH

embedding_dim = 96
rnn_dim = 92
input_dim = 89

melody_output_dim = 74  # 63
melody_note_start_dim = 1
bass_output_dim = 13
bass_note_start_dim = 1
tension_middle_dim = 128
tension_output_dim = 1

# demo.params
