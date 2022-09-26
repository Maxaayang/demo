HPARAMS_REGISTRY = {}
DEFAULTS = {}


class Hyperparams(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


def setup_hparams(hparam_set_names):
    kwargs = {'name': 'vqvae_drums_b4', 'sr': 44100, 'sample_length': 393216, 'bs': 4, 'audio_files_dir': './d_data/*.mid', \
        'labels': False, 'train': True, 'aug_shift': True, 'aug_blend': True}
    H = Hyperparams()
    if not isinstance(hparam_set_names, tuple):
        hparam_set_names = hparam_set_names.split(",")  # ['vqvae']
    hparam_sets = [HPARAMS_REGISTRY[x.strip()]
                   for x in hparam_set_names if x] + [kwargs]
    # hparam_sets = [{'levels': 1, 'downs_t': (1, 2, 2), 'strides_t': (6, 2, 2), 'emb_width': 64, 'l_bins': 2048, 'l_mu': 0.99, \
    #      'commit': 0.02, 'spectral': 0.0, 'multispectral': 1.0, 'hvqvae_multipliers': (2,), 'loss_fn': 'l1',
    #      'lmix_l2': 1.0, 'lmix_linf': 0.02, 'width': 32, 'depth': 4, 'm_conv': 1.0, 'dilation_growth_rate': 3, \
    #         'restore_vqvae': 'https://openaipublic.azureedge.net/jukebox/models/5b/vqvae.pth.tar'}, \
    #             {'name': 'vqvae_drums_b4', 'sr': 44100, 'sample_length': 393216, 'bs': 4, 'audio_files_dir': './d_data/*.mid', \
    #                 'labels': False, 'train': True, 'aug_shift': True, 'aug_blend': True}]
    for k, v in DEFAULTS.items():
        H.update(v)
    for hps in hparam_sets:
        for k in hps:
            if k not in H:
                raise ValueError(f"{k} not in default args")
        H.update(**hps)
    H.update(**kwargs)
    return H


# Teeny for testing
teeny = Hyperparams(
)
HPARAMS_REGISTRY["teeny"] = teeny

easy = Hyperparams(
    sr=22050,
)
HPARAMS_REGISTRY["easy"] = easy

REMOTE_PREFIX = 'https://openaipublic.azureedge.net/'

# Model hps
vqvae = Hyperparams(
    levels=1,  # in the original code 3 levels
    downs_t=(1, 2, 2),
    strides_t=(6, 2, 2),
    emb_width=64,
    l_bins=2048,
    l_mu=0.99,
    commit=0.02,
    spectral=0.0,
    multispectral=1.0,
    hvqvae_multipliers=(2,),  # (2, 1, 1),
    loss_fn='l1',
    lmix_l2=1.0,
    lmix_linf=0.02,
    width=32,
    depth=4,
    m_conv=1.0,
    dilation_growth_rate=3,
    restore_vqvae= REMOTE_PREFIX + 'jukebox/models/5b/vqvae.pth.tar',
)
HPARAMS_REGISTRY["vqvae"] = vqvae

# Small models
small_vqvae = Hyperparams(
    sr=22050,
    levels=1,
    downs_t=(5, 3),
    strides_t=(2, 2),
    emb_width=64,
    l_bins=1024,
    l_mu=0.99,
    commit=0.02,
    spectral=0.0,
    multispectral=1.0,
    loss_fn='l2',
    width=32,
    depth=4,
    m_conv=1.0,
    dilation_growth_rate=3,
)
HPARAMS_REGISTRY["small_vqvae"] = small_vqvae

all_fp16 = Hyperparams(
    fp16=True,
    fp16_params=True,
    fp16_opt=True,
    fp16_scale_window=250,
)
HPARAMS_REGISTRY["all_fp16"] = all_fp16

cpu_ema = Hyperparams(
    ema=True,
    cpu_ema=True,
    cpu_ema_freq=100,
    ema_fused=False,
)
HPARAMS_REGISTRY["cpu_ema"] = cpu_ema


DEFAULTS["rcall"] = Hyperparams(
    rcall_command="<unknown_rcall_command>",
    git_commit="<unknown_git_commit>",
)

DEFAULTS["script"] = Hyperparams(
    name='',
    debug_mem=False,
    debug_eval_files=False,
    debug_speed=False,
    debug_iters=100,
    debug_batch=False,
    debug_grad_accum=False,
    debug_inputs=False,
    local_path='',
    local_logdir='logs',
    max_len=24,
    max_log=32,
    save=True,
    save_iters=20000,
    seed=0,
    prior=False,
    log_steps=100,
    func='',
)

DEFAULTS["data"] = Hyperparams(
    audio_files_dir='',
    finetune='',
    english_only=False,
    bs=1,
    bs_sample=1,
    nworkers=1,
    aug_shift=False,
    aug_blend=False,
    train_test_split=0.9,
    train_shrink_factor=1.0,
    test_shrink_factor=1.0,
    p_unk=0.1,
    min_duration=None,
    max_duration=None,
    n_tokens=0,
    n_vocab=0,
    use_tokens=False,
    curr_epoch=-1,
)

DEFAULTS["vqvae"] = Hyperparams(
    restore_vqvae='',
    levels=2,
    downs_t=(1, 1),
    strides_t=(2, 2),
    hvqvae_multipliers=None,
    revival_threshold=1.0,
    emb_width=64,
    l_bins=512,
    l_mu=0.99,
    commit=1.0,
    spectral=0.0,
    multispectral=1.0,
    loss_fn='l2',
    linf_k=2048,
    lmix_l1=0.0,
    lmix_l2=0.0,
    lmix_linf=0.0,
    use_bottleneck=True,
)

DEFAULTS["vqvae_conv_block"] = Hyperparams(
    depth=3,
    width=128,
    m_conv=1.0,
    dilation_growth_rate=1,
    dilation_cycle=None,
    vqvae_reverse_decoder_dilation=True,
)


DEFAULTS["sample"] = Hyperparams(
    primed_chunk_size=None,
    selected_artists='',
    temp_top=1.0,
    temp_rest=0.99,
    sample_length_in_seconds=24,
    total_sample_length_in_seconds=240,
)

DEFAULTS["opt"] = Hyperparams(
    epochs=20000,
    lr=0.0003,
    clip=1.0,
    beta1=0.9,
    beta2=0.999,
    ignore_grad_norm=0,
    weight_decay=0.0,
    eps=1e-08,
    lr_warmup=100.0,
    lr_decay=10000000000.0,
    lr_gamma=1.0,
    lr_scale=1.0,
    lr_use_linear_decay=False,
    lr_start_linear_decay=0,
    lr_use_cosine_decay=False,
)

DEFAULTS["fp16"] = Hyperparams(
    fp16=False,
    fp16_params=False,
    fp16_loss_scale=None,
    fp16_scale_window=1000.0,
    fp16_opt=False,
)

DEFAULTS["train_test_eval"] = Hyperparams(
    labels=True,
    labels_v3=False,
    dump=False,
    ema=True,
    ema_fused=True,
    cpu_ema=False,
    cpu_ema_freq=100,
    reset_best_loss=False,
    reset_step=False,
    reset_opt=False,
    reset_shd=False,
    train=False,
    test=False,
    sample=False,
    sampler='ancestral',
    codes_logdir='',
    date=None,
    labeller='top_genres',
    label_line=0,
    iters_before_update=1,
    grad_accum_iters=0,
    mu=None,
    piped=False,
    pipe_depth=8,
    break_train=1e10,
    break_test=1e10,
    exit_train=1e10,
)

DEFAULTS["audio"] = Hyperparams(
    n_fft=1024,
    hop_length=256,
    window_size=1024,
    sr=44100,
    channels=2,
    wav='',
    n_inps=1,
    n_hops=2,
    n_segment=1,
    n_total_segment=1,
    n_segment_each=1,
    prime_chunks=4,
    sample_length=0,
    sample_hop_length=30000,
    max_silence_pad_length=0,
    ignore_boundaries=False,
    use_nonrelative_specloss=True,
    multispec_loss_n_fft=(2048, 1024, 512),
    multispec_loss_hop_length=(240, 120, 50),
    multispec_loss_window_size=(1200, 600, 240),
)

DEFAULTS["distributed"] = Hyperparams(
    bucket=128
)
