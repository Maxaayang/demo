python -m ipdb unmix/unmix/train.py --hps=vqvae --name=vqvae_drums_b4 --sr=44100 --sample_length=393216 --bs=4 \
--audio_files_dir="./d_data/*.mid" --labels=False --train --aug_shift --aug_blend