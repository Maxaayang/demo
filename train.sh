# conda activate vqcpc
# python train.py > tra.log

# Jukebox VQVAE
# mpiexec -n {1}
# python unmix/unmix/train.py --hps=vqvae --name=vqvae_drums_b4 --sr=44100 --sample_length=89 --bs=4 \
# --audio_files_dir="./r_data/*.mid" --labels=False --train --aug_shift --aug_blend > tra.log
# python unmix/unmix/train.py --hps=vqvae --name=vqvae_drums_b4 --sr=44100 --sample_length=89 --bs=4 \
# --audio_files_dir="../data/lmd/*.mid" --labels=False --train --aug_shift --aug_blend > tra.log

python train.py > ./log/vqvae20000.log

# https://www.cnblogs.com/Finley/p/6071463.html

# 各种类型之间的转换
# https://blog.csdn.net/qq_38703529/article/details/120216078

