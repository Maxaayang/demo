import sys
import warnings
import numpy as np
import librosa
from os import listdir
from os.path import isfile, join


channel_name_list = ["drums", "rest"]  # ,"rest","bass","drums"]
channel_id_list = ["_1", "_3"]  # ,"_3","_2","_1"]
# "trained_20K", "trained_40K", "trained_60K", "trained_80K"]
checkpoint_list = ["trained_1", "trained_20K",
                   "trained_40K", "trained_60K", "trained_80K"]


for channel_name, channel_id in zip(channel_name_list, channel_id_list):
    print(channel_name)
    for file_des in checkpoint_list:
        files = [f for f in listdir("/media/compute/homes/wzaielamri/ai_music/musdb18hq/test"+channel_id+"/")
                 if isfile(join("/media/compute/homes/wzaielamri/ai_music/musdb18hq/test"+channel_id+"/", f))]
        files.sort()

        files_estimate = [f for f in listdir("results_"+channel_name+"/") if (
            isfile(join("results_"+channel_name+"/", f)) and (file_des+"_" in f))]
        files_estimate.sort()

        def sdr(references, estimates):
            # compute SDR for one song
            delta = 1e-7  # avoid numerical errors
            num = np.sum(np.square(references), axis=(1, 2))
            den = np.sum(np.square(references - estimates), axis=(1, 2))
            num += delta
            den += delta
            return 10 * np.log10(num / den)

        all_sdr = []
        for j, file in enumerate(files):

            try:
                x_in, _ = librosa.load("/media/compute/homes/wzaielamri/ai_music/musdb18hq/test" +
                                       channel_id+"/"+file, sr=44100, mono=True, offset=0, duration=None)
                parts = 0  # find how many parts belonging to a specific file
                file = file.replace(channel_id, "_0")
                for ind, i in enumerate(files_estimate):
                    if file in i:
                        parts += 1
                print("file ", j, ": parts: ", parts)

                x_out = np.array([])
                for chunks in range(parts):
                    name_part = "results_"+channel_name + \
                        "/"+file_des+"_" + file+"_"+str(chunks)+".npy"
                    x_part = np.load(name_part)

                    x_part = x_part.reshape(x_part.shape[0]).flatten()

                    x_out = np.concatenate((x_out, x_part))

                sdr_value = sdr(x_in.reshape(
                    1, x_in.shape[0], 1), x_out.reshape(1, x_out.shape[0], 1))
                all_sdr.append(sdr_value)
                #print("parts: ", parts)
                print("sdr ", j, " :", sdr_value)
                print(x_in.shape)
                print(x_out.shape)
            except Exception as e:
                print("problem with: ", j, "error: ", e)
        all_sdr = np.array(all_sdr)
        print("\n", file_des)
        print("\nsdr: ", np.mean(all_sdr))

