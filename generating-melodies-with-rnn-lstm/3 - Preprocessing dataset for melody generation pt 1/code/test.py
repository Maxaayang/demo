import music21 as m21
import os

PATH = "/home/u21s052015/code/data/lmd"

def load_data(path):
    songs = []
    for path, subdirs, files in os.walk(path):
        for file in files:

            # consider only kern files
            if file[-3:] == "mid":
                song = m21.converter.parse(os.path.join(path, file))
                songs.append(song)
    return songs

if __name__ == "__main__":
    songs = load_data(PATH)