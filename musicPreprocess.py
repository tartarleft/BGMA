import glob
import random
import json
import librosa
import numpy as np
from pydub import AudioSegment
import os
import soundfile as sf
from tqdm import tqdm

random.seed(0)

with open('./config/music_basics.json') as config_file:
    settings = json.load(config_file)

musan_classical_path = glob.glob(settings["musan_classical"])


def read_mp3(file_name, SR=16000):
    song = AudioSegment.from_mp3(file_name)
    sr = song.frame_rate
    data = song.get_array_of_samples()
    data = np.array(data)
    data = np.divide(data, np.max(np.abs(data)), dtype='float32')  # making the floating numbers
    if len(data.shape) != 1:
        data = np.mean(data, axis=1, dtype='float32')
    if sr != SR:  # librosa: to_mono then resample
        print(data.shape)
        data = librosa.resample(data, sr, SR)
        sr = SR
    data_norm = np.divide(data, np.max(np.abs(data)), dtype='float32')
    return data_norm, sr


def get_musics(music_path=musan_classical_path):
    print('Generating Music Slices...')
    # mpath, sr = music_path[:50], settings["sr"]
    mpath, sr = music_path, settings["sr"]
    musics = []
    for msc in mpath:
        utter = librosa.load(msc, sr)[0]
        scale_rate = np.max([np.abs(utter.max()), np.abs(utter.min())])
        musics.append(utter / scale_rate)
    print('Generating Music Slices Done')
    return musics


def get_encoder_music(style='piano'):
    if style == 'piano':
        mpath = glob.glob(settings["piano_music"])
    elif style == 'guitar':
        mpath = glob.glob(settings["guitar_music"])
    sr = settings['sr']
    musics = []
    for msc in mpath:
        musics.append(read_mp3(msc, sr)[0])
    return musics

