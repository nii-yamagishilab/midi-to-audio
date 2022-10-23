#!/usr/bin/python3
"""
Extract Mel-spectrogram using Librosa

Example:
$: python3 00_get.py ../sample-data/HAW-001.wav HAW-001.mel  

Usage:
1. Modify the configuration in hparams.py
2. run python 00_get.py input_waveform output_mel

"""
import os
import numpy as np
import sys
import librosa

# from hparams import Hparams_class
# from audio import Audio

def read_raw_mat(filename,col,format='f4',end='l'):
    f = open(filename,'rb')
    if end=='l':
        format = '<'+format
    elif end=='b':
        format = '>'+format
    else:
        format = '='+format
    datatype = np.dtype((format,(col,)))
    data = np.fromfile(f,dtype=datatype)
    f.close()
    if data.ndim == 2 and data.shape[1] == 1:
        return data[:,0]
    else:
        return data

def wav_to_spec(wav):
    stft_spec = librosa.stft(y=wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # mel_spec = 20 * np.log10(np.maximum(1e-5, np.dot(mel_basis, np.abs(stft_spec)))) - 20.0
    mel_spec = np.log10(np.maximum(1e-5, np.dot(mel_basis, np.abs(stft_spec))))
    mel_spec = mel_spec.astype(np.float32)
    return mel_spec


if __name__ == "__main__":
    midi_filter_bank_dir = 'midi_filter_4.bank'

    n_freq = (4096 * 2 * 4 // 2) + 1
    n_fft = (n_freq - 1) * 2
    hop_length = int(12 / 1000 * 24000)
    win_length = int(50 / 1000 * 24000)
    mel_basis = read_raw_mat(midi_filter_bank_dir, n_freq)

    try:
        input_wav = sys.argv[1]
        output_mel = sys.argv[2]
    except IndexError:
        input_wav = "HAW-001.wav" 
        output_mel = "HAW-001.npy"
        pass

    wav = librosa.core.load(input_wav, sr=24000)[0]
    midi_spec = wav_to_spec(wav).T
    midi_spec.tofile(output_mel, format="<f4") # shape: (time, feat)


