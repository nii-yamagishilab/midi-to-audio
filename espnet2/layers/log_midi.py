import numpy as np
import librosa
import torch
from typing import Tuple

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask

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


class LogMIDI(torch.nn.Module):
    """Convert STFT to fbank feats

    The arguments is same as librosa.filters.midi

    Args:
        fs: number > 0 [scalar] sampling rate of the incoming signal
        n_fft: int > 0 [scalar] number of FFT components
        n_mels: int > 0 [scalar] number of Mel bands to generate
        fmin: float >= 0 [scalar] lowest frequency (in Hz)
        fmax: float >= 0 [scalar] highest frequency (in Hz).
            If `None`, use `fmax = fs / 2.0`
        htk: use HTK formula instead of Slaney
        midi_filter_bank_dir: the directory of the midi filterbank
    """

    def __init__(
        self,
        fs: int = 16000,
        n_fft: int = 512,
        n_mels: int = 80,
        fmin: float = None,
        fmax: float = None,
        htk: bool = False,
        log_base: float = None,
        midi_filter_bank_dir: str = 'midi_filter_4.bank',
    ):
        super().__init__()

        fmin = 0 if fmin is None else fmin
        fmax = fs / 2 if fmax is None else fmax
        _mel_options = dict(
            sr=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
        )
        self.mel_options = _mel_options
        self.log_base = log_base

        # Note(kamo): The mel matrix of librosa is different from kaldi.
        # melmat = librosa.filters.mel(**_mel_options)
        n_freq = n_fft // 2 + 1
        midimat = read_raw_mat(midi_filter_bank_dir, n_freq)
        # melmat: (D2, D1) -> (D1, D2)
        # self.register_buffer("melmat", torch.from_numpy(melmat.T).float())
        self.register_buffer("midimat", torch.from_numpy(midimat.T).float())

    def extra_repr(self):
        return ", ".join(f"{k}={v}" for k, v in self.mel_options.items())

    def forward(
        self,
        feat: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # feat: (B, T, D1) x melmat: (D1, D2) -> mel_feat: (B, T, D2)
        midi_feat = torch.matmul(feat, self.midimat)
        midi_feat = torch.clamp(midi_feat, min=1e-5)

        if self.log_base is None:
            logmidi_feat = midi_feat.log()
        elif self.log_base == 2.0:
            logmidi_feat = midi_feat.log2()
        elif self.log_base == 10.0:
            logmidi_feat = midi_feat.log10()
        else:
            logmidi_feat = midi_feat.log() / torch.log(self.log_base)

        # logmidi_feat = 20 * logmidi_feat - 20
        # Zero padding
        if ilens is not None:
            logmidi_feat = logmidi_feat.masked_fill(
                make_pad_mask(ilens, logmidi_feat, 1), 0.0
            )
        else:
            ilens = feat.new_full(
                [feat.size(0)], fill_value=feat.size(1), dtype=torch.long
            )
        return logmidi_feat, ilens
