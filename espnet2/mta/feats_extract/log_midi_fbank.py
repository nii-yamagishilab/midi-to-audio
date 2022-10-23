from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import humanfriendly
import torch
from typeguard import check_argument_types

from espnet2.layers.log_midi import LogMIDI
from espnet2.layers.stft import Stft
from espnet2.mta.feats_extract.abs_feats_extract import AbsFeatsExtract


class LogMIDIFbank(AbsFeatsExtract):
    """Conventional frontend structure for TTS.

    Stft -> amplitude-spec -> Log-MIDI-Fbank
    """

    def __init__(
        self,
        fs: Union[int, str] = 24000,
        n_fft: int = 32768,
        win_length: int = 1200,
        hop_length: int = 288,
        window: Optional[str] = "hann",
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        n_mels: int = 128,
        midi_filter_bank_dir: str = 'midi_filter_4.bank',
        fmin: Optional[int] = 5,
        fmax: Optional[int] = 12000,
        htk: bool = False,
        log_base: Optional[float] = 10.0,
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.fs = fs
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.fmin = fmin
        self.fmax = fmax

        self.stft = Stft(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            window=window,
            center=center,
            normalized=normalized,
            onesided=onesided,
        )

        # NOTE (Xuan): hard code to assign midi filter bank
        if n_fft == 32768:
            midi_filter_bank_dir = 'midi_filter_4.bank'
        elif n_fft == 8192:
            midi_filter_bank_dir = 'midi_filter.bank'
        else:
            raise NotImplementedError

        self.logmidi = LogMIDI(
            fs=fs,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax,
            htk=htk,
            log_base=log_base,
            midi_filter_bank_dir=midi_filter_bank_dir,
        )

    def output_size(self) -> int:
        return self.n_mels

    def get_parameters(self) -> Dict[str, Any]:
        """Return the parameters required by Vocoder"""
        return dict(
            fs=self.fs,
            n_fft=self.n_fft,
            n_shift=self.hop_length,
            window=self.window,
            n_mels=self.n_mels,
            win_length=self.win_length,
            fmin=self.fmin,
            fmax=self.fmax,
        )

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Domain-conversion: e.g. Stft: time -> time-freq
        input_stft, feats_lens = self.stft(input, input_lengths)
        assert input_stft.dim() >= 4, input_stft.shape
        # "2" refers to the real/imag parts of Complex
        assert input_stft.shape[-1] == 2, input_stft.shape

        # NOTE(kamo): We use different definition for log-spec between TTS and ASR
        #   TTS: log_10(abs(stft))
        #   ASR: log_e(power(stft))

        # input_stft: (..., F, 2) -> (..., F)
        input_power = input_stft[..., 0] ** 2 + input_stft[..., 1] ** 2
        input_amp = torch.sqrt(torch.clamp(input_power, min=1.0e-10))
        # input_amp = input_stft.abs()
        input_feats, _ = self.logmidi(input_amp, feats_lens)
        return input_feats, feats_lens
