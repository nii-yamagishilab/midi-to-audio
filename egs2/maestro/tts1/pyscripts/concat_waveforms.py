import glob
import os
import sys
from collections import defaultdict

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import scipy.io.wavfile
import scipy.sparse
import soundfile as sf
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func
from natsort import natsorted
from scipy import signal


def correlation_lags(in1_len, in2_len, mode='full'):
    r"""
    Calculates the lag / displacement indices array for 1D cross-correlation.
    Parameters
    ----------
    in1_size : int
        First input size.
    in2_size : int
        Second input size.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `correlate` for more information.
    See Also
    --------
    correlate : Compute the N-dimensional cross-correlation.
    Returns
    -------
    lags : array
        Returns an array containing cross-correlation lag/displacement indices.
        Indices can be indexed with the np.argmax of the correlation to return
        the lag/displacement.
    Notes
    -----
    Cross-correlation for continuous functions :math:`f` and :math:`g` is
    defined as:
    .. math::
        \left ( f\star g \right )\left ( \tau \right )
        \triangleq \int_{t_0}^{t_0 +T}
        \overline{f\left ( t \right )}g\left ( t+\tau \right )dt
    Where :math:`\tau` is defined as the displacement, also known as the lag.
    Cross correlation for discrete functions :math:`f` and :math:`g` is
    defined as:
    .. math::
        \left ( f\star g \right )\left [ n \right ]
        \triangleq \sum_{-\infty}^{\infty}
        \overline{f\left [ m \right ]}g\left [ m+n \right ]
    Where :math:`n` is the lag.
    Examples
    --------
    Cross-correlation of a signal with its time-delayed self.
    >>> from scipy import signal
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> x = rng.standard_normal(1000)
    >>> y = np.concatenate([rng.standard_normal(100), x])
    >>> correlation = signal.correlate(x, y, mode="full")
    >>> lags = signal.correlation_lags(x.size, y.size, mode="full")
    >>> lag = lags[np.argmax(correlation)]
    """

    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = np.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid-lag_bound):(mid+lag_bound)]
        else:
            lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags


def frame_time_idx_range(frame_idx, frame_length, frame_shift):
    # a, b = frame_time_idx_range(frame_idx, frame_length, frame_shift)
    #  a: start time index of the frame
    #  b: end time index of the frame
    return frame_idx * frame_shift, frame_idx * frame_shift + frame_length


def get_window(length):
    return np.hanning(length)


def concatenate_samples(wav_collections):
    # set variables
    frame_shift = 288
    frame_length = 1200
    buffer_frames = 5
    buffer_length = (buffer_frames - 1) * frame_shift + frame_length
    overlap_length = frame_shift + frame_length
    #m_overlapadder = OverlapAdder(frame_length, frame_shift)

    # set input variables
    # wav_collections = [wav_0.copy(), wav_1.copy(), wav_2.copy()]

    # buffer
    wav_pre = wav_collections[0]
    for wav_pos in wav_collections[1:]:
        
        wav_pre_base = wav_pre[:-buffer_length]
        wav_pos_base = wav_pos[buffer_length:]
        wav_pre_buffer = wav_pre[-buffer_length:]
        wav_pos_buffer = wav_pos[:buffer_length]
        wav_pre_overlap = wav_pre[-overlap_length:]
        wav_pos_overlap = wav_pos[:overlap_length]
        
        ### compute lag between overlap part
        correlation = signal.correlate(wav_pre_overlap, wav_pos_overlap, mode="same")
        lags = correlation_lags(wav_pre_overlap.size, wav_pos_overlap.size, mode="same")
        lag = lags[np.argmax(correlation)]
        
        ### add overlap
        add_buffer_length = 2 * buffer_length - overlap_length + lag
        add_buffer = np.zeros(add_buffer_length)
        overlap_length_calibrated = overlap_length - lag
        
        window = get_window(overlap_length_calibrated * 2)
        
        # Just add the window and do overlap-add
        wav_pre_buffer[-overlap_length_calibrated:] = wav_pre_buffer[-overlap_length_calibrated:] * window[overlap_length_calibrated:]
        wav_pos_buffer[:overlap_length_calibrated] = wav_pos_buffer[:overlap_length_calibrated] * window[:overlap_length_calibrated]
        add_buffer[:buffer_length] += wav_pre_buffer
        add_buffer[-buffer_length:] += wav_pos_buffer
        
        wav_pre = np.concatenate([wav_pre_base, add_buffer, wav_pos_base])

    return wav_pre


def main(input_seg_dir):
    file_list = glob.glob(os.path.join(input_seg_dir, '*.wav'))
    file_tar_dir = input_seg_dir.rsplit('/', 1)[0]
    if not os.path.isdir(os.path.join(file_tar_dir, 'concat')):
        os.mkdir(os.path.join(file_tar_dir, 'concat'))
    utt_seg_dict = defaultdict(list)
    for item in file_list:
        utt_id = os.path.basename(item).rsplit('_', 1)[0]
        utt_seg_dict[utt_id].append(item)

    for k, v in utt_seg_dict.items():
        sorted_wav_list = natsorted(v)
        wav_collections = [librosa.core.load(item, sr=24000)[0] for item in sorted_wav_list]
        concated_wave = concatenate_samples(wav_collections)
        sf.write(
            f"{file_tar_dir}/concat/{k}.wav",
            concated_wave,
            24000,
            "PCM_16",
        )
        print('{} has been concatenated.'.format(k))


if __name__ == '__main__':
    input_segments_dir = sys.argv[1]  # eg: 'maestro_joint_newmidi_nsfgan/overlap_1/'
    main(input_segments_dir)
