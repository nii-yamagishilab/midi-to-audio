import os
import argparse
import librosa
import pretty_midi
import scipy
import scipy.io.wavfile
import scipy.sparse
import numpy as np
import multiprocessing
"""

"""

FRAME_LENGTH_MS = 50
FRAME_SHIFT_MS = 12


def segment_func(midi_wav_args_zip):
    midi_item = midi_wav_args_zip[0]
    wav_item = midi_wav_args_zip[1]
    args = midi_wav_args_zip[2]

    wav_id, wav_dir = wav_item.strip().split()
    midi_id, midi_dir = midi_item.strip().split()
    assert wav_id == midi_id

    up_sample_rate = FRAME_SHIFT_MS / 1000 * args.sample_rate
    frame_length_point = int(FRAME_LENGTH_MS / 1000 * args.sample_rate)
    frame_shift_point = int(FRAME_SHIFT_MS / 1000 * args.sample_rate)
    segment_length_point = (args.num_segment_frame - 1) * frame_shift_point + frame_length_point

    try:
        mid_ob = pretty_midi.PrettyMIDI(midi_dir)
    except:
        if not os.path.isfile(midi_dir):
            raise ValueError('cannot find midifile from {}'.format(wav_dir))
        else:
            raise ValueError('cannot read midifile from {}'.format(wav_dir))
    if len(mid_ob.instruments) > 1:
        print("Track has >1 instrument %s" % (midi_dir))
    midi = mid_ob.get_piano_roll(fs=args.sample_rate/up_sample_rate)

    try:
        wav = librosa.core.load(wav_dir, sr=args.sample_rate)[0]
    except:
        raise ValueError('cannot read waveform from {}'.format(wav_dir))
    # time = librosa.get_duration(wav, sr=args.sample_rate)

    utt_id_list = []
    print('split uttid: {}'.format(wav_id))
    i = 0
    while (i+1)*segment_length_point < wav.shape[0]:
        utt_id = "{}_{}".format(wav_id, i)
        wav_segment_dir = os.path.join(args.wav_segments_dir, utt_id + '.wav')
        midi_segment_dir = os.path.join(args.text_segments_dir, utt_id + '.npz')
        # wav_segments_list.append('{} {}\n'.format(utt_id, wav_segment_dir))
        # midi_segments_list.append('{} {}\n'.format(utt_id, midi_segment_dir))
        utt_id_list.append(utt_id)

        wav_begin_point = int(args.num_segment_frame * frame_shift_point * i)
        wav_end_point = int(args.num_segment_frame * frame_shift_point * (i + 1) + frame_length_point)
        wav_segment = wav[wav_begin_point:wav_end_point]
        midi_segment = midi[:, int(args.num_segment_frame*i):int(args.num_segment_frame*(i+1))]
        sparse_midi_segment = scipy.sparse.csc_matrix(midi_segment)
        
        if not os.path.isdir(os.path.dirname(wav_segment_dir)):
            os.makedirs(os.path.dirname(wav_segment_dir))
        if not os.path.isdir(os.path.dirname(midi_segment_dir)):
            os.makedirs(os.path.dirname(midi_segment_dir))
        scipy.io.wavfile.write(wav_segment_dir, args.sample_rate, wav_segment)
        scipy.sparse.save_npz(midi_segment_dir, sparse_midi_segment)

        i += 1
    return utt_id_list


def main():
    # parser
    parser = argparse.ArgumentParser(description='generate segments')
    parser.add_argument('--wav_dir', type=str, default='',
                        help='directory of wav.scp')
    parser.add_argument('--wav_segments_dir', type=str, default='',
                        help='directory of segments for wav')
    parser.add_argument('--text_dir', type=str, default='',
                        help='directory of text (directory of midi)')
    parser.add_argument('--text_segments_dir', type=str, default='',
                        help='directory of segments for text')
    parser.add_argument('--sample_rate', type=int, default=24000,
                        help='number of sample rate')
    parser.add_argument('--num_segment_frame', type=float, default=800,
                        help='number of frames in each segment')
    parser.add_argument('--begin_cut', type=float, default=2,
                        help='the beginning of the frame')
    parser.add_argument('--end_cut', type=float, default=2,
                        help='the end of the frame')
    args = parser.parse_args()
    
    f_wav = open(args.wav_dir, 'r')
    lines_wav = f_wav.readlines()

    f_midi = open(args.text_dir, 'r')
    lines_midi = f_midi.readlines()

    wav_segments_list = []
    midi_segments_list = []

    midi_wav_args_list = []
    for wav_item, midi_item in zip(lines_wav, lines_midi):
        wav_id, wav_dir = wav_item.strip().split()
        midi_id, midi_dir = midi_item.strip().split()
        assert wav_id == midi_id
        midi_wav_args_list.append([midi_item, wav_item, args])

    pool = multiprocessing.Pool(processes=8)
    utt_id_all_list = pool.map(segment_func, midi_wav_args_list)
    utt_id_all_list_flatten = [utt_id for utt_id_list in utt_id_all_list for utt_id in utt_id_list]

    # write "wav_segments.scp" & "text_segments 0"
    utt_id_all_list_flatten.sort()
    for utt_id in utt_id_all_list_flatten:
        wav_segment_dir = os.path.join(args.wav_segments_dir, utt_id + '.wav')
        midi_segment_dir = os.path.join(args.text_segments_dir, utt_id + '.npz')
        wav_segments_list.append('{} {}\n'.format(utt_id, wav_segment_dir))
        midi_segments_list.append('{} {}\n'.format(utt_id, midi_segment_dir))

    wav_seg_dir = os.path.join(
        os.path.dirname(args.wav_dir), 'wav_segments.scp'
    )
    midi_set_dir = os.path.join(
        os.path.dirname(args.text_dir), 'text_segments.scp'
    )
    f_wav_segments = open(wav_seg_dir, 'w')
    f_midi_segments = open(midi_set_dir, 'w')
    for wav_seg, midi_seg in zip(wav_segments_list, midi_segments_list):
        f_wav_segments.write(wav_seg)
        f_midi_segments.write(midi_seg)
    f_wav_segments.close()
    f_midi_segments.close()


if __name__ == "__main__":
    main()