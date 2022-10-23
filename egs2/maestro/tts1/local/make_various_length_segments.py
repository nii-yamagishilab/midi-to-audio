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
This script is to generate various length of segments for inference.

Working Directory:
    |- data/
    |-- data/test_various/
    |--- data/test_various/audio_org_pool/xx.wav
    |--- data/test_various/text_org_pool/xx.midi
    
    |- dump/raw/
    |-- dump/raw/test_various/
    |--- dump/raw/test_various/audio_org_pool/xx.wav
    |--- dump/raw/test_various/text_org_pool/xx.midi

"""

SAMPLE_RATE = 24000
FRAME_LENGTH_MS = 50
FRAME_SHIFT_MS = 12

NUM_FRAME_MIN = 190
NUM_FRAME_STEP = 50

def main(in_wav_dir, in_midi_dir, out_data_dir, out_dump_dir):
    assert os.path.isfile(in_wav_dir)
    assert os.path.isfile(in_midi_dir)

    os.makedirs(os.path.join(out_data_dir, 'wav_segments'), exist_ok = True)
    os.makedirs(os.path.join(out_data_dir, 'text_segments'), exist_ok = True)
    os.makedirs(out_dump_dir, exist_ok = True)

    up_sample_rate = FRAME_SHIFT_MS / 1000 * SAMPLE_RATE
    frame_length_point = int(FRAME_LENGTH_MS / 1000 * SAMPLE_RATE)
    frame_shift_point = int(FRAME_SHIFT_MS / 1000 * SAMPLE_RATE)
    # segment_length_point = (args.num_segment_frame - 1) * frame_shift_point + frame_length_point

    try:
        mid_ob = pretty_midi.PrettyMIDI(in_midi_dir)
    except:
        if not os.path.isfile(in_midi_dir):
            raise ValueError('cannot find midifile from {}'.format(in_midi_dir))
        else:
            raise ValueError('cannot read midifile from {}'.format(in_midi_dir))
    if len(mid_ob.instruments) > 1:
        print("Track has >1 instrument %s" % (in_midi_dir))
    midi = mid_ob.get_piano_roll(fs=SAMPLE_RATE/up_sample_rate)

    try:
        wav = librosa.core.load(in_wav_dir, sr=SAMPLE_RATE)[0]
    except:
        raise ValueError('cannot read waveform from {}'.format(in_wav_dir))

    i = 0
    wav_begin_point = 0
    wav_end_point = int(NUM_FRAME_MIN * frame_shift_point + frame_length_point)
    cur_begin_frame = 0
    cur_end_frame = NUM_FRAME_MIN
    wav_id = os.path.basename(in_wav_dir)[:-4]
    utt_id_list = []
    utt2num_samples = []

    while wav_end_point < wav.shape[0]:
        utt_id = "{}_{}".format(wav_id, i)

        wav_segment = wav[wav_begin_point:wav_end_point]
        midi_segment = midi[:, cur_begin_frame:cur_end_frame]
        sparse_midi_segment = scipy.sparse.csc_matrix(midi_segment)

        wav_segment_dir = os.path.join(out_data_dir, 'wav_segments', utt_id + '.wav')
        midi_segment_dir = os.path.join(out_data_dir, 'text_segments', utt_id + '.npz')

        scipy.io.wavfile.write(wav_segment_dir, SAMPLE_RATE, wav_segment)
        scipy.sparse.save_npz(midi_segment_dir, sparse_midi_segment)

        utt_id_list.append(utt_id)
        utt2num_samples.append("{} {}\n".format(utt_id, wav_end_point - wav_begin_point))

        interval_frame = cur_end_frame - cur_begin_frame
        cur_begin_frame = cur_end_frame
        cur_end_frame = cur_end_frame + interval_frame + NUM_FRAME_STEP
        wav_begin_point = int(cur_begin_frame * frame_shift_point)
        wav_end_point = int(cur_end_frame * frame_shift_point + frame_length_point)
        i += 1

    f_wav = open(os.path.join(out_data_dir, 'wav.scp'), 'w')
    f_text = open(os.path.join(out_data_dir, 'text'), 'w')
    f_spk2utt = open(os.path.join(out_data_dir, 'spk2utt'), 'w')
    f_utt2spk = open(os.path.join(out_data_dir, 'utt2spk'), 'w')
    f_utt2num_samples = open(os.path.join(out_dump_dir, 'utt2num_samples'), 'w')
    f_uttid = open(os.path.join(out_dump_dir, 'utt'), 'w')

    f_spk2utt.write("piano ")
    for j in range(len(utt_id_list)):
        f_wav.write("{} {}\n".format(utt_id_list[j],
                                     os.path.join(out_data_dir, 'wav_segments', utt_id_list[j] + '.wav')))
        f_text.write("{} {}\n".format(utt_id_list[j],
                                      os.path.join(out_data_dir, 'text_segments', utt_id_list[j] + '.npz')))
        f_utt2spk.write("{} {}\n".format(utt_id_list[j], 'piano'))
        f_spk2utt.write("{} ".format(utt_id_list[j]))
        f_utt2num_samples.write(utt2num_samples[j])
        f_uttid.write("{}\n".format(utt_id_list[j]))

    f_wav.close()
    f_text.close()
    f_spk2utt.close()
    f_utt2spk.close()
    f_utt2num_samples.close()
    f_uttid.close()


if __name__ == '__main__':
    in_wav_dir = 'data/test_various/audio_org_pool/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--1.wav'
    in_midi_dir = 'data/test_various/text_org_pool/MIDI-UNPROCESSED_01-03_R1_2014_MID--AUDIO_01_R1_2014_wav--1.midi'
    out_data_dir = 'data/test_various'
    out_dump_dir = 'dump/raw/test_various'
    main(in_wav_dir, in_midi_dir, out_data_dir, out_dump_dir)
