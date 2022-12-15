#!/bin/sh
# require parallel and sox

PARALLEL=parallel
PYTHON='/home/smg/v-xuanshi/anaconda3/envs/midi2wav_39/bin/python'

DATADIR=data
scpname=wav.scp

train_set=train
valid_set=validation
test_set=test

for dset in "${test_set}" "${valid_set}" ${train_set}; do
    WAVDIR=${DATADIR}/${dset}/wav_segments
    MIDIDIR=${DATADIR}/${dset}/midifbank_segments
    mkdir -p ${MIDIDIR}
    cat ${DATADIR}/${dset}/${scpname} | ${PARALLEL} ${PYTHON} scripts/midi_fbank/get.py ${WAVDIR}/{/.}.wav ${MIDIDIR}/{/.}.npy
done
