#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

fs=24000
n_fft=32768
# n_fft=8192
n_shift=288
win_length=1200

opts=
if [ "${fs}" -eq 24000 ]; then
    # To suppress recreation, specify wav format
    opts="--audio_format wav "
else
    opts="--audio_format flac "
fi

train_set=train
valid_set=validation
test_sets="test"

train_config=conf/train.yaml
inference_config=conf/decode.yaml

# g2p=g2p_en # Include word separator
g2p=g2p_en_no_space # Include no word separator

./midi_to_wav.sh \
    --lang en \
    --feats_type raw \
    --feats_extract midifbank \
    --audio_format wav \
    --fs "${fs}" \
    --fmin 5 \
    --fmax 12000 \
    --n_mels 128 \
    --n_fft "${n_fft}" \
    --n_shift "${n_shift}" \
    --win_length "${win_length}" \
    --token_type proll \
    --cleaner none \
    --g2p "${g2p}" \
    --train_config "${train_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" \
    --stage 1 \
    --stop_stage 7 \
    --ngpu 1 \
    --tts_task mta \
    --gpu_inference true \
    ${opts} "$@"
