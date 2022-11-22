#!/usr/bin/env bash

set -e
set -u
# set -x
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
SECONDS=0

stage=-1
stop_stage=2

sample_rate=16000
num_segment_frame=800

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;
. ./db.sh || exit 1;

# ROOT=`pwd | sed 's%\(.*/REPO\)/.*%\1%'`
# maestro_root="${ROOT}/../DATA/maestro/Google_maestro-v2.0.0"
# maestro_root="${MAESTRO}/maestro/MIDI-filterbank"
db_root=${MAESTRO}

train_set="train"
valid_set="validation"
test_sets="test"

if [ ${stage} -le -1 ] && [ ${stop_stage} -ge -1 ]; then
    # generate downloads/maestro-v3.0.0
    log "stage -1: Data Download"
    local/data_download.sh "${db_root}"
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    log "stage 0: Split to Subsets"
    # split data into data/{train/validation/test} dataset
    # generate wav.scp & text in each dset
    # wav.scp: wav_id wav_dir
    # text: wav_id midi_dir
    [ -e data ] && rm -r data
    python local/data_parse.py "${db_root}" data
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    log "stage 1: Generate Segmentation for both midi & audio"
    # make segments (wav_segments.scp, wav_segments, text_segments.scp, text_segments)
    # wav_segments.scp (eg: segid data/{train/val/text}/wav_segments/{segid}.wav)
    # text_segments.scp (eg: segid data/{train/val/text}/text_segments/{segid}.npz)

    for dset in "${train_set}" "${valid_set}" ${test_sets}; do
        python local/data_segments.py --wav_dir data/"${dset}"/wav.scp \
            --wav_segments_dir data/"${dset}"/wav_segments \
            --text_dir data/"${dset}"/text \
            --text_segments_dir data/"${dset}"/text_segments \
            --sample_rate ${sample_rate} \
            --num_segment_frame ${num_segment_frame}
    done

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    log "stage 2: Generate utt2spk & spk2utt"
    # make utt2spk & spk2utt
    for dset in "${train_set}" "${valid_set}" ${test_sets}; do
        utt2spk=data/"${dset}"/utt2spk
        spk2utt=data/"${dset}"/spk2utt
        [ -e ${utt2spk} ] && rm ${utt2spk}
        [ -e ${spk2utt} ] && rm ${spk2utt}

        mv data/"${dset}"/wav.scp data/"${dset}"/wav_original.scp
        mv data/"${dset}"/text data/"${dset}"/text_original
        mv data/"${dset}"/wav_segments.scp data/"${dset}"/wav.scp
        mv data/"${dset}"/text_segments.scp data/"${dset}"/text

        python local/generate_utt2spk.py data/"${dset}"/wav.scp data/"${dset}"/utt2spk
        utils/utt2spk_to_spk2utt.pl ${utt2spk} > ${spk2utt}
    done
fi


log "Successfully finished. [elapsed=${SECONDS}s]"
