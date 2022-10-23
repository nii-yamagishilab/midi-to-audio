#!/usr/bin/env python3
from espnet2.tasks.mta import MTATask


def get_parser():
    parser = MTATask.get_parser()
    return parser


def main(cmd=None):
    """MTA(MIDI to Audio) training

    Example:

        % python tts_train.py asr --print_config --optim adadelta
        % python tts_train.py --config conf/train_asr.yaml
    """
    MTATask.main(cmd=cmd)


if __name__ == "__main__":
    main()
