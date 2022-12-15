#!/usr/bin/env python3
from espnet2.tasks.gan_mta import GANMTATask


def get_parser():
    parser = GANMTATask.get_parser()
    return parser


def main(cmd=None):
    """GAN-based MTA training

    Example:

        % python gan_mta_train.py --print_config --optim1 adadelta
        % python gan_mta_train.py --config conf/train.yaml
    """
    GANMTATask.main(cmd=cmd)


if __name__ == "__main__":
    main()
