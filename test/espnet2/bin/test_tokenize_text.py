from argparse import ArgumentParser

import pytest

from espnet2.bin.tokenize_text import get_parser
from espnet2.bin.tokenize_text import main


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()
