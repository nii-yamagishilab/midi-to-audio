import collections.abc
from pathlib import Path
from typing import Union

import numpy as np
import librosa
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text


class EmbedScpReader(collections.abc.Mapping):
    """Reader class for 'wav.scp'.

    Examples:
        key1 embeddings_if_key1
        key2 embeddings_if_key2
        ...

        >>> reader = EmbedScpReader('wav.scp')
        >>> rate, array = reader['key1']

    """

    def __init__(
        self,
        fname,
        dtype=np.int16,
        always_2d: bool = False,
        normalize: bool = False,
    ):
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.always_2d = always_2d
        self.normalize = normalize
        self.data = read_2column_text(fname)

    def __getitem__(self, key):
        feat_string = self.data[key]
        feat_array = np.array([float(item) for item in feat_string.split()[1:-1]])
        return feat_array

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class MeanEmbedScpReader(collections.abc.Mapping):
    """Reader class for 'wav.scp'.

    Examples:
        key1 embeddings_if_key1
        key2 embeddings_if_key2
        ...

        >>> reader = EmbedBAKScpReader('wav.scp')
        >>> rate, array = reader['key1']

    """

    def __init__(
        self,
        fname,
        dtype=np.int16,
        always_2d: bool = False,
        normalize: bool = False,
    ):
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.always_2d = always_2d
        self.normalize = normalize
        self.data = read_2column_text(fname)

    def __getitem__(self, key):
        # NOTE (Xuan): split with '-' on nsynth, while with '_' on urmp
        instr_key = key.split('_')[0]
        feat_string = self.data[instr_key]
        feat_array = np.array([float(item) for item in feat_string.split()[1:-1]])
        return feat_array

    def get_path(self, key):
        instr_key = key.split('_')[0]
        return self.data[instr_key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()



