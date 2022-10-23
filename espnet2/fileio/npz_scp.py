import collections.abc
from pathlib import Path
from typing import Union

import scipy
import scipy.sparse
import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text


class NpzScpWriter:
    """Writer class for a scp file of numpy file.

    Examples:
        key1 /some/path/a.npz
        key2 /some/path/b.npz
        key3 /some/path/c.npz
        key4 /some/path/d.npz
        ...

        >>> writer = NpzScpWriter('./data/', './data/feat.scp')
        >>> writer['aa'] = numpy_array
        >>> writer['bb'] = numpy_array

    """

    def __init__(self, outdir: Union[Path, str], scpfile: Union[Path, str]):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")

        self.data = {}

    def get_path(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        assert isinstance(value, np.ndarray), type(value)
        p = self.dir / f"{key}.npz"
        p.parent.mkdir(parents=True, exist_ok=True)
        sparse_matrix = scipy.sparse.csc_matrix(value)
        scipy.sparse.save_npz(str(p), sparse_matrix)
        self.fscp.write(f"{key} {p}\n")

        # Store the file path
        self.data[key] = str(p)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


class NpzScpReader(collections.abc.Mapping):
    """Reader class for a scp file of numpy file.

    Examples:
        key1 /some/path/a.npz
        key2 /some/path/b.npz
        key3 /some/path/c.npz
        key4 /some/path/d.npz
        ...

        >>> reader = NpzScpReader('npz.scp')
        >>> array = reader['key1']

    """

    def __init__(self, fname: Union[Path, str]):
        assert check_argument_types()
        self.fname = Path(fname)
        self.data = read_2column_text(fname)

    def get_path(self, key):
        return self.data[key]

    def __getitem__(self, key) -> np.ndarray:
        p = self.data[key]
        sparse_matrix = scipy.sparse.load_npz(p)
        ndarray = sparse_matrix.todense().T
        return ndarray

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()
