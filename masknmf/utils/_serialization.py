import os
from pathlib import Path
import traceback
from warnings import warn

import numpy as np
import h5py
import torch

# arrays smaller than this are written even when the file holds an identical one: a link would save next to nothing
NUM_BYTES_MIN_LINKED = 2**20
# an array is compared with a stored dataset this many bytes at a time, so the comparison holds little in memory
NUM_BYTES_COMPARED = 2**26


def save_dict(d, filename, group, exists_ok=False, raise_type_fail=True):
    """
    Recursively save a dict to a hdf5 group in a new file.

    An array the file already holds byte for byte, in any group, is written as an hdf5 hard link to it: a second name
    for the same stored dataset. The compression that CompressionArray and DemixingResults both hold is stored once
    that way, and every reader still sees two ordinary datasets.

    Args:
        d (dict): dict to save as an hdf5 file

        filename (str | Path): Full path to save the file to. Must not already exist unless ``exists_ok``.

        group (str): group name to save the dict to

        exists_ok (bool): Whether to write to an existing file; an existing ``group`` in it is replaced.
        Useful for writing multiple serialized objects to a file

        raise_type_fail (bool): If True: raise an exception if saving a part of the dict fails.
        If False: prints a warning instead and saves the
        object's __str__() return value.

    Returns:
        None

    Raises
        FileExistsError
            If the path specified by the `filename` parameter already exists.

        TypeError
            If a particular entry within the dict cannot be saved to hdf5 AND
            the argument `raise_type_fail` is set to `True`
    """
    if os.path.isfile(filename):
        if not exists_ok:
            raise FileExistsError
        else:
            writemode = 'r+'
    else:
        writemode = 'w'


    with h5py.File(filename, writemode) as h5file:
        if group in h5file:
            del h5file[group]
        paths_by_signature = {}
        _index_datasets(h5file, paths_by_signature)
        _dicts_to_group(h5file, "{}/".format(group), d,
                        raise_meta_fail=raise_type_fail, paths_by_signature=paths_by_signature)
    #
    # if not exists_ok:
    #     if os.path.isfile(filename):
    #         raise FileExistsError
    #
    # with h5py.File(filename, 'w') as h5file:
    #     _dicts_to_group(h5file, "{}/".format(group), d,
    #                     raise_meta_fail=raise_type_fail)


def _signature(shape, dtype, layout: str | None) -> tuple:
    """The shape, dtype and layout attribute two arrays must share before their bytes are compared."""
    return tuple(shape), np.dtype(dtype).str, layout


def _index_datasets(group: h5py.Group, paths_by_signature: dict) -> None:
    """
    Add the datasets below ``group`` that a new array could link to, by signature: those of at least
    ``NUM_BYTES_MIN_LINKED`` bytes that carry no attribute but ``layout``, since a link shares the attributes too.
    """
    for item in group.values():
        if isinstance(item, h5py.Group):
            _index_datasets(item, paths_by_signature)
        elif isinstance(item, h5py.Dataset) and item.nbytes >= NUM_BYTES_MIN_LINKED and set(item.attrs) <= {"layout"}:
            signature = _signature(item.shape, item.dtype, item.attrs.get("layout"))
            paths_by_signature.setdefault(signature, []).append(item.name)


def _holds_bytes(dataset: h5py.Dataset, array: np.ndarray) -> bool:
    """True when ``dataset`` holds exactly the bytes of ``array``, an array of its shape and dtype."""
    num_rows_block = max(1, NUM_BYTES_COMPARED * len(array) // array.nbytes)
    return all(
        dataset[start:start + num_rows_block].tobytes() == np.ascontiguousarray(array[start:start + num_rows_block]).tobytes()
        for start in range(0, len(array), num_rows_block)
    )


def _write_array(h5file, path: str, array: np.ndarray, layout: str | None, paths_by_signature: dict) -> None:
    """
    Write ``array`` at ``path``, or a hard link to a dataset in ``paths_by_signature`` holding the same bytes.

    Args:
        h5file (h5py.File): The open file.
        path (str): Where the array goes.
        array (np.ndarray): A numeric array.
        layout (str | None): The ``layout`` attribute to set, "strided" for a torch tensor; None sets none.
        paths_by_signature (dict): The datasets a link may point to, by :func:`_signature`; a new dataset is added.
    """
    signature = _signature(array.shape, array.dtype, layout)
    if array.nbytes >= NUM_BYTES_MIN_LINKED:
        for path_stored in paths_by_signature.get(signature, []):
            if _holds_bytes(dataset=h5file[path_stored], array=array):
                h5file[path] = h5file[path_stored]
                return
    h5file[path] = array
    if layout is not None:
        h5file[path].attrs["layout"] = layout
    if array.nbytes >= NUM_BYTES_MIN_LINKED:
        paths_by_signature.setdefault(signature, []).append(h5file[path].name)


def _dicts_to_group(h5file, path, d, raise_meta_fail, paths_by_signature):
    for key, item in d.items():
        if isinstance(item, torch.Tensor):
            if item.layout is torch.sparse_coo:
                # store this in a new group
                if not item.is_coalesced():
                    item = item.coalesce()
                sparse_dict = {
                    "indices": item.indices(),
                    "values": item.values(),
                    "size": np.asarray(item.size()) # size must be specified
                }

                # save to a group with indices and values arrays
                _dicts_to_group(
                    h5file,
                    "{}{}/".format(path, key),
                    sparse_dict,
                    True,
                    paths_by_signature
                )

                h5file[path + key].attrs["layout"] = "sparse_coo"

            elif item.layout is torch.strided:
                _write_array(h5file, path + key, item.cpu().numpy(), "strided", paths_by_signature)

            else:
                raise TypeError(
                    f"Serialization of Tensor layout: {item.layout} not supported, this should be fixed."
                )

        elif isinstance(item, np.ndarray):

            if item.dtype == np.dtype('O'):
                # see if h5py is ok with it
                try:
                    h5file[path + key] = item
                    # h5file[path + key].attrs['dtype'] = item.dtype.str
                except TypeError:
                    msg = "numpy dtype 'O' for item:\n{}\n" \
                          "not supported by HDF5\n{}" \
                          "".format(item, traceback.format_exc())

                    if raise_meta_fail:
                        raise TypeError(msg)
                    else:
                        h5file[path + key] = str(item)
                        warn("{}, storing whatever str(obj) returns"
                             "".format(msg))

            # numpy array of unicode strings
            elif item.dtype.str.startswith('<U'):
                h5file[path + key] = item.astype(h5py.special_dtype(vlen=str))

                # otherwise h5py doesn't restore the right dtype for str types
                h5file[path + key].attrs['dtype'] = item.dtype.str

            # other types
            else:
                _write_array(h5file, path + key, item, None, paths_by_signature)

        # single pieces of data
        elif isinstance(item, (str, int, np.int8,
                               np.int16, np.int32, np.int64, float,
                               np.float16, np.float32, np.float64)):
            h5file[path + key] = item

        elif isinstance(item, dict):
            _dicts_to_group(
                h5file, "{}{}/".format(path, key), item, raise_meta_fail, paths_by_signature
            )

        # last resort, try to convert this object
        # to a dict and save its attributes
        elif hasattr(item, '__dict__'):
            _dicts_to_group(
                h5file,
                "{}{}/".format(path, key),
                item.__dict__,
                raise_meta_fail,
                paths_by_signature
            )

        else:
            msg = "{} for item: {} not supported by HDF5" \
                  "".format(type(item), item)

            if raise_meta_fail:
                raise TypeError(msg)

            else:
                h5file[path + key] = str(item)
                warn("{}, storing whatever str(obj) returns"
                     "".format(msg))


def load_dict(filename, group):
    """
    Recursively load a dict from an hdf5 group in a file.

    Parameters
    ----------
    filename : str
        full path to the hdf5 file

    group : str
        Name of the group that contains the dict to load

    Returns
    -------
    d : dict
        dict loaded from the specified hdf5 group.
    """

    with h5py.File(filename, 'r') as h5file:
        return _dicts_from_group(h5file, "{}/".format(group))


def _dicts_from_group(h5file, path):
    ans = {}
    for key, item in h5file[path].items():
        if "layout" in item.attrs:
            # it's a torch tensor
            match item.attrs["layout"]:
                case "sparse_coo":
                    # reconstruct sparse_coo
                    indices = item["indices"][()]
                    values = item["values"][()]
                    size = item["size"][()]
                    # needs to a tuple[int] for `torch.sparse_coo_tensor` to be happy
                    size = tuple(size)

                    ans[key] = torch.sparse_coo_tensor(
                        indices=indices,
                        values=values,
                        size=size
                    )

                case "strided":
                    ans[key] = torch.from_numpy(item[()])

                case _:
                    raise TypeError

        elif isinstance(item, h5py.Dataset):
            if item.attrs.__contains__('dtype'):
                ans[key] = item[()].astype(item.attrs['dtype'])
            else:
                ans[key] = item[()]
        elif isinstance(item, h5py.Group):
            ans[key] = _dicts_from_group(h5file, path + key + '/')
    return ans


class Serializer:
    # should specify set of serialized attributes
    _serialized = {}

    def _to_dict(self) -> dict:
        d = {}

        for key in self._serialized:
            val = getattr(self, key)

            # skip None so we don't have to deal with serializing them.
            # HDF5 has no None type, and it can create headaches to properly
            # serialize and de-serialize them. To get around this, any attributes
            # that are None and required for de-serialization should be default None
            # kwargs in the constructor of the class they are serialized from.
            if val is None:
                continue

            if isinstance(val, (tuple, list)):
                val = np.asarray(val)

            d[key] = val

        return d

    def export(self, path: str | Path):
        """
        Export to an HDF5 file, as the group named after this class.
        Requires ``h5py`` http://docs.h5py.org/

        Args:
            path (str): Full file path. Created if missing; in an existing file the other groups are kept
                and this class's group is replaced, so one file can hold every stage of a pipeline.
        """

        d = self._to_dict()
        save_dict(d, filename=path, group=self.__class__.__name__, exists_ok=True)

    @classmethod
    def from_hdf5(cls, path, **kwargs):
        """Load result from an hdf5 file, from the group named after this class. Any additional kwargs are passed to the constructor"""
        d = load_dict(path, cls.__name__)
        return cls(**d, **kwargs)
