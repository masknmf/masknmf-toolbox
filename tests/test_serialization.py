"""save_dict stores an array the file already holds byte for byte as an hdf5 hard link to it."""

import os

import h5py
import numpy as np
import torch

from masknmf.io import drop_group
from masknmf.utils._serialization import load_dict, save_dict


def big_array(seed: int) -> np.ndarray:
    """2 MB of float32, above the size save_dict links."""
    return np.random.default_rng(seed).random((512, 1024), dtype=np.float32)


def test_an_array_two_groups_hold_is_stored_once_and_reads_back_from_both(tmp_path):
    path = tmp_path / "results.hdf5"
    shared = big_array(seed=0)
    shared[0, 0] = np.nan
    save_dict({"tensor": torch.as_tensor(shared), "array": shared}, path, "A", exists_ok=True)
    save_dict({"tensor": torch.as_tensor(shared), "array": shared, "other": big_array(seed=1)}, path, "B", exists_ok=True)
    with h5py.File(path, "r") as file:
        assert file["A/tensor"] == file["B/tensor"]
        assert file["A/array"] == file["B/array"]
        assert file["B/other"] != file["A/array"]
    assert os.path.getsize(path) < 4.5 * shared.nbytes
    for group in ("A", "B"):
        loaded = load_dict(path, group)
        assert isinstance(loaded["tensor"], torch.Tensor) and isinstance(loaded["array"], np.ndarray)
        assert loaded["tensor"].numpy().tobytes() == shared.tobytes() and loaded["array"].tobytes() == shared.tobytes()


def test_a_tensor_and_an_array_of_the_same_bytes_stay_apart_because_their_attributes_differ(tmp_path):
    path = tmp_path / "results.hdf5"
    shared = big_array(seed=0)
    save_dict({"x": torch.as_tensor(shared)}, path, "A", exists_ok=True)
    save_dict({"x": shared}, path, "B", exists_ok=True)
    with h5py.File(path, "r") as file:
        assert file["A/x"] != file["B/x"]
    assert isinstance(load_dict(path, "B")["x"], np.ndarray)


def test_a_stored_dataset_with_attributes_of_its_own_is_never_linked_to(tmp_path):
    path = tmp_path / "results.hdf5"
    shared = big_array(seed=0)
    with h5py.File(path, "w") as file:
        file["A/x"] = shared
        file["A/x"].attrs["units"] = "photons"
    save_dict({"x": shared}, path, "B", exists_ok=True)
    with h5py.File(path, "r") as file:
        assert file["A/x"] != file["B/x"]
        assert "units" not in file["B/x"].attrs


def test_sparse_tensors_of_one_sparsity_share_their_indices_and_keep_their_values(tmp_path):
    path = tmp_path / "results.hdf5"
    indices = torch.as_tensor(np.random.default_rng(0).integers(0, 1000, (2, 200_000)))
    first = torch.sparse_coo_tensor(indices, torch.ones(200_000), (1000, 1000)).coalesce()
    second = torch.sparse_coo_tensor(first.indices(), torch.arange(first.values().numel(), dtype=torch.float32), (1000, 1000))
    save_dict({"U": first, "projector": second}, path, "A", exists_ok=True)
    with h5py.File(path, "r") as file:
        assert file["A/U/indices"] == file["A/projector/indices"]
        assert file["A/U/values"] != file["A/projector/values"]
    loaded = load_dict(path, "A")
    assert torch.equal(loaded["projector"].to_dense(), second.to_dense())


def test_replacing_the_group_that_wrote_an_array_first_leaves_the_other_group_its_copy(tmp_path):
    path = tmp_path / "results.hdf5"
    shared = big_array(seed=0)
    save_dict({"x": shared}, path, "A", exists_ok=True)
    save_dict({"x": shared}, path, "B", exists_ok=True)
    save_dict({"x": big_array(seed=1)}, path, "A", exists_ok=True)
    assert load_dict(path, "B")["x"].tobytes() == shared.tobytes()
    assert load_dict(path, "A")["x"].tobytes() == big_array(seed=1).tobytes()


def test_dropping_the_group_that_wrote_an_array_first_keeps_one_copy_in_the_other(tmp_path):
    path = tmp_path / "results.hdf5"
    shared = big_array(seed=0)
    save_dict({"x": shared, "y": shared}, path, "A", exists_ok=True)
    save_dict({"x": shared, "y": shared}, path, "B", exists_ok=True)
    drop_group(path, "A")
    with h5py.File(path, "r") as file:
        assert set(file) == {"B"}
        assert file["B/x"] == file["B/y"]
        assert file["B/x"][()].tobytes() == shared.tobytes()
    assert os.path.getsize(path) < 1.5 * shared.nbytes
