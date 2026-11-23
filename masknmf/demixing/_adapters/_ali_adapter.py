"""
ALI results as a masknmf results object: the ``ali.mat`` that ``software/ALI/run_ali.m`` (masknmf-paper-analyses)
writes when it runs the published ALI demo (github.com/Tsai-WenChen/Activity-Localization-Imaging) tile by tile over
one movie, read through the movie ALI read.

ALI's model of the movie, which the arrays here reproduce:
    df = movie - medfilt1(movie, win, 'truncate')      each pixel along time, win = 10 ms of frames    (ali_hp_filter)
    df = footprints @ traces_ls + residual                                                            (ali_t_decompose)
ALI does no motion correction, so the registered movie is the movie it read. run_ali.m detects spikes on -df, so the
indicator is negative going: a footprint is the mean df at its cluster's spikes, negative, and traces_ls is positive
at spikes. The movie is shown inverted about its mean (OphysArray), so the footprints are negated here; then
background (the running median of the inverted movie) + signals (-footprints @ traces_ls) + residual is the inverted
movie exactly. Each tile fits its own clusters, plus those of its 16 px margin, by least squares; the residual here
puts every cluster's traces together over the whole field.

Conventions on the masknmf side: frames come first, a pixel (row, col) is index row * width + col (C order, as in
SignalsArray), and every index is 0-based. ali.mat's 1-based rows, columns, clusters and frames are converted on load.
"""
import os
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import torch

from masknmf.arrays import OphysArray
from masknmf.arrays.array_interfaces import ArrayLike
from masknmf.demixing._adapters._miniali_adapter import MiniALIResidualArray, _matlab_array, roi_average_operator
from masknmf.demixing._base_results import BaseResults
from masknmf.demixing.demixing_arrays import SignalsArray
from masknmf.demixing.demixing_utils import scipy_sparse_to_torch
from masknmf.utils import SparseCOOTensor, torch_select_device

RESULT_FILE_NAME = "ali.mat"
DATASETS = ("traces", "traces_ls", "centers", "footprint_row", "footprint_col", "footprint_value", "footprint_cluster",
            "spike_frame", "spike_cluster", "params")


def read_ali_params(file: h5py.File) -> dict:
    """run_ali.m's params struct, one float per field (fs, win, num_frames, th_spike, tile_core, ...)."""
    return {name: float(np.asarray(dataset[()]).item()) for name, dataset in file["params"].items()}


def read_ali_file(path: str | os.PathLike, width: int) -> dict:
    """
    Reads ali.mat, converted to 0-based indices.

    Args:
        path (str | os.PathLike): ali.mat
        width (int): width of the movie ALI read, which ali.mat does not record

    Returns:
        dict: params; traces and traces_ls (frames, clusters); pixels, clusters and values of the footprints' nonzero
        entries, pixel row * width + col; centers (clusters, 2) as (row, col); spike_frame and spike_cluster.
    """
    with h5py.File(path, "r") as f:
        missing = [name for name in DATASETS if name not in f]
        if missing:
            raise ValueError(f"{path} holds no {', '.join(missing)}; is it the ali.mat of run_ali.m")
        params = read_ali_params(f)
        fields = {name: _matlab_array(f[name]) for name in DATASETS if name != "params"}
    num_frames = int(params["num_frames"])
    num_clusters = fields["centers"].reshape(-1, 2).shape[0]
    rows, cols = (fields[name].ravel().astype(np.int64) - 1 for name in ("footprint_row", "footprint_col"))
    return {
        "params": params,
        "traces": fields["traces"].astype(np.float32).reshape(num_frames, num_clusters),
        "traces_ls": fields["traces_ls"].astype(np.float32).reshape(num_frames, num_clusters),
        "pixels": rows * width + cols,
        "clusters": fields["footprint_cluster"].ravel().astype(np.int64) - 1,
        "values": fields["footprint_value"].ravel().astype(np.float32),
        "rows": rows,
        "cols": cols,
        "centers": fields["centers"].reshape(-1, 2).astype(np.float32) - 1,
        "spike_frame": fields["spike_frame"].ravel().astype(np.int64) - 1,
        "spike_cluster": fields["spike_cluster"].ravel().astype(np.int64) - 1,
    }


def running_median(block: torch.Tensor, frames: np.ndarray, start: int, num_frames: int, window: int) -> torch.Tensor:
    """
    MATLAB's medfilt1(x, window, 'truncate') along time at the given frames: the median of frames
    t - window // 2 to t + (window - 1) // 2, cut at the movie's ends, the middle two averaged when their count is even.

    Args:
        block (torch.Tensor): frames start to start + len(block) of the movie, (frames, height, width), covering every
            window
        frames (np.ndarray): the frames to filter, indices into the whole movie
        start (int): the movie frame block starts at
        num_frames (int): frames in the whole movie, where the windows are cut
        window (int): medfilt1's n

    Returns:
        torch.Tensor: (len(frames), height, width)
    """
    medians = []
    for t in frames:
        first, stop = max(t - window // 2, 0), min(t + (window - 1) // 2 + 1, num_frames)
        values = torch.sort(block[first - start:stop - start], dim=0).values
        count = stop - first
        medians.append((values[(count - 1) // 2] + values[count // 2]) / 2)
    return torch.stack(medians)


class RunningMedianArray(ArrayLike):
    """Each pixel's running median along time (ali_hp_filter's baseline), computed from a movie on access."""

    def __init__(self, movie: ArrayLike, window: int):
        """
        Args:
            movie (ArrayLike): (frames, height, width), returning tensors
            window (int): frames in the median window, medfilt1's n
        """
        if window < 1:
            raise ValueError(f"window must be at least 1 frame, got {window}")
        self._movie = movie
        self._window = int(window)

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self._movie.shape)

    @property
    def window(self) -> int:
        return self._window

    def __getitem__(self, item) -> torch.Tensor:
        item = item if isinstance(item, tuple) else (item,)
        num_frames = self.shape[0]
        frames = np.atleast_1d(np.arange(num_frames)[item[0]])
        order = np.argsort(frames, kind="stable")
        ## NOTE: frames further apart than a window are read separately, so frames [0, 8000] read 2 windows, not 8000 frames
        runs = np.split(order, np.flatnonzero(np.diff(frames[order]) > self._window) + 1)
        medians = []
        for run in runs:
            start = max(int(frames[run[0]]) - self._window // 2, 0)
            stop = min(int(frames[run[-1]]) + (self._window - 1) // 2 + 1, num_frames)
            block = torch.as_tensor(self._movie[(slice(start, stop), *item[1:])], dtype=self.dtype)
            medians.append(running_median(block, frames[run], start, num_frames, self._window))
        return torch.cat(medians)[np.argsort(order)]


class ALIResults(BaseResults):

    pipeline = "ali"
    source = "ALI's ali.mat from run_ali.m"
    raw_source = "the movie ALI read: its recording folder or movie file"
    patterns = ("ali.mat",)

    @classmethod
    def expand(cls, path: Path) -> list[Path]:
        """Accepts ali.mat or the folder holding it."""
        path = Path(path).expanduser()
        if path.is_dir():
            path = path / RESULT_FILE_NAME
        if not path.is_file() or path.suffix.lower() != ".mat" or not h5py.is_hdf5(str(path)):
            return []
        with h5py.File(path, "r") as f:
            return [path] if all(name in f for name in DATASETS) else []

    @classmethod
    def matches(cls, path: Path) -> bool:
        return len(cls.expand(path)) > 0

    @classmethod
    def describe(cls, path: Path) -> str:
        with h5py.File(cls.expand(path)[0], "r") as f:
            num_frames = int(read_ali_params(f)["num_frames"])
            num_clusters = _matlab_array(f["centers"]).reshape(-1, 2).shape[0]
            num_spikes = f["spike_frame"].size
        return f"ali: {num_clusters} clusters, the first {num_frames} frames, {num_spikes} spikes"

    @classmethod
    def from_path(cls, path: Path, raw: ArrayLike | str | Path | None = None, device: str = "cpu") -> tuple["ALIResults", ArrayLike | None, list[str]]:
        """raw is the movie ALI read: its recording folder (the .bin and output_data.mat), a movie file, or an array."""
        from masknmf.io import load_movie

        path = cls.expand(path)[0]
        if raw is None:
            raise ValueError(f"{path} holds ALI results, which need the movie ALI read: its recording folder or movie file")
        if isinstance(raw, (str, Path)):
            raw = load_movie(raw)
        results = cls(path, raw_video=raw, device=device)
        return results, results.raw_array, []

    def __init__(self, path: str | os.PathLike, raw_video: ArrayLike | np.ndarray, device: str | torch.device = "auto"):
        """
        Args:
            path (str | os.PathLike): ali.mat, or the folder holding it
            raw_video (ArrayLike | np.ndarray): the movie ALI read, (frames, height, width), with at least the frames
                ALI processed, which are the first ones
            device (str | torch.device): where the tensors live
        """
        self._device = torch_select_device(device)
        expanded = self.expand(Path(path))
        if not expanded:
            raise ValueError(f"{path} is not run_ali.m's ali.mat or a folder holding it")
        self._path = expanded[0]

        num_frames_raw, height, width = (int(d) for d in raw_video.shape)
        fields = read_ali_file(self._path, width=width)
        self._params = fields["params"]
        num_frames = int(self._params["num_frames"])
        if num_frames_raw < num_frames:
            raise ValueError(f"the movie has {num_frames_raw} frames, ALI processed {num_frames}")
        if fields["rows"].size and (fields["rows"].max() >= height or fields["cols"].max() >= width):
            raise ValueError(f"footprints reach pixel ({fields['rows'].max()}, {fields['cols'].max()}), outside the "
                             f"{height} x {width} movie; is it the movie ALI read")
        self._num_frames_raw = num_frames_raw

        self._raw_array = OphysArray(raw_video, negative_indicator=True, include_mean=False, device=self._device,
                                     good_frames=np.arange(num_frames_raw) < num_frames)
        num_clusters = fields["traces"].shape[1]
        self._spatial_scipy = sp.csc_matrix((-fields["values"], (fields["pixels"], fields["clusters"])),
                                            shape=(height * width, num_clusters), dtype=np.float32)
        self._spatial_demixed = scipy_sparse_to_torch(self._spatial_scipy).coalesce().float()
        self._temporal_demixed = torch.from_numpy(fields["traces"])
        self._temporal_demixed_raw = torch.from_numpy(fields["traces_ls"])
        self._centers = fields["centers"]
        self._spike_frames = [fields["spike_frame"][fields["spike_cluster"] == k] for k in range(num_clusters)]

        self._signals_array = SignalsArray.from_tensors((height, width), self._spatial_demixed, self._temporal_demixed_raw)
        self._background_array = RunningMedianArray(self._raw_array, window=int(self._params["win"]))
        self._raw_residual_array = MiniALIResidualArray(self._raw_array, self._background_array, self._signals_array)

        # averages over each footprint's support, (clusters, frames), as the viewer's trace panel draws them
        self._roi_average = roi_average_operator(self._spatial_scipy)
        self._signal_roi_averages = torch.from_numpy(
            np.asarray((self._roi_average @ self._spatial_scipy) @ fields["traces_ls"].T, np.float32))
        self._registered_roi_averages = None
        self._background_roi_averages = None
        self.to(self._device)

    def to(self, new_device: str | torch.device):
        device = torch.device(new_device)
        self._spatial_demixed = self._spatial_demixed.to(device)
        self._temporal_demixed = self._temporal_demixed.to(device)
        self._temporal_demixed_raw = self._temporal_demixed_raw.to(device)
        self._signal_roi_averages = self._signal_roi_averages.to(device)
        self._signals_array.to(device)
        for name in ("_registered_roi_averages", "_background_roi_averages"):
            if getattr(self, name) is not None:
                setattr(self, name, getattr(self, name).to(device))
        self._device = device

    def compute_roi_averages(self, frame_batch_size: int = 256) -> None:
        """
        Averages the registered movie and the background over each footprint's support, one pass over the movie, so
        that fluctuating_background_roi_averages and residual_roi_averages are no longer None.

        Args:
            frame_batch_size (int): frames read at a time
        """
        operator = scipy_sparse_to_torch(sp.coo_matrix(self._roi_average)).to(self._device).float()
        registered, background = [], []
        for start in range(0, self.shape[0], frame_batch_size):
            frames = slice(start, min(start + frame_batch_size, self.shape[0]))
            for averages, movie in ((registered, self._raw_array), (background, self._background_array)):
                block = torch.as_tensor(movie[frames], dtype=torch.float32, device=self._device)
                averages.append(torch.sparse.mm(operator, block.reshape(block.shape[0], -1).T))
        self._registered_roi_averages = torch.cat(registered, dim=1)
        self._background_roi_averages = torch.cat(background, dim=1)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def path(self) -> Path:
        return self._path

    @property
    def params(self) -> dict:
        """run_ali.m's parameters: fs, win, num_frames, th_spike, tile_core, tile_margin, nsvd, factor, th, r, ..."""
        return self._params

    @property
    def shape(self) -> tuple[int, int, int]:
        """(frames ALI processed, height, width)"""
        return self._raw_array.shape

    @property
    def fov_shape(self) -> tuple[int, int]:
        return tuple(self.shape[1:])

    @property
    def num_frames_raw(self) -> int:
        return self._num_frames_raw

    @property
    def num_sources(self) -> int:
        return self._spatial_demixed.shape[1]

    @property
    def frame_rate(self) -> float:
        return self._params["fs"]

    @property
    def centers(self) -> np.ndarray:
        """(clusters, 2), each cluster's (row, col) on the ALI map, sub-pixel, 0-based."""
        return self._centers

    @property
    def spike_frames(self) -> list[np.ndarray]:
        """Per cluster, the frames of the spikes ALI assigned to it, 0-based."""
        return self._spike_frames

    @property
    def spatial_demixed(self) -> SparseCOOTensor:
        """Footprints, (pixels, clusters), pixel row * width + col: minus the mean df at each cluster's spikes, so positive."""
        return self._spatial_demixed

    @property
    def temporal_demixed(self) -> torch.Tensor:
        """``traces``, (frames, clusters): ALI's published traces, its improved estimate minus its median, negatives set to 0."""
        return self._temporal_demixed

    @property
    def temporal_demixed_raw(self) -> torch.Tensor:
        """``traces_ls``, (frames, clusters): the least-squares fit of the footprints to df, positive at spikes."""
        return self._temporal_demixed_raw

    @property
    def raw_array(self) -> OphysArray:
        """The movie ALI read, its first num_frames frames, inverted about the mean and mean subtracted."""
        return self._raw_array

    @property
    def registered_array(self) -> OphysArray:
        """raw_array: ALI does no motion correction."""
        return self._raw_array

    @property
    def signals_array(self) -> SignalsArray:
        """spatial_demixed @ traces_ls, as a movie: the part of df the footprints fit."""
        return self._signals_array

    @property
    def fluctuating_background_array(self) -> RunningMedianArray:
        """Each pixel's running median over params['win'] frames, the baseline ali_hp_filter subtracts."""
        return self._background_array

    @property
    def raw_residual_array(self) -> MiniALIResidualArray:
        """registered - background - signals: df minus what the footprints fit, inverted."""
        return self._raw_residual_array

    # ROI averages, all (clusters, frames): each movie averaged uniformly over the support of spatial_demixed. ALI's
    # footprints are disks of radius 10 px and overlap, so an average includes neighbours.
    @property
    def signal_roi_averages(self) -> torch.Tensor:
        return self._signal_roi_averages

    @property
    def fluctuating_background_roi_averages(self) -> torch.Tensor | None:
        """None until compute_roi_averages."""
        return self._background_roi_averages

    @property
    def residual_roi_averages(self) -> torch.Tensor | None:
        """None until compute_roi_averages."""
        if self._registered_roi_averages is None:
            return None
        return self._registered_roi_averages - self._signal_roi_averages - self._background_roi_averages
