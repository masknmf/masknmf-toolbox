"""
The ways a viewer panel fits its color limits, by the name ``--contrast`` takes.

Every panel is made at ``VMIN_FIXED``, ``VMAX_FIXED`` (its fastplotlib ``NDImage``'s ``graphic_kwargs``), so fastplotlib
computes no limits of its own. Each method takes the panel's ``NDImage`` and runs when the panel opens, when another
results file loads into it and when it switches movie. A new method is a function here and an entry in
``CONTRAST_METHODS``. A movie in ``CONTRAST_BY_MOVIE`` takes its own method, whatever ``--contrast`` names. A method
that leaves fastplotlib's histogram off gets a colorbar from the viewer, over ``histogram_sampled``.

Fits and histograms that read the movie read ``sample_frames``: frames spread across it, without the all-zero ones
that pad a recording whose first or last frames were cut.
"""

import numpy as np

VMIN_FIXED = -0.05
VMAX_FIXED = 1.0
NUM_FRAMES_SAMPLED = 16
# a colorbar spans these percentiles of the sampled frames: the few pixels beyond them would take most of the bar
# (KG263 F01 background: 99.98% of values within -0.04 to 0.03, extremes at -0.12 and 0.09)
PERCENTILES_REACH = (0.01, 99.99)


def sample_frames(data) -> np.ndarray:
    """
    ``NUM_FRAMES_SAMPLED`` frames spread across a movie, without the all-zero ones.

    Parameters
    ----------
    data : array-like
        Movie indexed by frame. Shape (frames, height, width[, channels]).

    Returns
    -------
    np.ndarray
        The sampled frames that hold a nonzero value; one zero frame when none does. Shape (sampled, height, width[, channels]).
    """
    indices = np.unique(np.linspace(0, data.shape[0] - 1, NUM_FRAMES_SAMPLED).astype(int))
    frames = np.stack([np.asarray(data[int(t)], dtype=np.float32) for t in indices])
    has_values = np.nan_to_num(frames).reshape(len(frames), -1).any(axis=1)
    return frames[has_values] if has_values.any() else frames[:1]


def fit_sampled_percentiles(frames: np.ndarray, percentiles: tuple[float, float]) -> tuple[float, float]:
    """
    The median over ``frames`` of each frame's two ``percentiles``, so one atypical frame does not move the limits.

    Parameters
    ----------
    frames : np.ndarray
        Frames from ``sample_frames``. Shape (sampled, height, width[, channels]).
    percentiles : tuple[float, float]
        Low and high percentile of a frame, 0 to 100.

    Returns
    -------
    tuple[float, float]
        vmin, vmax; vmax is vmin + 1 when the two are equal.
    """
    per_frame = np.nanpercentile(frames.reshape(len(frames), -1), percentiles, axis=1)  ## (2, sampled)
    vmin, vmax = (float(value) for value in np.median(per_frame, axis=1))
    return vmin, (vmax if vmax > vmin else vmin + 1.0)


def contrast_fixed(graphic) -> None:
    """Sets the limits to ``VMIN_FIXED``, ``VMAX_FIXED`` without reading the movie."""
    graphic.compute_histogram = False
    graphic.graphic.vmin, graphic.graphic.vmax = VMIN_FIXED, VMAX_FIXED


def contrast_fastplotlib(graphic) -> None:
    """Fits the limits to the frame on show and docks a colorbar over a histogram of frames sampled across the movie."""
    graphic.compute_histogram = True
    # NOTE: NDImage leaves limits given in graphic_kwargs alone, so the fit to the frame is asked for here
    graphic.graphic.reset_vmin_vmax()


def contrast_first_frame(graphic) -> None:
    """Fits the limits to the movie's first frame."""
    graphic.compute_histogram = False
    frame = graphic.data[0]
    graphic.graphic.vmin, graphic.graphic.vmax = float(np.nanmin(frame)), float(np.nanmax(frame))


def contrast_sampled_percentiles(graphic) -> None:
    """
    Sets the limits to the median, over the frames of ``sample_frames``, of each frame's 0.1 and 99.9 percentiles.
    One atypical frame, such as one in the motion transient at the start of a recording, does not move them.
    """
    graphic.compute_histogram = False
    graphic.graphic.vmin, graphic.graphic.vmax = fit_sampled_percentiles(
        frames=sample_frames(data=graphic.data), percentiles=(0.1, 99.9)
    )


def histogram_sampled(frames: np.ndarray, vmin: float, vmax: float) -> tuple[np.ndarray, np.ndarray]:
    """
    A 100-bin histogram of frames spread across a movie, between their ``PERCENTILES_REACH`` and the limits.

    Parameters
    ----------
    frames : np.ndarray
        Frames from ``sample_frames``.
    vmin, vmax : float
        The panel's color limits.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Counts and their 101 bin edges: the value axis of a colorbar, which its handles move over.
    """
    values = frames[np.isfinite(frames)]
    if values.size == 0:
        return np.histogram(values, bins=100, range=(vmin, vmax))
    low, high = np.percentile(values, PERCENTILES_REACH)
    return np.histogram(values, bins=100, range=(min(float(low), vmin), max(float(high), vmax)))


CONTRAST_METHODS = {
    "fixed": contrast_fixed,
    "fastplotlib": contrast_fastplotlib,
    "first-frame": contrast_first_frame,
}

# the background is about +/-0.02 where the signals reach 3, so the fixed limits show it flat; 16 of its frames
# take 4 ms to make on a 16000-frame 2p recording
CONTRAST_BY_MOVIE = {"background": contrast_sampled_percentiles}
