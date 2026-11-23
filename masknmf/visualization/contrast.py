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
PERCENTILES_FIT = (0.1, 99.9)
PERCENTILES_OVER_FRAMES = (10, 90)
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


def fit_limits(frames: np.ndarray) -> tuple[float, float]:
    """
    Color limits of a movie from frames spread across it, by one rule for every movie and unit.

    Each frame gives the ``PERCENTILES_FIT`` of its nonzero pixels; vmin is the low ones' and vmax the high ones'
    ``PERCENTILES_OVER_FRAMES``. Zero pixels are left out because pixels outside a masked field of view and outside
    every footprint are exactly 0, and can be most of a frame. Ranking the frames keeps the bright ones in range
    and drops the one or two most extreme, such as a frame in a motion transient.

    Parameters
    ----------
    frames : np.ndarray
        Frames from ``sample_frames``. Shape (sampled, height, width[, channels]).

    Returns
    -------
    tuple[float, float]
        vmin, vmax; vmax is vmin + 1 when the two are equal. 0, 1 when no pixel is finite and nonzero.
    """
    per_frame = [
        np.percentile(frame[np.isfinite(frame) & (frame != 0)], PERCENTILES_FIT)
        for frame in frames
        if (np.isfinite(frame) & (frame != 0)).any()
    ]
    if len(per_frame) == 0:
        return 0.0, 1.0
    lows, highs = np.array(per_frame).T  ## (frames with values,) each
    vmin = float(np.percentile(lows, PERCENTILES_OVER_FRAMES[0]))
    vmax = float(np.percentile(highs, PERCENTILES_OVER_FRAMES[1]))
    return vmin, (vmax if vmax > vmin else vmin + 1.0)


def contrast_fixed(graphic) -> None:
    """Sets the limits to ``VMIN_FIXED``, ``VMAX_FIXED`` without reading the movie."""
    graphic.compute_histogram = False
    graphic.graphic.vmin, graphic.graphic.vmax = VMIN_FIXED, VMAX_FIXED


def contrast_sampled(graphic) -> None:
    """Sets the limits to ``fit_limits`` of the frames of ``sample_frames``, in the movie's own units."""
    graphic.compute_histogram = False
    graphic.graphic.vmin, graphic.graphic.vmax = fit_limits(frames=sample_frames(data=graphic.data))


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
    "sampled": contrast_sampled,
    "fixed": contrast_fixed,
    "fastplotlib": contrast_fastplotlib,
    "first-frame": contrast_first_frame,
}

# the background is about +/-0.02 where the signals reach 3, so the fixed limits show it flat
CONTRAST_BY_MOVIE = {"background": contrast_sampled}
