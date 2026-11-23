"""
The ways a viewer panel fits its color limits, by the name ``--contrast`` takes.

Every panel is made at ``VMIN_FIXED``, ``VMAX_FIXED`` (its fastplotlib ``NDImage``'s ``graphic_kwargs``), so fastplotlib
computes no limits of its own. Each method takes the panel's ``NDImage`` and runs when the panel opens, when another
results file loads into it and when it switches movie. A new method is a function here and an entry in
``CONTRAST_METHODS``. A movie in ``CONTRAST_BY_MOVIE`` takes its own method, whatever ``--contrast`` names. A method
that leaves fastplotlib's histogram off gets a colorbar from the viewer, over ``histogram_on_show``.
"""

import numpy as np

VMIN_FIXED = -0.05
VMAX_FIXED = 1.0


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
    Sets the limits to the median, over 16 frames spread across the movie, of each frame's 0.1 and 99.9 percentiles.
    One atypical frame, such as one in the motion transient at the start of a recording, does not move them.
    """
    graphic.compute_histogram = False
    indices = np.linspace(0, graphic.data.shape[0] - 1, 16).astype(int)
    percentiles = np.stack([np.nanpercentile(np.asarray(graphic.data[t]), [0.1, 99.9]) for t in indices])  ## (16, 2)
    vmin, vmax = np.median(percentiles, axis=0)
    graphic.graphic.vmin, graphic.graphic.vmax = float(vmin), float(vmax)


def histogram_on_show(image) -> tuple[np.ndarray, np.ndarray]:
    """A 100-bin histogram of the frame an ``ImageGraphic`` shows, over a range that takes in its limits too."""
    values = image.data.value[np.isfinite(image.data.value)]
    if values.size == 0:
        return np.histogram(values, bins=100, range=(image.vmin, image.vmax))
    return np.histogram(values, bins=100, range=(min(values.min(), image.vmin), max(values.max(), image.vmax)))


CONTRAST_METHODS = {
    "fixed": contrast_fixed,
    "fastplotlib": contrast_fastplotlib,
    "first-frame": contrast_first_frame,
}

# the background is about +/-0.02 where the signals reach 3, so the fixed limits show it flat; 16 of its frames
# take 4 ms to make on a 16000-frame 2p recording
CONTRAST_BY_MOVIE = {"background": contrast_sampled_percentiles}
