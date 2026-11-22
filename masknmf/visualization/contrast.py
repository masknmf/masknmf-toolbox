"""
The ways a viewer panel fits its color limits, by the name ``--contrast`` takes.

Each method takes the panel's fastplotlib ``NDImage`` and runs when the panel opens, when another results file loads
into it and when it switches movie. A new method is a function here and an entry in ``CONTRAST_METHODS``.
"""

import numpy as np


def contrast_fastplotlib(graphic) -> None:
    """Fits the limits to the frame on show and docks a colorbar over a histogram of frames sampled across the movie."""
    graphic.compute_histogram = True


def contrast_first_frame(graphic) -> None:
    """Fits the limits to the movie's first frame; there is no histogram, so no colorbar."""
    graphic.compute_histogram = False
    frame = graphic.data[0]
    graphic.graphic.vmin, graphic.graphic.vmax = float(np.nanmin(frame)), float(np.nanmax(frame))


def contrast_none(graphic) -> None:
    """Leaves the limits fastplotlib gave the panel's image when it was made; there is no histogram, so no colorbar."""
    graphic.compute_histogram = False


CONTRAST_METHODS = {
    "fastplotlib": contrast_fastplotlib,
    "first-frame": contrast_first_frame,
    "none": contrast_none,
}
