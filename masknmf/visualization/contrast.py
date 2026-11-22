"""
The ways a viewer panel fits its color limits, by the name ``--contrast`` takes.

Every panel is made at ``VMIN_FIXED``, ``VMAX_FIXED`` (its fastplotlib ``NDImage``'s ``graphic_kwargs``), so fastplotlib
computes no limits of its own. Each method takes the panel's ``NDImage`` and runs when the panel opens, when another
results file loads into it and when it switches movie. A new method is a function here and an entry in
``CONTRAST_METHODS``.
"""

import numpy as np

VMIN_FIXED = -0.05
VMAX_FIXED = 1.0


def contrast_fixed(graphic) -> None:
    """Sets the limits to ``VMIN_FIXED``, ``VMAX_FIXED`` without reading the movie; there is no histogram, so no colorbar."""
    graphic.compute_histogram = False
    graphic.graphic.vmin, graphic.graphic.vmax = VMIN_FIXED, VMAX_FIXED


def contrast_fastplotlib(graphic) -> None:
    """Fits the limits to the frame on show and docks a colorbar over a histogram of frames sampled across the movie."""
    graphic.compute_histogram = True
    # NOTE: NDImage leaves limits given in graphic_kwargs alone, so the fit to the frame is asked for here
    graphic.graphic.reset_vmin_vmax()


def contrast_first_frame(graphic) -> None:
    """Fits the limits to the movie's first frame; there is no histogram, so no colorbar."""
    graphic.compute_histogram = False
    frame = graphic.data[0]
    graphic.graphic.vmin, graphic.graphic.vmax = float(np.nanmin(frame)), float(np.nanmax(frame))


CONTRAST_METHODS = {
    "fixed": contrast_fixed,
    "fastplotlib": contrast_fastplotlib,
    "first-frame": contrast_first_frame,
}
