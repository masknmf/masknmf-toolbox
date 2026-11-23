"""Each contrast method fits a panel's color limits on fastplotlib's NDImage as its docstring says."""

import fastplotlib as fpl
import fastplotlib.graphics.image
import numpy as np
import pytest
from fastplotlib.widgets.nd_widget._async import run_sync
from rendercanvas.offscreen import RenderCanvas

from masknmf.visualization.contrast import CONTRAST_BY_MOVIE, CONTRAST_METHODS, VMAX_FIXED, VMIN_FIXED, fit_pooled_percentiles, histogram_sampled, sample_frames

NUM_FRAMES = 20


def make_movie(offset: float = 0.0) -> np.ndarray:
    """A movie whose frame t spans [t + offset, t + offset + 1)."""
    frames = np.arange(NUM_FRAMES, dtype=np.float32)[:, None, None] + offset
    return frames + np.random.default_rng(0).random((NUM_FRAMES, 32, 32), dtype=np.float32)


def make_panel(movie: np.ndarray | None = None):
    """An offscreen panel made as the viewer makes them: no histogram, limits fixed through graphic_kwargs."""
    movie = make_movie() if movie is None else movie
    widget = fpl.NDWidget({"time": (0, NUM_FRAMES, 1)}, names=["panel"], canvas=RenderCanvas(size=(400, 400), pixel_ratio=1))
    graphic = widget["panel"].add_nd_image(
        movie, ["time", "m", "n"], ["m", "n"], slider_maps={"time": np.arange(NUM_FRAMES)}, compute_histogram=False,
        graphic_kwargs={"vmin": VMIN_FIXED, "vmax": VMAX_FIXED},
    )
    return widget, graphic, movie


def show_frame(widget, graphic, index: int) -> None:
    """Slice the panel at ``index`` now instead of on the next draw."""
    widget.indices = {"time": index}
    run_sync(graphic._set_indices_())


def refuse_to_compute(data):
    raise AssertionError("fastplotlib computed color limits")


def test_fixed_opens_and_reloads_at_the_fixed_limits_without_fastplotlib_computing_any(monkeypatch):
    monkeypatch.setattr(fastplotlib.graphics.image, "quick_min_max", refuse_to_compute)
    widget, graphic, _ = make_panel()
    show_frame(widget=widget, graphic=graphic, index=10)
    CONTRAST_METHODS["fixed"](graphic)
    assert (graphic.graphic.vmin, graphic.graphic.vmax) == pytest.approx((VMIN_FIXED, VMAX_FIXED))
    graphic.data = make_movie(offset=100.0)
    CONTRAST_METHODS["fixed"](graphic)
    assert (graphic.graphic.vmin, graphic.graphic.vmax) == pytest.approx((VMIN_FIXED, VMAX_FIXED))
    assert graphic.slicer.histogram is None
    assert graphic.histogram_widget is None


def test_fastplotlib_docks_a_colorbar_over_the_whole_movie_and_fits_the_frame_on_show():
    widget, graphic, _ = make_panel()
    show_frame(widget=widget, graphic=graphic, index=10)
    CONTRAST_METHODS["fastplotlib"](graphic)
    _, edges = graphic.slicer.histogram
    assert graphic.histogram_widget is not None
    assert edges[0] < 1 and edges[-1] > NUM_FRAMES - 1
    assert 10 <= graphic.graphic.vmin < graphic.graphic.vmax < 11


def test_first_frame_fits_the_first_frame_whatever_frame_is_on_show():
    widget, graphic, movie = make_panel()
    CONTRAST_METHODS["fastplotlib"](graphic)
    show_frame(widget=widget, graphic=graphic, index=10)
    CONTRAST_METHODS["first-frame"](graphic)
    assert (graphic.graphic.vmin, graphic.graphic.vmax) == pytest.approx((movie[0].min(), movie[0].max()))
    assert graphic.histogram_widget is None


def test_the_background_fits_frames_across_the_movie_and_ignores_one_atypical_frame_on_show(monkeypatch):
    monkeypatch.setattr(fastplotlib.graphics.image, "quick_min_max", refuse_to_compute)
    movie = make_movie()
    # frame 4 is not among the 16 sampled of 20
    movie[4] += 1000
    widget, graphic, _ = make_panel(movie=movie)
    show_frame(widget=widget, graphic=graphic, index=4)
    CONTRAST_BY_MOVIE["background"](graphic)
    assert 0 < graphic.graphic.vmin < 1 and NUM_FRAMES - 1 < graphic.graphic.vmax < NUM_FRAMES
    assert graphic.histogram_widget is None


def test_sampled_frames_leave_out_the_all_zero_frames_that_pad_a_cut_recording():
    movie = make_movie()
    movie[:5] = 0
    movie[-1] = 0
    frames = sample_frames(data=movie)
    assert 0 < len(frames) < 16
    assert frames.reshape(len(frames), -1).any(axis=1).all()
    assert len(sample_frames(data=np.zeros_like(movie))) == 1


def test_the_background_fit_ignores_zero_frames():
    movie = make_movie()
    movie[:12] = 0
    widget, graphic, _ = make_panel(movie=movie)
    show_frame(widget=widget, graphic=graphic, index=0)
    CONTRAST_BY_MOVIE["background"](graphic)
    assert 12 < graphic.graphic.vmin < 14 and NUM_FRAMES - 1 < graphic.graphic.vmax < NUM_FRAMES


def test_the_sampled_histogram_spans_the_movie_and_the_limits_whatever_frame_is_on_show():
    movie = make_movie()
    movie[0] = 0
    frames = sample_frames(data=movie)
    counts, edges = histogram_sampled(frames=frames, vmin=VMIN_FIXED, vmax=VMAX_FIXED)
    assert edges[0] == pytest.approx(VMIN_FIXED)
    assert NUM_FRAMES - 1 < edges[-1] <= movie.max()
    assert counts.sum() > 0.999 * frames.size


def test_sampled_fits_frames_across_the_movie_in_its_own_units():
    widget, graphic, _ = make_panel(movie=make_movie(offset=5000.0))
    show_frame(widget=widget, graphic=graphic, index=0)
    CONTRAST_METHODS["sampled"](graphic)
    assert 5000 < graphic.graphic.vmin < 5001 and 5000 + NUM_FRAMES - 1 < graphic.graphic.vmax < 5000 + NUM_FRAMES
    assert graphic.histogram_widget is None
    assert next(iter(CONTRAST_METHODS)) == "sampled"


def test_the_pooled_fit_leaves_out_the_zero_pixels_of_a_masked_field_of_view():
    movie = make_movie(offset=5.0)
    movie[:, :, 4:] = 0
    vmin, vmax = fit_pooled_percentiles(frames=sample_frames(data=movie))
    assert 5 < vmin < 6 and NUM_FRAMES + 4 < vmax < NUM_FRAMES + 5
    assert fit_pooled_percentiles(frames=np.zeros((2, 4, 4), dtype=np.float32)) == (0.0, 1.0)
