"""The opening color limits and the colorbar histogram, on synthetic movies."""

import numpy as np
import pytest

from masknmf.visualization.contrast import fit_limits, histogram_sampled, sample_frames

NUM_FRAMES = 20


def make_movie(offset: float = 0.0) -> np.ndarray:
    """A movie whose frame t spans [t + offset, t + offset + 1)."""
    frames = np.arange(NUM_FRAMES, dtype=np.float32)[:, None, None] + offset
    return frames + np.random.default_rng(0).random((NUM_FRAMES, 32, 32), dtype=np.float32)


def test_sampled_frames_leave_out_the_all_zero_frames_that_pad_a_cut_recording():
    movie = make_movie()
    movie[:5] = 0
    movie[-1] = 0
    frames = sample_frames(data=movie)
    assert 0 < len(frames) < 16
    assert frames.reshape(len(frames), -1).any(axis=1).all()
    assert len(sample_frames(data=np.zeros_like(movie))) == 1


def test_the_sampled_histogram_spans_the_movie_and_the_limits():
    movie = make_movie()
    movie[0] = 0
    frames = sample_frames(data=movie)
    counts, edges = histogram_sampled(frames=frames, vmin=-0.05, vmax=1.0)
    assert edges[0] == pytest.approx(-0.05)
    assert NUM_FRAMES - 1 < edges[-1] <= movie.max()
    assert counts.sum() > 0.999 * frames.size


def make_cells(scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """16 frames of unit noise with a 6 x 6 cell at 20 in every frame and a second one at 40 in frames 8 to 11."""
    frames = np.random.default_rng(1).normal(size=(16, 64, 64)).astype(np.float32)
    frames[:, 10:16, 10:16] += 20
    frames[8:12, 40:46, 40:46] += 40
    return frames * scale + offset


def test_the_fit_keeps_cells_in_range_that_are_bright_in_a_quarter_of_the_frames():
    vmin, vmax = fit_limits(frames=make_cells())
    assert -5 < vmin < -2 and 38 < vmax < 45


@pytest.mark.parametrize("scale, offset", [(1e-3, 0.0), (1.0, 0.0), (300.0, 5000.0), (-1.0, 0.0)])
def test_the_fit_follows_the_units_of_the_movie(scale, offset):
    vmin, vmax = fit_limits(frames=make_cells())
    limits = sorted((vmin * scale + offset, vmax * scale + offset))
    assert fit_limits(frames=make_cells(scale=scale, offset=offset)) == pytest.approx(limits, rel=0.05, abs=abs(scale) * 0.5)


def test_one_atypical_frame_among_the_sampled_does_not_move_the_fit():
    frames = make_cells()
    frames[3] += 1000
    assert fit_limits(frames=frames) == pytest.approx(fit_limits(frames=make_cells()), abs=1.5)


def test_the_fit_leaves_out_the_zero_pixels_of_a_masked_field_of_view():
    frames = make_cells()
    masked = frames.copy()
    masked[:, :, 20:] = 0
    masked[:, 20:, :] = 0
    vmin, vmax = fit_limits(frames=masked)
    assert vmin < -2 and 20 < vmax


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.float64])
def test_the_fit_reads_integer_and_float_movies(dtype):
    frames = (make_cells(scale=2.0, offset=100.0)).astype(dtype)
    vmin, vmax = fit_limits(frames=frames)
    assert 85 < vmin < 100 and 170 < vmax < 195


def test_the_fit_reads_rgb_frames():
    frames = np.repeat(make_cells()[..., None], 3, axis=-1) * np.array([1.0, 0.5, 0.0], dtype=np.float32)
    vmin, vmax = fit_limits(frames=frames)
    assert vmin < 0 and 30 < vmax < 45


def test_the_fit_returns_a_range_for_movies_without_one():
    assert fit_limits(frames=np.zeros((4, 8, 8), dtype=np.float32)) == (0.0, 1.0)
    assert fit_limits(frames=np.full((4, 8, 8), np.nan, dtype=np.float32)) == (0.0, 1.0)
    assert fit_limits(frames=np.full((4, 8, 8), 7.0, dtype=np.float32)) == (7.0, 8.0)
    frames = make_cells()
    frames[:, :5] = np.nan
    frames[0, 30, 30] = np.inf
    vmin, vmax = fit_limits(frames=frames)
    assert np.isfinite([vmin, vmax]).all() and vmin < vmax
    assert fit_limits(frames=make_cells()[:1])[1] > fit_limits(frames=make_cells()[:1])[0]
