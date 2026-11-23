"""The time slider runs in times or frame numbers; panels and the trace plot's playhead show the frame nearest it."""

import fastplotlib as fpl
import numpy as np
import pytest
from fastplotlib.widgets.nd_widget._async import run_sync
from rendercanvas.offscreen import RenderCanvas

from masknmf.cli import timings
from masknmf.visualization.imgui import TracePlot, frame_slider_map, resolve_time_reference

NUM_FRAMES = 1000
FRAME_RATE = 30.0


def make_viewer(frame_timings: np.ndarray | None):
    """An offscreen panel over a movie whose frame k is filled with k, its index linked to a TracePlot."""
    ref_range, frame_timings = resolve_time_reference(NUM_FRAMES, frame_timings)
    movie = np.broadcast_to(np.arange(NUM_FRAMES, dtype=np.float32)[:, None, None], (NUM_FRAMES, 4, 4))
    widget = fpl.NDWidget(ref_range, names=["panel"], canvas=RenderCanvas(size=(200, 200)))
    graphic = widget["panel"].add_nd_image(
        movie, ["time", "m", "n"], ["m", "n"], slider_maps={"time": frame_slider_map(frame_timings)}, compute_histogram=False
    )
    plot = TracePlot(["traces"], num_frames=NUM_FRAMES, frame_timings=frame_timings)
    plot.link(widget.indices)
    return widget.indices, graphic, plot, frame_timings


def shown_frame(graphic) -> int:
    """The frame the panel shows: its movie's frame k is filled with k."""
    run_sync(graphic._set_indices_())
    return int(graphic.graphic.data.value[0, 0])


@pytest.mark.parametrize("frame_rate", [None, FRAME_RATE], ids=["frame numbers", "times"])
def test_a_drag_resting_between_two_frames_keeps_its_value_and_shows_the_nearer_frame(frame_rate):
    indices, graphic, plot, frame_timings = make_viewer(timings(NUM_FRAMES, frame_rate))
    held = frame_timings[500] + 0.3 * (frame_timings[501] - frame_timings[500])
    # the imgui slider reports the mouse's value again on every frame it is held
    for _ in range(6):
        indices.set_dim_index("time", held, cancel_awaiting=True)
        assert (indices["time"], plot.frame, shown_frame(graphic)) == (held, 500, 500)


@pytest.mark.parametrize("frame_rate", [None, FRAME_RATE], ids=["frame numbers", "times"])
def test_each_step_of_play_shows_the_next_frame(frame_rate):
    indices, graphic, plot, frame_timings = make_viewer(timings(NUM_FRAMES, frame_rate))
    indices.set_dim_index("time", frame_timings[500])
    step = indices.ref_ranges["time"].step
    for k in range(501, 601):
        indices.set_dim_index("time", indices["time"] + step)
        assert (plot.frame, shown_frame(graphic)) == (k, k)


def test_the_slider_map_takes_the_nearest_frame_where_searchsorted_takes_the_next():
    frame_timings = timings(NUM_FRAMES, FRAME_RATE)
    just_past = frame_timings[500] + 0.3 / FRAME_RATE
    assert frame_slider_map(frame_timings)(just_past) == 500
    assert frame_timings.searchsorted(just_past) == 501


def test_frame_times_come_in_the_unit_asked_for_and_are_none_without_a_frame_rate():
    assert np.allclose(timings(3, 10.0), [0.0, 0.1, 0.2])
    assert np.allclose(timings(3, 10.0, unit="ms"), [0.0, 100.0, 200.0])
    assert timings(3, None) is None
