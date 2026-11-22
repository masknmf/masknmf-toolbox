"""
A contour runs smoothly around the pixels its mask fills, in the mask's color, and footprints that overlap each keep
theirs. Footprints marked to merge draw as one mask and one contour.
"""

import fastplotlib as fpl
import numpy as np
import pytest
import scipy.ndimage
from rendercanvas.offscreen import RenderCanvas

from masknmf.visualization.rois import MARKED_COLOR, MASK_CUTOFF, FootprintContours, FootprintSet, outline, support, union

SHAPE = (16, 16)
COLORS = RED, BLUE, WHITE = (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)
MASK_BYTES = (0, 255, 0, 255)


def footprint(mask: np.ndarray, weight: float = 1.0) -> tuple:
    """(ypix, xpix, lam) of a boolean mask, every pixel at ``weight``."""
    ypix, xpix = np.nonzero(mask)
    return ypix.astype(np.int32), xpix.astype(np.int32), np.full(len(ypix), weight, np.float32)


def disc(radius: float) -> np.ndarray:
    """Boolean mask of the pixels within ``radius`` of the center of a (4 * radius + 1) square."""
    rows, cols = np.mgrid[: 4 * radius + 1, : 4 * radius + 1] - 2 * radius
    return np.hypot(rows, cols) <= radius


def edge_midpoints(mask: np.ndarray) -> np.ndarray:
    """(x, y) of the middle of every pixel edge with ``mask`` on one side only. Shape (n_edges, 2)."""
    padded = np.pad(mask, 1)
    rows, cols = np.nonzero(padded[1:] != padded[:-1])
    below = np.column_stack([cols - 1, rows - 0.5])
    rows, cols = np.nonzero(padded[:, 1:] != padded[:, :-1])
    beside = np.column_stack([cols - 0.5, rows - 1])
    return np.concatenate([below, beside])


def loops_of(vertices: np.ndarray) -> list:
    """The loops of an outline, split at its NaN rows, each without that row."""
    ends = np.flatnonzero(np.isnan(vertices[:, 0]))
    return [loop[:-1] for loop in np.split(vertices, ends + 1) if len(loop)]


def turning(loop: np.ndarray) -> float:
    """Radians a closed loop turns through, left and right both counted: 2 pi for a convex one."""
    headings = np.arctan2(*np.diff(loop, axis=0).T[::-1])
    turns = np.diff(np.append(headings, headings[0]))
    return float(np.abs((turns + np.pi) % (2 * np.pi) - np.pi).sum())


def draw(figure) -> np.ndarray:
    """The rendered frame, rgb."""
    figure.show()
    for _ in range(3):
        frame = figure.canvas.draw()
    return np.asarray(frame)[..., :3]


@pytest.fixture
def panel():
    """A panel as the viewer makes them, a black movie frame under a mask overlay, and its frame without contours."""
    figure = fpl.Figure(size=(320, 320), canvas=RenderCanvas(size=(320, 320), pixel_ratio=1))
    figure[0, 0].axes.visible = False
    image = figure[0, 0].add_image(np.zeros(SHAPE, np.float32), cmap="gray", vmin=0, vmax=1)
    masks = np.zeros((*SHAPE, 4), np.uint8)
    masks[2:14, 2:14] = MASK_BYTES
    overlay = figure[0, 0].add_image(masks, vmin=0, vmax=255, alpha_mode="blend", offset=(0, 0, 0.5))
    for tile in overlay.world_object.children:
        tile.material.pick_write = False
    return figure, image, draw(figure)


def pixels_of(frame: np.ndarray, blank: np.ndarray, color: tuple) -> np.ndarray:
    """(row, col) of the pixels the contours changed from ``blank`` that are nearest ``color``, of the contour colors, the black frame and the mask."""
    palette = np.array([*np.multiply(COLORS, 255), (0, 0, 0), MASK_BYTES[:3]])
    nearest = np.linalg.norm(frame[..., None, :] - palette, axis=-1).argmin(axis=-1)
    return np.argwhere((frame != blank).any(axis=-1) & (nearest == COLORS.index(color)))


@pytest.mark.parametrize("seed", range(20))
def test_outline_has_a_closed_loop_per_piece_and_per_hole(seed):
    mask = np.random.default_rng(seed).random((9, 13)) < 0.4
    # marching squares joins the pixels over corners, so the background joins over edges only
    _, n_pieces = scipy.ndimage.label(mask, structure=np.ones((3, 3)))
    _, n_background = scipy.ndimage.label(~np.pad(mask, 1))
    loops = loops_of(outline(*np.nonzero(mask)))
    assert len(loops) == n_pieces + n_background - 1
    assert all((loop[0] == loop[-1]).all() for loop in loops)


def test_outline_of_a_disc_is_smooth_and_on_its_edge():
    mask = disc(radius=6)
    (loop,) = loops_of(outline(*np.nonzero(mask)))
    exact = edge_midpoints(mask)
    assert turning(exact[np.argsort(np.arctan2(*(exact - 12).T))][np.r_[: len(exact), 0]]) > 4 * np.pi
    assert turning(loop) < 1.05 * 2 * np.pi
    distances = np.linalg.norm(loop[:, None] - exact[None], axis=-1).min(axis=1)  ## (n_vertices,)
    assert distances.max() < 0.5


def test_outline_around_a_few_pixels_keeps_its_size():
    (loop,) = loops_of(outline(np.array([3]), np.array([5])))
    np.testing.assert_allclose(np.linalg.norm(loop - (5, 3), axis=1), 0.5, atol=1e-6)
    block = np.zeros((6, 6), bool)
    block[2:4, 2:4] = True
    (loop,) = loops_of(outline(*np.nonzero(block)))
    np.testing.assert_allclose(np.linalg.norm(loop - (2.5, 2.5), axis=1), np.hypot(1, 0.5), rtol=0.05)


def test_outline_keeps_thin_and_detached_pieces():
    assert len(loops_of(outline(*np.nonzero(np.eye(8, dtype=bool))))) == 1
    detached = np.zeros((4, 12), bool)
    detached[1:3, [1, 2, 9, 10]] = True
    assert len(loops_of(outline(*np.nonzero(detached)))) == 2
    assert outline(np.empty(0, int), np.empty(0, int)).shape == (0, 2)


def test_support_drops_the_halo_and_fills_alternate_rows():
    core = np.zeros((12, 12), bool)
    core[3:10:2, 3:9] = True
    halo = scipy.ndimage.binary_dilation(core, iterations=2) & ~core
    ypix, xpix = np.nonzero(core | halo)
    lam = np.where(core[ypix, xpix], np.linspace(0.5, 1.0, len(ypix)), 0.5 * MASK_CUTOFF).astype(np.float32)
    lam[(ypix == 5) & (xpix == 6)] = 2 * MASK_CUTOFF  # a dim pixel between bright ones keeps its weight
    weights = np.zeros(core.shape, np.float32)
    weights[support(ypix, xpix, lam)[:2]] = support(ypix, xpix, lam)[2]
    filled = np.zeros_like(core)
    filled[3:10, 3:9] = True
    np.testing.assert_array_equal(weights > 0, filled)
    np.testing.assert_array_equal(weights[core], lam[core[ypix, xpix]])
    assert (weights[filled & ~core] >= 0.5).all()
    assert support(np.empty(0, int), np.empty(0, int), np.empty(0, np.float32))[2].size == 0


def test_mask_and_contour_draw_the_same_pixels():
    core = np.zeros((12, 12), bool)
    core[3:10:2, 3:9] = True
    footprints = FootprintSet([footprint(core)])
    drawn = footprints.rgba(core.shape, opacity=1.0, by_peak=False)[..., 3] > 0
    assert drawn.sum() == 7 * 6
    np.testing.assert_array_equal(footprints.outlines[0], outline(*np.nonzero(drawn)))


def test_a_pixel_belongs_to_the_lowest_footprint_whose_mask_fills_it():
    core = np.zeros((12, 12), bool)
    core[3:10:2, 3:9] = True
    footprints = FootprintSet([footprint(np.eye(12, dtype=bool)), footprint(core), footprint(core)])
    assert footprints.at(4, 6) == 1  # a row the closing filled
    assert footprints.at(5, 5) == 0
    assert footprints.at(0, 11) is None


def test_marked_and_grouped_footprints_take_those_colors():
    footprints = FootprintSet([footprint(np.eye(4, dtype=bool))] * 3, colors=np.array([RED, RED, RED]))
    colors = footprints.mask_colors(marked=[1, 2], grouped={0: BLUE, 1: BLUE})
    np.testing.assert_allclose(colors, [BLUE, MARKED_COLOR, MARKED_COLOR])
    assert FootprintSet([]).mask_colors().shape == (0, 3)


def halves() -> tuple:
    """Boolean masks of two 8 x 4 blocks a column apart, and of a block away from both."""
    left, right, apart = np.zeros((3, *SHAPE), bool)
    left[3:11, 2:6] = True
    right[3:11, 7:11] = True
    apart[13:15, 12:15] = True
    return left, right, apart


def test_union_scales_every_footprint_to_the_first_peak_and_takes_the_larger_weight():
    left, right, _apart = halves()
    right[:, 5] = left[:, 5]  # the two share a column
    bright, dim = np.zeros((2, *SHAPE), np.float32)
    bright[left] = np.linspace(1.0, 4.0, left.sum())
    dim[right] = 0.5 * MASK_CUTOFF
    dim[:, 5] /= 2  # at half its peak in the shared column: over the bright one in the top rows, under it in the others
    footprints = [(*np.nonzero(weights), weights[weights > 0]) for weights in (bright, dim)]

    ypix, xpix, lam = union(footprints)

    merged = np.zeros(SHAPE, np.float32)
    merged[ypix, xpix] = lam
    np.testing.assert_allclose(merged, np.maximum(bright, dim * (4.0 / dim.max())))
    assert (merged[:, 5] > bright[:, 5]).any() and (merged[:, 5] == bright[:, 5])[left[:, 5]].any()
    empty = footprint(np.zeros(SHAPE, bool))
    assert union([empty, empty])[2].size == 0
    np.testing.assert_array_equal(union([empty, footprints[0]])[2], footprints[0][2])


def test_merged_footprints_draw_as_one_mask_and_one_contour_until_unmerged():
    left, right, apart = halves()
    footprints = FootprintSet(
        [footprint(left, weight=3.0), footprint(apart), footprint(right, weight=0.2)],
        colors=np.array([RED, WHITE, BLUE]),
        peaks=np.array([0.5, 3.0, 1.0]),
    )
    outlines = list(footprints.outlines)
    apart_only = footprints.rgba(SHAPE, opacity=1.0, by_peak=False)
    alpha_left = footprints.rgba(SHAPE, opacity=1.0, by_peak=True)[left, 3]

    footprints.merge([[2, 0]])

    assert footprints.merges == [[0, 2]]
    merged = footprints.rgba(SHAPE, opacity=1.0, by_peak=False)
    both = np.zeros(SHAPE, bool)
    both[3:11, 2:11] = True  # the column between the halves is filled
    np.testing.assert_array_equal(merged[..., 3] > 0, both | apart)
    np.testing.assert_array_equal(merged[both][:, :3], np.broadcast_to((255, 0, 0), (both.sum(), 3)))
    assert len(loops_of(footprints.outlines[0])) == 1 and len(footprints.outlines[2]) == 0
    np.testing.assert_array_equal(footprints.outlines[0], outline(*np.nonzero(both)))
    assert footprints.outlines[1] is outlines[1]
    # with the signal peak weighting, the union is as bright as its lowest footprint was
    assert 0 < alpha_left.max() < 255
    np.testing.assert_array_equal(footprints.rgba(SHAPE, opacity=1.0, by_peak=True)[both, 3], alpha_left.max())
    np.testing.assert_array_equal(footprints.areas, [left.sum(), apart.sum(), right.sum()])

    footprints.merge([])

    np.testing.assert_array_equal(footprints.rgba(SHAPE, opacity=1.0, by_peak=False), apart_only)
    for redrawn, original in zip(footprints.outlines, outlines):
        np.testing.assert_array_equal(redrawn, original)


def test_a_footprint_that_leaves_a_merge_group_draws_as_itself_again():
    left, right, apart = halves()
    footprints = FootprintSet([footprint(left), footprint(right), footprint(apart)])
    own = [set(zip(*support[:2])) for support in footprints.supports]

    footprints.merge([[0, 1]])
    footprints.merge([[1, 2]])

    drawn = [set(zip(*support[:2])) for support in footprints.supports]
    assert drawn == [own[0], own[1] | own[2], set()]
    for k, (ypix, xpix, _lam) in enumerate(footprints.supports):
        np.testing.assert_array_equal(footprints.outlines[k], outline(ypix, xpix))


@pytest.mark.parametrize("groups", [[[0]], [[0, 0]], [[0, 1], [1, 2]]], ids=["one footprint", "one footprint twice", "a footprint in two groups"])
def test_merge_groups_hold_two_or_more_footprints_and_share_none(groups):
    left, right, apart = halves()
    footprints = FootprintSet([footprint(left), footprint(right), footprint(apart)])
    with pytest.raises(ValueError, match="merge group"):
        footprints.merge(groups)
    assert footprints.merges == []


def test_overlapping_footprints_each_keep_a_contour_and_the_highlighted_ones_are_on_top(panel):
    figure, image, blank = panel
    square = np.zeros(SHAPE, bool)
    square[4:12, 4:12] = True
    footprints = FootprintSet([footprint(square), footprint(square), footprint(np.roll(square, 3, axis=1))])
    colors = np.array(COLORS)
    contours = FootprintContours(footprints.outlines)
    contours.add_to(figure[0, 0])
    contours.set_opacity(1.0, 1.0)

    # footprints 0 and 1 share every vertex and one line draws both: the later one is on top
    contours.set_colors(colors)
    every = draw(figure)
    n_blue, n_white = len(pixels_of(every, blank, BLUE)), len(pixels_of(every, blank, WHITE))
    assert n_blue > 0 and n_white > 0 and len(pixels_of(every, blank, RED)) == 0

    # the upper line is the wider: footprint 0 covers footprint 1, and footprint 2 where they cross
    contours.set_colors(colors, highlighted=[0])
    frame = draw(figure)
    red = pixels_of(frame, blank, RED)
    assert len(red) > n_blue and len(pixels_of(frame, blank, BLUE)) == 0 and 0 < len(pixels_of(frame, blank, WHITE)) < n_white

    # a click on a contour lands on the image under it
    row, col = red[len(red) // 2]
    assert figure.renderer.get_pick_info((col, row))["world_object"] in image.world_object.children

    contours.set_opacity(1.0, 0.0)
    frame = draw(figure)
    assert len(pixels_of(frame, blank, RED)) == 0 and len(pixels_of(frame, blank, BLUE)) > 0

    contours.set_opacity(1.0, 1.0)
    contours.set_colors(colors)
    np.testing.assert_array_equal(draw(figure), every)

    # with nothing highlighted the upper line is transparent everywhere, and the masks under it are whole
    contours.set_opacity(0.0, 1.0)
    np.testing.assert_array_equal(draw(figure), blank)

    contours.set_opacity(1.0, 1.0)
    contours.remove_from(figure[0, 0])
    np.testing.assert_array_equal(draw(figure), blank)


def test_highlighted_footprint_is_left_out_of_the_lower_line(panel):
    figure, _image, blank = panel
    square = np.zeros(SHAPE, bool)
    square[4:12, 4:12] = True
    contours = FootprintContours(FootprintSet([footprint(square)]).outlines)
    contours.add_to(figure[0, 0])
    contours.set_colors(np.array([BLUE]), highlighted=[0])
    contours.set_opacity(1.0, 0.5)
    frame = draw(figure)
    drawn = (frame != blank).any(axis=-1)
    # blue at half opacity; the lower line under it would bring it to 255
    assert drawn.any() and frame[drawn][:, 2].max() < 240


def test_results_without_footprints_draw_nothing(panel):
    figure, _image, blank = panel
    footprints = FootprintSet([])
    contours = FootprintContours(footprints.outlines)
    contours.add_to(figure[0, 0])
    contours.set_opacity(1.0, 1.0)
    contours.set_colors(footprints.mask_colors())
    np.testing.assert_array_equal(draw(figure), blank)
