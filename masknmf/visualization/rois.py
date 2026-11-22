"""Feathered RGBA overlays and line contours of demixed footprints for the interactive viewers."""

import colorsys
import functools
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Tuple

import matplotlib
import numpy as np
import scipy.ndimage
import skimage.measure
import torch
from fastplotlib.graphics.features import VertexColors, VertexPositions

__all__ = [
    "FootprintContours",
    "FootprintSet",
    "MASK_CUTOFF",
    "SELECTED_ALPHA",
    "feathered_rgba",
    "outline",
    "roi_color",
    "support",
]

# opacity a selected mask is filled at, whatever the overlay opacity
SELECTED_ALPHA = 0.9
MARKED_COLOR = (1.0, 0.15, 0.15)  # footprints marked for deletion
# fraction of a footprint's own peak weight below which the masks and the contours drop its pixels: the halo
MASK_CUTOFF = 0.1
# vertices, which lie about a pixel apart: sigma of the Gaussian a contour is smoothed with along its length
CONTOUR_SMOOTHING = 1.5
# screen px; the highlighted contours are the wider, so they read over the others crossing them
CONTOUR_THICKNESS = (1.0, 2.0)
# z of the contour lines: over the mask overlay at 0.5, the highlighted over the others
CONTOUR_Z = (0.6, 0.7)
TRANSPARENT = (0.0, 0.0, 0.0, 0.0)


def _make_roi_colors() -> np.ndarray:
    # one saturated color per footprint, hues shuffled so neighbors contrast
    hues = np.random.default_rng(0).permutation(180)
    return np.array(
        [[int(round(c * 255)) for c in colorsys.hsv_to_rgb(h / 180.0, 1.0, 1.0)] for h in hues],
        dtype=np.uint8,
    )


ROI_COLORS = _make_roi_colors()


def roi_color(index: int) -> Tuple[int, int, int]:
    """uint8 rgb for a footprint, wrapping past the palette end"""
    return tuple(int(v) for v in ROI_COLORS[index % len(ROI_COLORS)])


def _rim(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a boolean mask, 4-connected."""
    core = mask.copy()
    core[1:, :] &= mask[:-1, :]
    core[:-1, :] &= mask[1:, :]
    core[:, 1:] &= mask[:, :-1]
    core[:, :-1] &= mask[:, 1:]
    return mask & ~core


def support(ypix: np.ndarray, xpix: np.ndarray, lam: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The pixels of a footprint that its mask fills and its contour runs around.

    Pixels under MASK_CUTOFF of the footprint's peak are dropped. Gaps of up to two pixels between the rest are then
    filled at the level of the weights around them (a 3x3 grey closing), so a footprint on alternate rows draws as
    one piece.

    Args:
        ypix (np.ndarray): rows of the footprint's pixels
        xpix (np.ndarray): their columns
        lam (np.ndarray): their weights

    Returns:
        (ypix, xpix, lam) of the pixels drawn
    """
    if not lam.size:
        return ypix, xpix, lam
    keep = lam >= MASK_CUTOFF * lam.max()
    ypix, xpix, lam = ypix[keep], xpix[keep], lam[keep]
    # NOTE: a pixel of margin on every side: the closing erodes whatever touches the edge of its image
    y0, x0 = ypix.min() - 1, xpix.min() - 1
    weights = np.zeros((ypix.max() - y0 + 2, xpix.max() - x0 + 2), np.float32)
    weights[ypix - y0, xpix - x0] = lam
    filled = scipy.ndimage.grey_closing(weights, size=(3, 3), mode="constant")
    weights = np.where(weights > 0, weights, filled)
    ypix, xpix = np.nonzero(weights)
    return ypix + y0, xpix + x0, weights[ypix, xpix]


def _smooth(loop: np.ndarray) -> np.ndarray:
    """
    A closed loop, smoothed along its length by a Gaussian of CONTOUR_SMOOTHING vertices.

    The Gaussian multiplies the loop's Fourier series, so it wraps around the loop. The first harmonic, the loop's
    ellipse (Kuhl & Giardina 1982, elliptic Fourier features), passes whole: a loop around a few pixels keeps its
    size, where the Gaussian alone shrinks it toward a point.

    Args:
        loop (np.ndarray): (n, 2) vertices, the last repeating the first

    Returns:
        (n, 2) vertices, the last repeating the first
    """
    ring = loop[:-1]
    gain = np.exp(-2 * (np.pi * CONTOUR_SMOOTHING * np.fft.rfftfreq(len(ring))) ** 2)
    gain[1] = 1
    ring = np.fft.irfft(np.fft.rfft(ring, axis=0) * gain[:, None], n=len(ring), axis=0)
    return np.concatenate([ring, ring[:1]])


def outline(ypix: np.ndarray, xpix: np.ndarray) -> np.ndarray:
    """
    The smoothed closed loops around a set of pixels.

    Marching squares at one half traces the pixels' outer edges. Diagonal neighbors join, which keeps a one pixel
    wide process in one loop, and a hole or a detached piece gets a loop of its own. Each loop is then smoothed
    along its length, which rounds the pixel steps and neither merges nor drops a loop.

    Args:
        ypix (np.ndarray): rows of the pixels
        xpix (np.ndarray): columns of the pixels

    Returns:
        (n, 2) float32 of (x, y) with pixel centers at the integers, a NaN row after every loop; no pixels, no rows
    """
    if not ypix.size:
        return np.empty((0, 2), np.float32)
    # a pixel of margin on every side, so the loops close around pixels on the mask's edge
    y0, x0 = ypix.min() - 1, xpix.min() - 1
    mask = np.zeros((ypix.max() - y0 + 2, xpix.max() - x0 + 2), bool)
    mask[ypix - y0, xpix - x0] = True
    loops = skimage.measure.find_contours(mask, 0.5, fully_connected="high")
    gap = np.full((1, 2), np.nan)
    return np.concatenate([part for loop in loops for part in (_smooth(loop)[:, ::-1] + (x0, y0), gap)]).astype(np.float32)


def feathered_rgba(shape: Tuple[int, int], comps, selected=(), selected_alpha: float = SELECTED_ALPHA) -> np.ndarray:
    """
    Compose an (ny, nx, 4) uint8 overlay from footprints.

    Args:
        shape (tuple): (ny, nx) of the FOV
        comps: iterable of (ypix, xpix, weight, rgb, fill), weights in 0..1; each pixel takes weight * fill
            as its alpha, and where footprints overlap the higher alpha wins color and coverage
        selected: iterable of (ypix, xpix, weight, rgb), each pixel filled at weight * ``selected_alpha`` with
            a white rim at ``selected_alpha``, drawn over everything else, in order
        selected_alpha (float): opacity the selected footprints peak at
    """
    ny, nx = shape
    rgba = np.zeros((ny, nx, 4), np.uint8)
    best = np.zeros((ny, nx), np.float32)
    for ypix, xpix, weight, rgb, fill in comps:
        color = np.rint(np.asarray(rgb, np.float32) * 255).astype(np.uint8)
        alpha = np.asarray(weight, np.float32) * fill
        win = alpha > best[ypix, xpix]
        yy, xx = ypix[win], xpix[win]
        best[yy, xx] = alpha[win]
        rgba[yy, xx, :3] = color
        rgba[yy, xx, 3] = np.rint(alpha[win] * 255).astype(np.uint8)
    for ypix, xpix, weight, rgb in selected:
        mask = np.zeros((ny, nx), bool)
        mask[ypix, xpix] = True
        rgba[ypix, xpix, :3] = np.rint(np.asarray(rgb, np.float32) * 255).astype(np.uint8)
        rgba[ypix, xpix, 3] = np.rint(np.asarray(weight, np.float32) * selected_alpha * 255).astype(np.uint8)
        rgba[_rim(mask)] = (255, 255, 255, round(selected_alpha * 255))
    return rgba


@dataclass
class FootprintSet:
    """
    Footprints an algorithm produced, as (ypix, xpix, lam) per component; ``colors`` overrides the id palette and
    ``peaks`` (one per footprint, its trace's maximum) scales the overlay so a weak signal draws faint.
    """

    footprints: list
    colors: Optional[np.ndarray] = None
    peaks: Optional[np.ndarray] = None

    @classmethod
    def from_sparse(cls, a: torch.Tensor, shape: Tuple[int, int]) -> "FootprintSet":
        """
        Args:
            a (torch.Tensor): sparse (pixels, components) footprints
            shape (tuple): (ny, nx) the pixel axis unravels to
        """
        a = a.coalesce().cpu()
        rows, cols = a.indices().numpy()
        values = a.values().numpy().astype(np.float32)
        keep = values > 0
        rows, cols, values = rows[keep], cols[keep], values[keep]
        order = np.argsort(cols, kind="stable")
        rows, cols, values = rows[order], cols[order], values[order]
        bounds = np.searchsorted(cols, np.arange(a.shape[1] + 1))
        footprints = []
        for k in range(a.shape[1]):
            lo, hi = bounds[k], bounds[k + 1]
            ypix, xpix = np.divmod(rows[lo:hi], shape[1])
            footprints.append((ypix.astype(np.int32), xpix.astype(np.int32), values[lo:hi]))
        return cls(footprints)

    def __len__(self) -> int:
        return len(self.footprints)

    def color(self, index: int) -> Tuple[float, float, float]:
        if self.colors is not None:
            return tuple(float(v) for v in self.colors[index])
        return tuple(v / 255.0 for v in roi_color(index))

    def recolor(self, values=None, cmap: str = "viridis"):
        """Color every footprint by its rank in ``values`` (no value: gray), or back to the id palette with None."""
        if values is None:
            self.colors = None
            return
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        ranks = np.zeros(len(values))
        ranks[finite] = np.argsort(np.argsort(values[finite])) / max(finite.sum() - 1, 1)
        colors = np.asarray(matplotlib.colormaps[cmap](ranks))[:, :3]
        colors[~finite] = 0.5
        self.colors = colors

    @property
    def areas(self) -> np.ndarray:
        """Pixel count of every footprint."""
        return np.array([len(ypix) for ypix, _xpix, _lam in self.footprints], dtype=np.int64)

    @functools.cached_property
    def supports(self) -> list:
        """The ``support`` of every footprint, in footprint order. Computed once, from the footprints as they are then."""
        return [support(*footprint) for footprint in self.footprints]

    def outlines(self) -> list:
        """The ``outline`` of every footprint's support, in footprint order."""
        return [outline(ypix, xpix) for ypix, xpix, _lam in self.supports]

    def mask_colors(self, marked: Iterable[int] = (), grouped: Optional[Mapping[int, Tuple[float, float, float]]] = None) -> np.ndarray:
        """(footprints, 3) color of every mask and its contour: MARKED_COLOR if ``marked``, else its ``grouped`` color, else its own."""
        colors = np.array([self.color(k) for k in range(len(self))], np.float32).reshape(-1, 3)
        for k, color in (grouped or {}).items():
            colors[k] = color
        colors[list(marked)] = MARKED_COLOR
        return colors

    def rgba(
        self,
        shape: Tuple[int, int],
        opacity: float,
        selected: Optional[int] = None,
        marked: Iterable[int] = (),
        grouped: Optional[Mapping[int, Tuple[float, float, float]]] = None,
        selected_opacity: float = SELECTED_ALPHA,
        by_peak: bool = True,
    ) -> np.ndarray:
        """
        (ny, nx, 4) uint8 overlay. With ``by_peak`` a pixel's alpha is its weight times the footprint's peak
        against one maximum over the field, so the masks read like the signals movie; without, every footprint
        is scaled to its own peak. A footprint draws the pixels of its ``support``, which its contour outlines.
        ``grouped`` (index -> rgb) and then ``selected`` are feathered to ``selected_opacity`` at
        their own peak with a white rim, and ``marked`` footprints are drawn in MARKED_COLOR.
        """
        grouped = dict(grouped or {})
        colors = self.mask_colors(marked, grouped)
        peaks = np.ones(len(self.footprints), np.float32) if self.peaks is None else np.asarray(self.peaks, np.float32)
        top = max((float(lam.max()) * peaks[k] for k, (_y, _x, lam) in enumerate(self.footprints) if lam.size), default=0.0) or 1.0
        comps, highlighted = [], []
        for k, (ypix, xpix, lam) in enumerate(self.supports):
            if not lam.size:
                continue
            scale = peaks[k] / top if by_peak else 1.0 / float(lam.max())
            comps.append((ypix, xpix, lam * scale, colors[k], opacity))
        picks = [k for k in grouped if k != selected]
        if selected is not None:
            picks.append(selected)
        for k in picks:
            ypix, xpix, lam = self.supports[k]
            if not lam.size:
                continue
            highlighted.append((ypix, xpix, lam / lam.max(), colors[k]))
        return feathered_rgba(shape, comps, highlighted, selected_opacity)


class FootprintContours:
    """
    The outlines of footprints, drawn as lines over image panels, each in its footprint's color.

    One vertex buffer holds every outline and each panel reads it through two lines. A footprint's contour is drawn
    by the lower line, or by the upper and wider one while it is highlighted. Every footprint has its own vertices,
    so footprints that overlap all show; a label image holds one footprint per pixel.
    """

    def __init__(self, outlines: list):
        """
        Args:
            outlines (list): per footprint, its ``outline``
        """
        # NOTE: a GPU buffer cannot be empty, so a NaN vertex follows the outlines and results without footprints still draw
        self._counts = [*map(len, outlines), 1]  # vertices per footprint, then that one
        self._positions = VertexPositions(np.concatenate([*outlines, np.full((1, 2), np.nan, np.float32)]))
        self._colors = tuple(VertexColors(TRANSPARENT, sum(self._counts)) for _ in CONTOUR_Z)
        self._opacities = (0.0, 0.0)
        self._lines = {}  # subplot -> its two lines

    def add_to(self, subplot):
        """Draw the contours in ``subplot``, at the opacities they have in the others."""
        # NOTE: in pygfx's default alpha mode a line writes depth where it is transparent too and draws before the mask
        # overlay, so the transparent vertices would cut their outlines out of the masks
        self._lines[subplot] = tuple(
            subplot.add_line(self._positions, colors=colors, thickness=thickness, alpha_mode="blend", offset=(0, 0, z))
            for colors, thickness, z in zip(self._colors, CONTOUR_THICKNESS, CONTOUR_Z)
        )
        for line in self._lines[subplot]:
            # a click lands on the image under the line
            line.world_object.material.pick_write = False
        self.set_opacity(*self._opacities)

    def remove_from(self, subplot):
        for line in self._lines.pop(subplot):
            subplot.delete_graphic(line)

    def set_opacity(self, others: float, highlighted: float):
        """Opacity of the highlighted contours and of the others; 0 hides them."""
        self._opacities = (others, highlighted)
        for lines in self._lines.values():
            for line, opacity in zip(lines, self._opacities):
                line.visible, line.alpha = opacity > 0, opacity

    def set_colors(self, colors: np.ndarray, highlighted: Iterable[int] = ()):
        """
        Color every footprint's contour, and draw the ``highlighted`` ones in the upper line.

        Args:
            colors (np.ndarray): (footprints, 3) rgb
            highlighted: indices of the footprints the upper line draws; the lower line draws the rest
        """
        upper = np.zeros(len(colors), bool)
        upper[list(highlighted)] = True
        for buffer, drawn in zip(self._colors, (~upper, upper)):
            # a transparent row for the NaN vertex
            rgba = np.pad(np.column_stack([colors, drawn]), ((0, 1), (0, 0)))
            buffer[:, :] = np.repeat(rgba, self._counts, axis=0).astype(np.float32)
