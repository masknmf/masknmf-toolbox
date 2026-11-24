import math

import fastplotlib as fpl
from imgui_bundle import imgui


class ReachColorbar(fpl.ui.ImguiColorbar):
    """
    A colorbar whose value axis always spans its histogram and its limits, with vmin and vmax fields in the
    right-click menu.

    fastplotlib's colorbar stops its handles 10% past the histogram and draws a limit beyond that at the end of the
    axis. Here the axis is recomputed on every frame with the mouse button up, from the histogram's span and the
    current limits alone: a limit typed far outside the histogram (-30 and 4000 over a histogram of -0.05 to 0.05)
    widens the axis to show it, and the axis narrows again when the limit comes back. The axis holds still during a
    drag, so a handle let go at its end has room to go further.
    """

    _axis = None

    def __init__(self, graphics, histogram: tuple, limits: tuple[float, float]):
        super().__init__(graphics=graphics, histogram=histogram)
        self._limits_opening = limits

    def set_fit(self, histogram: tuple, limits: tuple[float, float]):
        """
        Take the histogram and the opening limits of another movie.

        Parameters
        ----------
        histogram : tuple
            Counts and bin edges, as np.histogram returns them.
        limits : tuple[float, float]
            vmin, vmax that "Reset limits" restores.
        """
        self.histogram = histogram
        self._limits_opening = limits

    def _span_axis(self) -> tuple[float, float]:
        """The histogram's span widened to the finite limits, with fastplotlib's 10% flank."""
        low, high = self._data_min, self._data_max
        for limit in (self._vmin, self._vmax):
            if math.isfinite(limit):
                low, high = min(low, limit), max(high, limit)
        flank = 0.1 * (high - low)
        return low - flank, high + flank

    def _axis_range(self) -> tuple[float, float]:
        return self._span_axis() if self._axis is None else self._axis

    def update(self):
        if not imgui.is_mouse_down(0):
            axis = self._span_axis()
            if axis != self._axis:
                self._axis = axis
                self._update_bar_texture()
        super().update()

    def _draw_popup(self):
        for attr in ("vmax", "vmin"):
            imgui.set_next_item_width(150)
            _, value = imgui.input_double(attr, getattr(self, f"_{attr}"), format="%.6g")
            if imgui.is_item_deactivated_after_edit() and math.isfinite(value):
                setattr(self, attr, value)
        if imgui.menu_item("Reset limits", "", False)[0]:
            self.vmin, self.vmax = self._limits_opening
        imgui.separator()
        super()._draw_popup()
