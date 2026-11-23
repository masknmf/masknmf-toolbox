import fastplotlib as fpl
from imgui_bundle import imgui


class ReachColorbar(fpl.ui.ImguiColorbar):
    """
    A colorbar over a histogram whose value axis grows to take in a limit set past its end.

    fastplotlib's colorbar stops its handles 10% past the histogram and draws a typed limit beyond that at the
    end of the axis. Here, once the mouse button is up, an end of the axis that a limit lies past moves ``GROWTH`` of
    the axis beyond that limit: dragging a handle to the end and letting go makes room to drag further, and a typed
    limit is drawn where it is. "Reset to fit" in the right-click menu brings back the limits and the axis of the fit.
    """

    GROWTH = 0.25
    # a limit this fraction of the axis past its end is at the end: float32 limits against float64 histogram edges
    TOLERANCE = 1e-3

    def __init__(self, graphics, histogram: tuple, limits: tuple[float, float]):
        super().__init__(graphics=graphics, histogram=histogram)
        self._limits_fit = limits

    def set_fit(self, histogram: tuple, limits: tuple[float, float]):
        """
        Take the histogram and the limits of a new fit; the axis goes back to the histogram's span.

        Parameters
        ----------
        histogram : tuple
            Counts and bin edges, as np.histogram returns them.
        limits : tuple[float, float]
            vmin, vmax that "Reset to fit" restores.
        """
        self.histogram = histogram
        self._limits_fit = limits

    def update(self):
        super().update()
        if not imgui.is_mouse_down(0):
            self._grow_to_limits()

    def _grow_to_limits(self):
        span = self._data_max - self._data_min
        low, high = self._data_min, self._data_max
        if self._vmin < low - self.TOLERANCE * span:
            low = self._vmin - self.GROWTH * span
        if self._vmax > high + self.TOLERANCE * span:
            high = self._vmax + self.GROWTH * span
        if (low, high) != (self._data_min, self._data_max):
            self.data_range = (low, high)

    def _draw_popup(self):
        if imgui.menu_item("Reset to fit", "", False)[0]:
            self.histogram = self._histogram
            self.vmin, self.vmax = self._limits_fit
        super()._draw_popup()
