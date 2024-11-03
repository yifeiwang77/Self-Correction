from typing import Any, Mapping, Optional, Sequence

from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from analysis.metrics.binomial import ValueCI, error_bars


def stem_plot(
    values: Mapping[str, ValueCI],
    *,
    axhlines: Sequence[float] = (0.0,),
    xcoords: Optional[Sequence[float]] = None,
    multiplier: float = 1.0,
    basefmt: str = "None",
    errorlabel: str = "",
    axes: Optional[Axes] = None,
    **kwargs: Any,
) -> Axes:
    """Plot values and confidence intervals using a stem plot

    values is a dictionary mapping X labels to ValueCI objects.

    axhlines can be used to mark important Y values, similar to the background grid.

    multiplier can be used to scale values, for instance from the interval [0, 1] to
    [0, 100].
    """
    if axes is None:
        _, axes = plt.subplots()
        axes.grid(False)
    if xcoords is None:
        xcoords = list(range(len(values)))
    ycoords = [val.value * multiplier for val in values.values()]
    axes.set_xticks(xcoords, values.keys())
    axes.set_xlabel("Results File")
    add_axhlines(axes=axes, ycoords=[y * multiplier for y in axhlines])

    axes.stem(
        xcoords,
        ycoords,
        basefmt=basefmt,
        **kwargs,
    )

    # axes.stem() doesn't support error bars, so they need to be overlaid separately
    axes.errorbar(
        xcoords,
        ycoords,
        yerr=error_bars(tuple(values.values()), multiplier=multiplier),
        marker="None",
        linestyle="None",
        ecolor="black",
        capsize=5.0,
        label=errorlabel,
    )
    axes.legend()
    return axes


def add_axhlines(
    axes: Axes,
    ycoords: Sequence[float],
    *,
    color: str = "black",
    alpha: float = 0.5,
    linewidth: float = 0.5,
    **kwargs: Any,
) -> None:
    """Add horizontal lines to signify important Y values

    The lines act as a trimmed-down version of the background grid and are styled to
    look like it. If the grid is enabled, it makes little sense to use this function.
    """
    for yval in ycoords:
        axes.axhline(yval, color=color, alpha=alpha, linewidth=linewidth, **kwargs)
