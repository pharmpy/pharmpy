# The pharmpy visualization module

# Since the python visualization and plotting landscape is rapidly
# evolving and there are many different modules to choose from
# all visualization API calls should be made from this module so
# that we could start using another API more easily.
# Design conciderations:
# We would like to be able to have interactive plots which currently
# means to select a package that can render to html. The two main
# contenders that can do this are the altair and the bokeh libraries.
# Bokeh seems to have a larger community, but the strength in altair
# is its use of the standard vega-lite format, which decouples the
# creation of a plot from the rendering. Altair plots (or rather vega
# plots) can be changed from the html directly via the online vega
# editor. So using altair for now, but expecting to revisit this
# decision shortly.

# Will provide base functions for creating different types of plots
# or other types of visualizations

import altair as alt
import pandas as pd

_chart_width = 500
_chart_height = 500


def scatter_plot_correlation(df, x, y, title=""):
    chart = alt.Chart(df, width=_chart_width, height=_chart_height).mark_circle(size=100).encode(
        alt.X(x),
        alt.Y(y),
        tooltip=[x, y]
    ).properties(
        title=title,
    ).interactive()

    line = alt.Chart(
        pd.DataFrame({x: [min(df[x]), max(df[x])],
                      y: [min(df[y]), max(df[y])]})).mark_line().encode(
            alt.X(x),
            alt.Y(y),
    ).interactive()

    plot = chart + line

    plot = plot.configure_title(fontSize=16)
    plot = plot.configure_axis(labelFontSize=12, titleFontSize=14)
    return plot
