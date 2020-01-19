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


def histogram(values, title=""):
    """Histogram with percentage on y and a rule at mean
       slider for reducing the number of values used.
    """
    df = pd.DataFrame({values.name: values, 'num': list(range(1, len(values) + 1))})

    slider = alt.binding_range(min=1, max=len(values), step=1, name='Number of samples: ')
    selection = alt.selection_single(bind=slider, fields=['num'], name="num",
                                     init={'num': len(values)})

    base = alt.Chart(df).transform_filter('datum.num <= num_num')

    plot = base.transform_joinaggregate(
        total='count(*)'
    ).transform_calculate(
        pct='1 / datum.total'
    ).mark_bar().encode(
        alt.X(f'{values.name}:Q', bin=True),
        alt.Y('sum(pct):Q', axis=alt.Axis(format='%'))
    ).add_selection(
        selection
    ).properties(
        title=title
    )

    rule = base.mark_rule(color='red').encode(
        x=f'mean({values.name}):Q',
        size=alt.value(5)
    )

    return plot + rule
