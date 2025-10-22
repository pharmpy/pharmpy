# The pharmpy visualization module

# Since the python visualization and plotting landscape is rapidly
# evolving and there are many different modules to choose from
# all visualization API calls should be made from this module so
# that we could start using another API more easily.
# Design considerations:
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

from pharmpy.deps import altair as alt
from pharmpy.deps import pandas as pd

_chart_width = 500
_chart_height = 500


def scatter_plot_correlation(df, x, y, tooltip_columns=None, title=""):
    if tooltip_columns is None:
        tooltip_columns = []

    chart = (
        alt.Chart(df, width=_chart_width, height=_chart_height)
        .mark_circle(size=100)
        .encode(alt.X(x), alt.Y(y), tooltip=[x, y] + tooltip_columns)
        .properties(
            title=title,
        )
        .interactive()
    )

    line = (
        alt.Chart(pd.DataFrame({x: [min(df[x]), max(df[x])], y: [min(df[y]), max(df[y])]}))
        .mark_line()
        .encode(
            alt.X(x),
            alt.Y(y),
        )
        .interactive()
    )

    plot = chart + line

    plot = plot.configure_title(fontSize=16)
    plot = plot.configure_axis(labelFontSize=12, titleFontSize=14)
    return plot


def scatter_matrix(df):
    """Scatter matrix plot

    Each column will be scatter plotted against all columns.
    """

    base = (
        alt.Chart(df)
        .transform_fold(list(df.columns), as_=['key_x', 'value_x'])
        .transform_fold(list(df.columns), as_=['key_y', 'value_y'])
        .encode(
            x=alt.X('value_y:Q', title=None, scale=alt.Scale(zero=False)),
            y=alt.Y('value_x:Q', title=None, scale=alt.Scale(zero=False)),
        )
        .properties(width=150, height=150)
    )

    plot = (
        alt.layer(
            base.mark_circle(),
            # base.transform_regression('value_y', 'value_x', method='poly', order=4).mark_line(
            #     color='red'
            # ),
        )
        .facet(
            column=alt.Column('key_x:N', sort=list(df.columns), title=None),
            row=alt.Row('key_y:N', sort=list(reversed(df.columns)), title=None),
        )
        .resolve_scale(x='independent', y='independent')
        .configure_header(labelFontStyle='bold')
    )
    return plot


def line_plot(df, x, title='', xlabel='', ylabel='', legend_title=''):
    """Line plot for multiple lines

    Parameters
    ----------
    df : pd.DataFrame
         DataFrame with one x column and multiple columns with y values
    x
         Name of the x column
    title : str
         Plot title
    xlabel : str
         Label of the x-axis
    ylabel : str
         Label of the y-axis
    legend_title : str
         Title of the legend

    """
    df = df.melt(id_vars=[x])
    plot = (
        alt.Chart(df)
        .mark_line()
        .encode(
            alt.X(f'{x}:Q', title=xlabel),
            alt.Y('value:Q', title=ylabel),
            color=alt.Color(
                'variable:N',
                sort=None,
                legend=alt.Legend(
                    title=legend_title,
                    orient='top-left',
                    fillColor='#EEEEEE',
                    padding=10,
                    cornerRadius=10,
                ),
            ),
        )
        .properties(
            title=title,
            width=800,
            height=300,
        )
        .configure_legend(labelLimit=0)
    )
    return plot


def histogram(values, title=""):
    """Histogram with percentage on y and a rule at mean
    slider for reducing the number of values used.
    """
    df = pd.DataFrame({values.name: values, 'num': list(range(1, len(values) + 1))})

    slider = alt.binding_range(min=1, max=len(values), step=1, name='Number of samples: ')
    selection = alt.selection_point(
        bind=slider,
        fields=['num'],
        name="num",
        value=len(values),
    )

    base = alt.Chart(df).transform_filter('datum.num <= num_num')

    plot = (
        base.transform_joinaggregate(total='count(*)')
        .transform_calculate(pct='1 / datum.total')
        .mark_bar()
        .encode(
            alt.X(f'{values.name}:Q', bin=alt.Bin(maxbins=50)),
            alt.Y('sum(pct):Q', axis=alt.Axis(format='%')),
        )
        .add_params(selection)
        .properties(
            title=title,
            width=800,
            height=300,
        )
    )

    rule = base.mark_rule(color='red').encode(x=f'mean({values.name}):Q', size=alt.value(5))

    return plot + rule


def facetted_histogram(df):
    """Facet of one histogram per column with cross filter interaction"""
    brush = alt.selection_interval(encodings=['x'])

    base = (
        alt.Chart()
        .mark_bar()
        .encode(
            x=alt.X(alt.repeat('column'), type='quantitative', bin=alt.Bin(maxbins=20)),
            y=alt.Y('count()', axis=alt.Axis(title='')),
        )
        .properties(width=200, height=150)
    )

    background = base.encode(color=alt.value('#ddd')).add_params(brush)

    highlight = base.transform_filter(brush)

    chart = alt.layer(background, highlight, data=df).repeat(column=list(df.columns))

    return chart


def _default_formatter(x):
    if pd.isna(x):
        return ""
    if isinstance(x, float):
        return "{:,.4f}".format(x)
    return x


def display_table(df, format=None, remove_nan_columns=True):
    if remove_nan_columns:
        df = df.dropna(axis=1, how='all')
    if format is not None:
        format = {
            key: lambda x, value=value: "" if pd.isna(x) else value.format(x)
            for key, value in format.items()
            if key in df.columns
        }
    else:
        format = _default_formatter
    styler = df.style.format(format)

    from itables import show

    show(
        styler,
        buttons=["copyHtml5", "csvHtml5"],
        style='width: fit-content; float: left',
        allow_html=True,
        pageLength=50,
    )
