import pharmpy.visualization
from pharmpy.deps import altair as alt
from pharmpy.deps import pandas as pd

from .data import get_observations


def plot_iofv_vs_iofv(iofv1, iofv2, name1, name2):
    """Plot individual OFV of two models against each other

    Parameters
    ----------
    iofv1 : pd.Series
        Estimated iOFV of the first model
    iofv2 : pd.Series
        Estimated iOFV of the second model
    name1 : str
        Name of first model
    name2 : str
        Name of second model

    Results
    -------
    alt.Chart
        Scatterplot

    """
    x_label = f'{name1} iOFV'
    y_label = f'{name2} iOFV'
    df = pd.DataFrame(
        {
            x_label: iofv1,
            y_label: iofv2,
        }
    )
    id_name = df.index.name
    df = df.reset_index()
    plot = pharmpy.visualization.scatter_plot_correlation(
        df, x_label, y_label, tooltip_columns=[id_name], title='iOFV vs iOFV'
    )
    return plot


def plot_individual_predictions(model, predictions=None, individuals=None):
    """Plot DV and predictions grouped on individuals

    Parameters
    ----------
    model : Model
        Previously run Pharmpy model.
    predictions : list
        A list of names of predictions to plot. None for all available
    individuals : list
        A list of individuals to include. None for all individuals

    Returns
    -------
    alt.Chart
        Plot

    """
    res = model.modelfit_results
    pred = res.predictions
    if pred is None:
        raise ValueError("No predictions available in modelfit_results")
    obs = get_observations(model)
    indexcols = pred.index.names
    idcol = indexcols[0]
    idvcol = indexcols[1]

    data = pred.join(obs).reset_index()
    data = data.melt(id_vars=indexcols)

    if individuals is not None:
        data = data[data[idcol].isin(individuals)]
    if predictions is not None:
        dvcol = obs.name
        data = data[data['variable'].isin(predictions + [dvcol])]

    plot = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(x=idvcol, y='value', color='variable', tooltip=[idvcol, 'value'])
        .facet(f'{idcol}:N', columns=5)
    )
    return plot
