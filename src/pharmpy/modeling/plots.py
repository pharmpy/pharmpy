import altair as alt

import pharmpy.visualization
from pharmpy.data import PharmDataFrame


def plot_iofv_vs_iofv(model, other):
    """Plot individual OFV of two models against each other

    Parameters
    ----------
    model : Model
        The first model
    other : Model
        The second model

    Results
    -------
    alt.Chart
        Scatterplot

    """
    x_label = f'{model.name} iOFV'
    y_label = f'{other.name} iOFV'
    df = PharmDataFrame(
        {
            x_label: model.modelfit_results.individual_ofv,
            y_label: other.modelfit_results.individual_ofv,
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
    obs = model.dataset.pharmpy.observations
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
