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
    Plot
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
