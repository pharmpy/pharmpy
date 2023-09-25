import re
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import pharmpy.visualization
from pharmpy.internals.expr.units import unit_string
from pharmpy.model import Assignment, Model

from .data import get_observations

if TYPE_CHECKING:
    import altair as alt
    import numpy as np
    import pandas as pd
    import scipy
    import sympy
    import sympy.stats as sympy_stats
else:
    from pharmpy.deps import altair as alt
    from pharmpy.deps import numpy as np
    from pharmpy.deps import pandas as pd
    from pharmpy.deps import scipy, sympy, sympy_stats

norm = scipy.stats.norm


def plot_iofv_vs_iofv(iofv1: pd.Series, iofv2: pd.Series, name1: str, name2: str):
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

    Returns
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


def plot_individual_predictions(
    model: Model, predictions: pd.DataFrame, individuals: Optional[List[int]] = None
):
    """Plot DV and predictions grouped on individuals

    Parameters
    ----------
    model : Model
        Previously run Pharmpy model.
    predictions : pd.DataFrame
        One column for each type of prediction
    individuals : list
        A list of individuals to include. None for all individuals

    Returns
    -------
    alt.Chart
        Plot

    """
    obs = get_observations(model)
    indexcols = predictions.index.names
    idcol = indexcols[0]
    idvcol = indexcols[1]

    data = predictions.join(obs).reset_index()
    data = data.melt(id_vars=indexcols)

    if individuals is not None:
        data = data[data[idcol].isin(individuals)]

    plot = (
        alt.Chart(data)
        .mark_line(point=True)
        .encode(x=idvcol, y='value', color='variable', tooltip=[idvcol, 'value'])
        .facet(f'{idcol}:N', columns=5)
    )
    return plot


def plot_transformed_eta_distributions(
    model: Model,
    parameter_estimates: Union[pd.Series, Dict[str, float]],
    individual_estimates: pd.DataFrame,
):
    """Plot transformed eta distributions for all transformed etas

    Parameters
    ----------
    model : Model
        Previously run Pharmpy model.
    parameter_estimates : Union[pd.Series, Dict[str, float]]
        Parameter estimates of model fit
    individual_estimates : pd.DataFrame
        Individual estimates for etas

    Returns
    -------
    alt.Chart
        Plot
    """
    parameter_estimates = dict(parameter_estimates)
    eta_symbols = {sympy.Symbol(name) for name in model.random_variables.etas.names}

    transformations = []
    for s in model.statements.before_odes:
        if isinstance(s, Assignment):
            m = re.match(r'ETA\w(\d+)', s.symbol.name)
            if m:
                symbols = s.expression.free_symbols
                inter = symbols.intersection(eta_symbols)
                if len(inter) == 1:
                    transformations.append((inter.pop(), s.expression))

    x = np.linspace(-2.0, 2.0, 1000)
    i = 1

    df = pd.DataFrame()

    for eta, expr in transformations:
        var = model.random_variables.etas.get_covariance(eta.name, eta.name).subs(
            parameter_estimates
        )
        subdf = pd.DataFrame({'x': x, 'original': norm.pdf(x, scale=float(var) ** 0.5)})
        rv = sympy_stats.Normal('eta', 0, sympy.sqrt(var))
        expr = expr.subs(parameter_estimates).subs({eta: rv})
        curdens = sympy_stats.density(expr)
        densfn = sympy.lambdify(curdens.variables[0], curdens.expr)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            subdf['boxcox'] = densfn(x)
        subdf = pd.melt(subdf, id_vars=['x'], value_name='density')
        eta_name = f'ETA_{i}'
        ebes = pd.DataFrame(
            {'x': individual_estimates[eta.name], 'density': 0.0, 'variable': 'ebes'}
        )
        subdf = pd.concat([subdf, ebes])

        subdf['eta'] = eta_name
        df = pd.concat([df, subdf])
        i += 1

    single = (
        alt.Chart()
        .mark_area(opacity=0.3)
        .encode(
            x=alt.X('x', axis=alt.Axis(labels=False, ticks=False, title=None)),
            y='density',
            color=alt.Color('variable:N', scale=alt.Scale(domain=['original', 'boxcox'])),
        )
    )
    ticks = (
        alt.Chart()
        .mark_tick()
        .encode(
            x='x',
            y='density',
        )
        .transform_filter(alt.datum.variable == 'ebes')
    )
    layer = alt.layer(single, ticks, data=df)

    facet = layer.facet(facet='eta:N', columns=3)
    return facet


def plot_dv_vs_ipred(model: Model, predictions: pd.DataFrame) -> alt.Chart:
    """Plot DV vs IPRED

    Parameters
    ----------
    model : Model
        Pharmpy model
    predictions : pd.DataFrame
        DataFrame containing the predictions

    Returns
    -------
    alt.Chart
        Plot
    """

    obs = get_observations(model)

    if 'CIPREDI' in predictions.columns:
        ipred = 'CIPREDI'
    elif 'IPRED' in predictions.columns:
        ipred = 'IPRED'
    else:
        raise ValueError("Cannot find individual predictions")
    di = model.datainfo
    idv = di.idv_column.name
    dvcol = di.dv_column
    dv = dvcol.name
    dv_unit = dvcol.unit
    idname = di.id_column.name

    predictions = predictions[[ipred]]
    df = predictions.join(obs, how='inner').reset_index()

    if dv_unit == 1:
        unit = ""
    else:
        unit = f" ({unit_string(dv_unit)})"

    chart = (
        alt.Chart(df)
        .mark_line(point=True, opacity=0.7)
        .encode(
            x=alt.X(ipred).title(f"Individual prediction{unit}"),
            y=alt.Y(dv).title(f"Observation{unit}"),
            detail=f"{idname}:N",
            tooltip=[ipred, dv, idname, idv],
        )
        .properties(title="Observations vs. Individual predictions", width=600, height=300)
        .interactive()
    )

    line = (
        alt.Chart(
            pd.DataFrame(
                {'var1': [df[ipred].min(), df[ipred].max()], 'var2': [df[dv].min(), df[dv].max()]}
            )
        )
        .mark_line()
        .encode(x=alt.X('var1'), y=alt.Y('var2'), color=alt.value("#000000"))
    )

    loess_smooth = chart.transform_loess(ipred, dv).mark_line().encode(color=alt.value("#FF0000"))

    layer = chart + loess_smooth + line
    layer = layer.configure_point(size=60)

    return layer


def plot_cwres_vs_idv(model: Model, residuals: pd.DataFrame) -> alt.Chart:
    """Plot CWRES vs idv

    Parameters
    ----------
    model : Model
        Pharmpy model
    residuals : pd.DataFrame
        DataFrame containing CWRES

    Returns
    -------
    alt.Chart
        Plot
    """

    if 'CWRES' not in residuals.columns:
        raise ValueError("CWRES not available in residuals")
    di = model.datainfo
    idv = di.idv_column.name
    idv_unit = di.idv_column.unit
    dv_unit = di.dv_column.unit
    idname = di.id_column.name

    df = residuals[['CWRES']].reset_index()

    if dv_unit == 1:
        cwres_unit_string = ""
    else:
        cwres_unit_string = f" ({unit_string(dv_unit)})"

    if idv_unit == 1:
        time_unit_string = ""
    else:
        time_unit_string = f" ({unit_string(idv_unit)})"

    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X(idv).title(f"Time{time_unit_string}"),
            y=alt.Y('CWRES').title(f"CWRES{cwres_unit_string}"),
            detail='ID:N',
            tooltip=['CWRES', idv, idname],
        )
        .properties(title="Conditional weighted residuals vs. time", width=600, height=300)
        .interactive()
    )

    zero_line = alt.Chart().mark_rule().encode(y=alt.datum(0), color=alt.value("#000000"))
    loess_smooth = (
        chart.transform_loess(idv, 'CWRES').mark_line().encode(color=alt.value("#FF0000"))
    )

    layer = chart + loess_smooth + zero_line
    layer = layer.configure_point(size=60)

    return layer
