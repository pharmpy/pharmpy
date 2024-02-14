import re
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import pharmpy.visualization
from pharmpy.basic import Expr
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

    Examples
    --------

    .. pharmpy-execute::

        from pharmpy.modeling import plot_iofv_vs_iofv
        from pharmpy.tools import load_example_modelfit_results

        res1 = load_example_modelfit_results("pheno")
        res2 = load_example_modelfit_results("pheno_linear")
        plot_iofv_vs_iofv(res1.individual_ofv, res2.individual_ofv, "nonlin", "linear")

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


    Examples
    --------

    .. pharmpy-execute::

        from pharmpy.modeling import load_example_model, plot_individual_predictions
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_individual_predictions(model, res.predictions, individuals=[1, 2, 3, 4, 5])

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
    eta_symbols = {Expr.symbol(name) for name in model.random_variables.etas.names}

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
        expr = sympy.sympify(expr).subs(parameter_estimates).subs({eta: rv})
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


def plot_dv_vs_pred(model: Model, predictions: pd.DataFrame) -> alt.Chart:
    """Plot DV vs PRED

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

    Examples
    --------

    .. pharmpy-execute::

        from pharmpy.modeling import load_example_model, plot_dv_vs_pred
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_dv_vs_pred(model, res.predictions)

    """
    if 'PRED' in predictions.columns:
        pred = 'PRED'
    else:
        raise ValueError("Cannot find population predictions")
    pred_name = "Population prediction"
    return _dv_vs_anypred(model, predictions, pred, pred_name)


def plot_dv_vs_ipred(
    model: Model, predictions: pd.DataFrame, strat: str = None, bins: int = 8
) -> alt.Chart:
    """Plot DV vs IPRED

    Parameters
    ----------
    model : Model
        Pharmpy model
    predictions : pd.DataFrame
        DataFrame containing the predictions
    strat : str
        Name of parameter for stratification
    bins : int
        Number of bins for stratification

    Returns
    -------
    alt.Chart
        Plot

    Examples
    --------

    .. pharmpy-execute::

        from pharmpy.modeling import load_example_model, plot_dv_vs_ipred
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_dv_vs_ipred(model, res.predictions)

    .. pharmpy-execute::

        from pharmpy.modeling import load_example_model, plot_dv_vs_ipred
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_dv_vs_ipred(model, res.predictions)

        plot_dv_vs_ipred(model, res.predictions, 'WGT')

    """
    if 'CIPREDI' in predictions.columns:
        ipred = 'CIPREDI'
    elif 'IPRED' in predictions.columns:
        ipred = 'IPRED'
    else:
        raise ValueError("Cannot find individual predictions")
    ipred_name = "Individual prediction"

    if strat is not None:
        return _dv_vs_anypred_stratify(model, predictions, ipred, ipred_name, strat, bins)
    else:
        return _dv_vs_anypred(model, predictions, ipred, ipred_name)


def plot_abs_cwres_vs_ipred(
    model: Model, predictions: pd.DataFrame, residuals: pd.DataFrame
) -> alt.Chart:
    r"""Plot \|CWRES\| vs IPRED

    Parameters
    ----------
    model : Model
        Pharmpy model
    predictions : pd.DataFrame
        DataFrame containing the predictions
    residuals : pd.DataFrame
        DataFrame containing the residuals

    Returns
    -------
    alt.Chart
        Plot

    Examples
    --------

    .. pharmpy-execute::

        from pharmpy.modeling import load_example_model, plot_abs_cwres_vs_ipred
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_abs_cwres_vs_ipred(model, res.predictions, res.residuals)

    """
    if 'CIPREDI' in predictions.columns:
        ipred = 'CIPREDI'
    elif 'IPRED' in predictions.columns:
        ipred = 'IPRED'
    else:
        raise ValueError("Cannot find individual predictions")
    if 'CWRES' not in residuals.columns:
        raise ValueError("CWRES not available in residuals")

    predictions = predictions[[ipred]].reset_index()
    observations = get_observations(model, keep_index=True)
    residuals = abs(residuals[['CWRES']]).set_index(observations.index)
    df = predictions.join(residuals, how='inner').reset_index(drop=True)

    idv = model.datainfo.idv_column.name
    idname = model.datainfo.id_column.name
    dv_unit = model.datainfo.dv_column.unit

    chart = _scatter(
        df,
        x=ipred,
        y='CWRES',
        title="Conditional weighted resuals vs Individual predictions",
        xtitle="Individual predictions",
        ytitle="|Conditional weighted residuals|",
        xunit=dv_unit,
        yunit=dv_unit,
        tooltip=(idname, idv),
    )

    layer = chart + _smooth(chart, ipred, 'CWRES')
    layer = layer.configure_point(size=60)

    return layer


def _dv_vs_anypred(model, predictions, predcol_name, predcol_descr):
    obs = get_observations(model)
    di = model.datainfo
    idv = di.idv_column.name
    dvcol = di.dv_column
    dv = dvcol.name
    dv_unit = dvcol.unit
    idname = di.id_column.name

    predictions = predictions[[predcol_name]]
    df = predictions.join(obs, how='inner').reset_index()

    chart = _grouped_scatter(
        df,
        x=predcol_name,
        y=dv,
        group=idname,
        title=f"Observations vs. {predcol_descr}s",
        xtitle=predcol_descr,
        ytitle="Observation",
        xunit=dv_unit,
        yunit=dv_unit,
        tooltip=(idv,),
    )

    line = _identity_line(
        df[predcol_name].min(), df[predcol_name].max(), df[dv].min(), df[dv].max()
    )

    layer = chart + _smooth(chart, predcol_name, dv) + line
    layer = layer.configure_point(size=60)

    return layer


def _dv_vs_anypred_stratify(model, predictions, predcol_name, predcol_descr, strat=None, bins=8):
    obs = get_observations(model)
    di = model.datainfo
    idv = di.idv_column.name
    dvcol = di.dv_column
    dv = dvcol.name
    dv_unit = dvcol.unit
    idname = di.id_column.name

    if f'{strat}' not in model.dataset.columns:
        raise ValueError(f'{strat} column does not exist in dataset.')
    if bins > 8:
        raise ValueError('bins must be 8 or less.')

    if f'{strat}' not in predictions.columns and f'{strat}' not in predictions.index.names:
        newcol = model.dataset[['ID', 'TIME', f'{strat}']].set_index(['ID', 'TIME'])
        predictions = predictions.join(newcol, how='inner')
        predictions = predictions[[predcol_name, f'{strat}']]
    else:
        predictions = predictions[[predcol_name]]
    df = predictions.join(obs, how='inner').reset_index()

    df = df.sort_values(by=[f'{strat}'])
    # bin data if more than bins values
    if len(list(df[strat].unique())) > bins:
        bins = np.linspace(df[strat].min(), df[strat].max(), bins + 1)
        unit = di[f'{strat}'].unit
        labels = [_title_with_unit(f'{bins[i]} - {bins[i+1]}', unit) for i in range(len(bins) - 1)]
        df[f'{strat}'] = pd.cut(df[f'{strat}'], bins=bins, labels=labels, include_lowest=True)

    layers = []
    for j in list(df[strat].unique()):
        df_filtered = df.query(f'{strat} == @j')
        chart = _grouped_scatter(
            df_filtered,
            x=predcol_name,
            y=dv,
            group=idname,
            title=f"{strat} = {j}",
            xtitle=predcol_descr,
            ytitle="Observation",
            xunit=dv_unit,
            yunit=dv_unit,
            tooltip=(idv,),
        )

        line = _identity_line(
            df_filtered[predcol_name].min(),
            df_filtered[predcol_name].max(),
            df_filtered[dv].min(),
            df_filtered[dv].max(),
        )

        layer = chart + _smooth(chart, predcol_name, dv) + line
        layers.append(layer)

    layer = _concat(layers)
    layer = (
        layer.properties(title=f"Observations vs. {predcol_descr}s")
        .configure_point(size=60)
        .configure_title(anchor='middle')
    )

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

    Examples
    --------

    .. pharmpy-execute::

        from pharmpy.modeling import load_example_model, plot_cwres_vs_idv
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_cwres_vs_idv(model, res.residuals)

    """

    if 'CWRES' not in residuals.columns:
        raise ValueError("CWRES not available in residuals")
    di = model.datainfo
    idv = di.idv_column.name
    idv_unit = di.idv_column.unit
    dv_unit = di.dv_column.unit
    idname = di.id_column.name

    df = residuals[['CWRES']].reset_index()

    chart = _grouped_scatter(
        df,
        x=idv,
        y='CWRES',
        group=idname,
        title="Conditional weighted residuals vs. time",
        xtitle="Time",
        ytitle="CWRES",
        xunit=idv_unit,
        yunit=dv_unit,
        tooltip=(),
    )

    layer = chart + _smooth(chart, idv, 'CWRES') + _zero_line()
    layer = layer.configure_point(size=60)

    return layer


def _smooth(chart, x, y):
    loess_smooth = chart.transform_loess(x, y).mark_line().encode(color=alt.value("#FF0000"))
    return loess_smooth


def _identity_line(xmin, xmax, ymin, ymax):
    line = (
        alt.Chart(
            pd.DataFrame(
                {
                    'var1': [xmin, xmax],
                    'var2': [ymin, ymax],
                }
            )
        )
        .mark_line()
        .encode(x=alt.X('var1'), y=alt.Y('var2'), color=alt.value("#000000"))
    )
    return line


def _zero_line():
    zero_line = alt.Chart().mark_rule().encode(y=alt.datum(0), color=alt.value("#000000"))
    return zero_line


def _title_with_unit(title, unit):
    if unit == 1:
        s = ""
    else:
        s = f" ({unit.unicode()})"
    return title + s


def _grouped_scatter(df, x, y, group, title, xtitle, ytitle, xunit, yunit, tooltip=()):
    chart = (
        alt.Chart(df)
        .mark_line(point=True, opacity=0.7)
        .encode(
            x=alt.X(x).title(_title_with_unit(xtitle, xunit)),
            y=alt.Y(y).title(_title_with_unit(ytitle, yunit)),
            detail=f"{group}:N",
            tooltip=[x, y, group] + list(tooltip),
        )
        .properties(title=title, width=600, height=300)
        .interactive()
    )
    return chart


def _scatter(df, x, y, title, xtitle, ytitle, xunit, yunit, tooltip=()):
    chart = (
        alt.Chart(df)
        .mark_point(opacity=0.7)
        .encode(
            x=alt.X(x).title(_title_with_unit(xtitle, xunit)),
            y=alt.Y(y).title(_title_with_unit(ytitle, yunit)),
            tooltip=[x, y] + list(tooltip),
        )
        .properties(title=title, width=600, height=300)
        .interactive()
    )
    return chart


def _concat(charts):
    # Concatenate charts up to 2x4
    n = len(charts)
    if n == 1:
        return charts[0]
    elif n > 1:
        chart = alt.hconcat(charts[0], charts[1])
    if n == 3:
        chart = alt.vconcat(chart, charts[2])
    if n > 3:
        chart_tmp = alt.hconcat(charts[2], charts[3])
        chart = alt.vconcat(chart, chart_tmp)
    if n == 5:
        chart = alt.vconcat(chart, charts[4])
    if n > 5:
        chart_tmp = alt.hconcat(charts[4], charts[5])
        chart = alt.vconcat(chart, chart_tmp)
    if n == 7:
        chart = alt.vconcat(chart, charts[6])
    if n == 8:
        chart_tmp = alt.hconcat(charts[6], charts[7])
        chart = alt.vconcat(chart, chart_tmp)
    if n > 8:
        raise ValueError('No more than 8 subplots allowed.')
        return None
    return chart
