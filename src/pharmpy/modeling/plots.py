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


def plot_dv_vs_pred(
    model: Model,
    predictions: pd.DataFrame,
    stratify_by: str = None,
    bins: int = 8,
) -> alt.Chart:
    """Plot DV vs PRED

    Parameters
    ----------
    model : Model
        Pharmpy model
    predictions : pd.DataFrame
        DataFrame containing the predictions
    stratify_by : str
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

        from pharmpy.modeling import load_example_model, plot_dv_vs_pred
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_dv_vs_pred(model, res.predictions)

    .. pharmpy-execute::

        from pharmpy.modeling import load_example_model, plot_dv_vs_ipred
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_dv_vs_pred(model, res.predictions, 'WGT', bins=4)

    """
    if 'PRED' in predictions.columns:
        pred = 'PRED'
    else:
        raise ValueError("Cannot find population predictions")
    pred_name = "Population prediction"

    return _dv_vs_anypred(model, predictions, pred, pred_name, stratify_by, bins)


def plot_dv_vs_ipred(
    model: Model,
    predictions: pd.DataFrame,
    stratify_by: str = None,
    bins: int = 8,
) -> alt.Chart:
    """Plot DV vs IPRED

    Parameters
    ----------
    model : Model
        Pharmpy model
    predictions : pd.DataFrame
        DataFrame containing the predictions
    stratify_by : str
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
        plot_dv_vs_ipred(model, res.predictions, 'WGT', bins=4)

    """
    if 'CIPREDI' in predictions.columns:
        ipred = 'CIPREDI'
    elif 'IPRED' in predictions.columns:
        ipred = 'IPRED'
    else:
        raise ValueError("Cannot find individual predictions")
    ipred_name = "Individual prediction"

    return _dv_vs_anypred(model, predictions, ipred, ipred_name, stratify_by, bins)


def plot_abs_cwres_vs_ipred(
    model: Model,
    predictions: pd.DataFrame,
    residuals: pd.DataFrame,
    stratify_by: str = None,
    bins: int = 8,
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
    stratify_by : str
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

        from pharmpy.modeling import load_example_model, plot_abs_cwres_vs_ipred
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_abs_cwres_vs_ipred(model, res.predictions, res.residuals)

    .. pharmpy-execute::

        from pharmpy.modeling import load_example_model, plot_abs_cwres_vs_ipred
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_abs_cwres_vs_ipred(model, res.predictions, res.residuals, 'WGT', bins=4)

    """
    if 'CIPREDI' in predictions.columns:
        ipred = 'CIPREDI'
    elif 'IPRED' in predictions.columns:
        ipred = 'IPRED'
    else:
        raise ValueError("Cannot find individual predictions")
    if 'CWRES' not in residuals.columns:
        raise ValueError("CWRES not available in residuals")

    if stratify_by is not None:
        _validate_strat(model, stratify_by)
        idv_name = model.datainfo.idv_column.name
        id_name = model.datainfo.id_column.name
        if (
            f'{stratify_by}' not in predictions.columns
            and f'{stratify_by}' not in predictions.index.names
        ):
            newcol = model.dataset[[id_name, idv_name, f'{stratify_by}']].set_index(
                [id_name, idv_name]
            )
            predictions = predictions.join(newcol, how='inner')
            predictions = predictions[[ipred, f'{stratify_by}']].reset_index()
        else:
            predictions = predictions[[ipred]].reset_index()
    else:
        predictions = predictions[[ipred]].reset_index()
    observations = get_observations(model, keep_index=True)
    residuals = abs(residuals[['CWRES']]).set_index(observations.index)
    df = predictions.join(residuals, how='inner').reset_index(drop=True)

    if stratify_by is not None:
        df = _bin_data(df, model, stratify_by, bins)

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

    if stratify_by is None:
        layer = chart + _smooth(chart, ipred, 'CWRES')
    else:
        chart = chart.properties(height=300, width=300)
        layer = (
            alt.layer(chart, _smooth(chart, ipred, 'CWRES'), data=df)
            .facet(facet=f'{stratify_by}', columns=2)
            .resolve_scale(x='shared', y='shared')
        )
    layer = layer.configure_point(size=60)

    return layer


def _dv_vs_anypred(model, predictions, predcol_name, predcol_descr, stratify_by, bins):
    obs = get_observations(model)
    di = model.datainfo
    idv = di.idv_column.name
    dvcol = di.dv_column
    dv = dvcol.name
    dv_unit = dvcol.unit
    idname = di.id_column.name

    if stratify_by is None:
        predictions = predictions[[predcol_name]]
        df = predictions.join(obs, how='inner').reset_index()
        title = f"Observations vs. {predcol_descr}s"
    else:
        _validate_strat(model, stratify_by)
        idv_name = model.datainfo.idv_column.name
        id_name = model.datainfo.id_column.name
        if (
            f'{stratify_by}' not in predictions.columns
            and f'{stratify_by}' not in predictions.index.names
        ):
            newcol = model.dataset[[id_name, idv_name, f'{stratify_by}']].set_index(
                [id_name, idv_name]
            )
            predictions = predictions.join(newcol, how='inner')
            predictions = predictions[[predcol_name, f'{stratify_by}']]
        else:
            predictions = predictions[[predcol_name]]
        df = predictions.join(obs, how='inner').reset_index()
        df = _bin_data(df, model, stratify_by, bins)
        title = f"{stratify_by}"

    chart = _grouped_scatter(
        df,
        x=predcol_name,
        y=dv,
        group=idname,
        title=title,
        xtitle=predcol_descr,
        ytitle="Observation",
        xunit=dv_unit,
        yunit=dv_unit,
        tooltip=(idv,),
    )
    line = _identity_line(
        (min_value := min([df[predcol_name].min(), df[dv].min()])) - abs(min_value) * 10,
        max([df[predcol_name].max(), df[dv].max()]) * 10,
    )
    if stratify_by is not None:
        chart = chart.properties(height=300, width=300)
        layer = (
            alt.layer(chart, _smooth(chart, predcol_name, dv), line, data=df)
            .facet(facet=f'{stratify_by}', columns=2)
            .resolve_scale(x='shared', y='shared')
        )
    else:
        layer = chart + _smooth(chart, predcol_name, dv) + line
    layer = layer.configure_point(size=60)

    return layer


def plot_cwres_vs_idv(
    model: Model,
    residuals: pd.DataFrame,
    stratify_by: str = None,
    bins: int = 8,
) -> alt.Chart:
    """Plot CWRES vs idv

    Parameters
    ----------
    model : Model
        Pharmpy model
    residuals : pd.DataFrame
        DataFrame containing CWRES
    stratify_by : str
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

        from pharmpy.modeling import load_example_model, plot_cwres_vs_idv
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_cwres_vs_idv(model, res.residuals)

    .. pharmpy-execute::

        from pharmpy.modeling import load_example_model, plot_cwres_vs_idv
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_cwres_vs_idv(model, res.residuals, 'WGT', bins=4)

    """
    if 'CWRES' not in residuals.columns:
        raise ValueError("CWRES not available in residuals")
    di = model.datainfo
    idv = di.idv_column.name
    idv_unit = di.idv_column.unit
    dv_unit = di.dv_column.unit
    idname = di.id_column.name

    if stratify_by is not None:
        _validate_strat(model, stratify_by)
        idv_name = model.datainfo.idv_column.name
        id_name = model.datainfo.id_column.name
        if (
            f'{stratify_by}' not in residuals.columns
            and f'{stratify_by}' not in residuals.index.names
        ):
            newcol = model.dataset[[id_name, idv_name, f'{stratify_by}']].set_index(
                [id_name, idv_name]
            )
            residuals = residuals.join(newcol, how='inner')
            df = residuals[['CWRES', f'{stratify_by}']].reset_index()
    else:
        df = residuals[['CWRES']].reset_index()

    if stratify_by is not None:
        df = _bin_data(df, model, stratify_by, bins)

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

    if stratify_by is None:
        layer = (
            chart + _smooth(chart, idv, 'CWRES') + _vertical_line().transform_calculate(offset="0")
        )
    else:
        chart = chart.properties(height=300, width=300)
        layer = (
            alt.layer(chart, _smooth(chart, idv, 'CWRES'), _vertical_line(), data=df)
            .transform_calculate(offset="0")
            .facet(facet=f'{stratify_by}', columns=2)
            .resolve_scale(x='shared', y='shared')
        )
    layer = layer.configure_point(size=60)

    return layer


def _smooth(chart, x, y):
    loess_smooth = chart.transform_loess(x, y).mark_line().encode(color=alt.value("#FF0000"))
    return loess_smooth


def _vertical_line():
    line = alt.Chart().mark_rule().encode(y="offset:Q", color=alt.value("#000000"))
    return line


def _identity_line(min_value, max_value):
    x1, x2 = alt.param(value=min_value), alt.param(value=max_value)
    y1, y2 = alt.param(value=min_value), alt.param(value=max_value)
    line = (
        alt.Chart()
        .mark_rule()
        .encode(
            x=alt.datum(x1, type="quantitative"),
            x2=alt.datum(x2, type="quantitative"),
            y=alt.datum(y1, type="quantitative"),
            y2=alt.datum(y2, type="quantitative"),
            color=alt.value("#000000"),
            strokeWidth=alt.value(1),
        )
        .add_params(x1, x2, y1, y2)
    )
    return line


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


def _bin_data(df, model, stratify_by, bins):
    df = df.sort_values(by=[f'{stratify_by}'])
    if len(list(df[stratify_by].unique())) > bins:
        bins = np.linspace(df[stratify_by].min(), df[stratify_by].max(), bins + 1)
        unit = model.datainfo[f'{stratify_by}'].unit
        labels = [_title_with_unit(f'{bins[i]} - {bins[i+1]}', unit) for i in range(len(bins) - 1)]
        df[f'{stratify_by}'] = pd.cut(
            df[f'{stratify_by}'], bins=bins, labels=labels, include_lowest=True
        )
    return df


def _validate_strat(model, stratify_by):
    if f'{stratify_by}' not in model.dataset.columns:
        raise ValueError(f'{stratify_by} column does not exist in dataset.')
