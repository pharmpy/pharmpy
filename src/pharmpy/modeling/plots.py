from __future__ import annotations

import re
import warnings
from functools import partial
from pathlib import Path
from typing import Literal, Mapping, Optional, Union

import pharmpy.visualization
from pharmpy.basic import Expr
from pharmpy.deps import altair as alt
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import scipy, sympy, sympy_stats
from pharmpy.model import Assignment, Model, get_and_check_dataset
from pharmpy.modeling import bin_observations

from .data import get_observations

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
    model: Model, predictions: pd.DataFrame, individuals: Optional[list[int]] = None
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
    obs = get_observations(model, keep_index=True)
    idcol = model.datainfo.id_column.name
    idvcol = model.datainfo.idv_column.name
    columns = (idcol, idvcol, model.datainfo.dv_column.name)
    dataset = get_and_check_dataset(model)
    df = dataset.loc[obs.index, columns]

    data = df.join(predictions)
    data = data.melt(id_vars=(idcol, idvcol))

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
    parameter_estimates: Union[pd.Series, Mapping[str, float]],
    individual_estimates: pd.DataFrame,
):
    """Plot transformed eta distributions for all transformed etas

    Parameters
    ----------
    model : Model
        Previously run Pharmpy model.
    parameter_estimates : Union[pd.Series, Mapping[str, float]]
        Parameter estimates of model fit
    individual_estimates : pd.DataFrame
        Individual estimates for etas

    Returns
    -------
    alt.Chart
        Plot
    """
    pe = {Expr.symbol(str(key)): Expr.float(value) for key, value in parameter_estimates.items()}
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
        var = model.random_variables.etas.get_covariance(eta.name, eta.name).subs(pe)
        subdf = pd.DataFrame({'x': x, 'original': norm.pdf(x, scale=float(var) ** 0.5)})
        rv = sympy_stats.Normal('eta', 0, sympy.sqrt(var))
        expr = sympy.sympify(expr).subs(pe).subs({eta: rv})
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


def plot_eta_distributions(
    model: Model,
    individual_estimates: pd.DataFrame,
):
    """Plot eta distributions for all etas

    Parameters
    ----------
    model : Model
        Previously run Pharmpy model.
    individual_estimates : pd.DataFrame
        Individual estimates for etas

    Returns
    -------
    alt.Chart
        Plot

    Examples
    --------

    .. pharmpy-execute::

        from pharmpy.modeling import load_example_model, plot_eta_distributions
        from pharmpy.tools import load_example_modelfit_results

        model = load_example_model("pheno")
        res = load_example_modelfit_results("pheno")
        plot_eta_distributions(model, res.individual_estimates)

    """
    i = 1
    df = pd.DataFrame()
    for eta in model.random_variables.etas.names:
        subdf = pd.DataFrame({'value': individual_estimates[eta]})
        subdf['eta'] = eta
        df = pd.concat([df, subdf])
        i += 1

    df['y'] = 0
    single = (
        alt.Chart(width=400, height=400)
        .transform_filter('isValid(datum.eta)')
        .transform_density(
            'value',
            groupby=['eta'],
            as_=['value', 'density'],
        )
        .mark_area(opacity=0.4)
        .encode(
            x=alt.X('value:Q', axis=alt.Axis(title='Value')),
            y=alt.Y('density:Q', title='Density'),
            color=alt.value('blue'),
        )
    )
    ticks = alt.Chart().mark_tick().encode(x='value', y='y', color=alt.value('black'))
    layer = alt.layer(single, ticks, data=df)
    facet = layer.facet(facet='eta:N', columns=3).interactive()
    return facet


def plot_dv_vs_pred(
    model: Model,
    predictions: pd.DataFrame,
    stratify_on: Optional[str] = None,
    bins: int = 8,
) -> alt.Chart:
    """Plot DV vs PRED

    Parameters
    ----------
    model : Model
        Pharmpy model
    predictions : pd.DataFrame
        DataFrame containing the predictions
    stratify_on : str
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

    return _dv_vs_anypred(model, predictions, pred, pred_name, stratify_on, bins)


def plot_dv_vs_ipred(
    model: Model,
    predictions: pd.DataFrame,
    stratify_on: Optional[str] = None,
    bins: int = 8,
) -> alt.Chart:
    """Plot DV vs IPRED

    Parameters
    ----------
    model : Model
        Pharmpy model
    predictions : pd.DataFrame
        DataFrame containing the predictions
    stratify_on : str
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

    return _dv_vs_anypred(model, predictions, ipred, ipred_name, stratify_on, bins)


def plot_abs_cwres_vs_ipred(
    model: Model,
    predictions: pd.DataFrame,
    residuals: pd.DataFrame,
    stratify_on: Optional[str] = None,
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
    stratify_on : str
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
    idv = model.datainfo.idv_column.name
    idname = model.datainfo.id_column.name

    columns = [idname, idv]
    if stratify_on is not None:
        _validate_strat(model, stratify_on)
        columns.append(stratify_on)
    df = get_and_check_dataset(model)
    df = df.loc[residuals.index, columns]
    df = df.join(residuals[['CWRES']]).join(predictions[[ipred]])

    if stratify_on is not None:
        df = _bin_data(df, model, stratify_on, bins)

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

    if stratify_on is None:
        layer = chart + _smooth(chart, ipred, 'CWRES')
    else:
        chart = chart.properties(height=300, width=300)
        layer = (
            alt.layer(chart, _smooth(chart, ipred, 'CWRES'), data=df)
            .facet(facet=f'{stratify_on}', columns=2)
            .resolve_scale(x='shared', y='shared')
        )
    layer = layer.configure_point(size=60)

    return layer


def _dv_vs_anypred(model, predictions, predcol_name, predcol_descr, stratify_on, bins):
    obs = get_observations(model, keep_index=True)
    di = model.datainfo
    idv = di.idv_column.name
    dvcol = di.dv_column
    dv = dvcol.name
    dv_unit = dvcol.unit
    idname = di.id_column.name
    columns = [idname, idv, dv]
    if stratify_on is not None:
        _validate_strat(model, stratify_on)
        columns.append(stratify_on)
    df = model.dataset.loc[obs.index, columns]
    df = df.join(predictions[[predcol_name]])

    if stratify_on is None:
        title = f"Observations vs. {predcol_descr}s"
    else:
        df = _bin_data(df, model, stratify_on, bins)
        title = f"{stratify_on}"

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
        min([df[predcol_name].min(), df[dv].min()]), max([df[predcol_name].max(), df[dv].max()])
    )
    if stratify_on is not None:
        chart = chart.properties(height=300, width=300)
        layer = (
            alt.layer(chart, _smooth(chart, predcol_name, dv), line, data=df)
            .facet(facet=f'{stratify_on}', columns=2)
            .resolve_scale(x='shared', y='shared')
        )
    else:
        layer = chart + _smooth(chart, predcol_name, dv) + line
    layer = layer.configure_point(size=60)

    return layer


def plot_cwres_vs_idv(
    model: Model,
    residuals: pd.DataFrame,
    stratify_on: Optional[str] = None,
    bins: int = 8,
) -> alt.Chart:
    """Plot CWRES vs idv

    Parameters
    ----------
    model : Model
        Pharmpy model
    residuals : pd.DataFrame
        DataFrame containing CWRES
    stratify_on : str
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

    columns = [idname, idv]
    if stratify_on is not None:
        _validate_strat(model, stratify_on)
        columns.append(stratify_on)
    df = get_and_check_dataset(model)
    df = df.loc[residuals.index, columns]
    df = df.join(residuals[['CWRES']])

    if stratify_on is not None:
        df = _bin_data(df, model, stratify_on, bins)

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

    if stratify_on is None:
        layer = (
            chart + _smooth(chart, idv, 'CWRES') + _vertical_line().transform_calculate(offset="0")
        )
    else:
        chart = chart.properties(height=300, width=300)
        layer = (
            alt.layer(chart, _smooth(chart, idv, 'CWRES'), _vertical_line(), data=df)
            .transform_calculate(offset="0")
            .facet(facet=f'{stratify_on}', columns=2)
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
    # Extend the line
    min_value = (min_value - abs(min_value) * 10,)
    max_value = max_value * 10
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


def _bin_data(df, model, stratify_on, bins):
    df = df.sort_values(by=[f'{stratify_on}'])
    if len(list(df[stratify_on].unique())) > bins:
        bins = np.linspace(df[stratify_on].min(), df[stratify_on].max(), bins + 1)
        unit = model.datainfo[f'{stratify_on}'].unit
        labels = [_title_with_unit(f'{bins[i]} - {bins[i+1]}', unit) for i in range(len(bins) - 1)]
        df[f'{stratify_on}'] = pd.cut(
            df[f'{stratify_on}'], bins=bins, labels=labels, include_lowest=True
        )
    return df


def _validate_strat(model, stratify_on):
    if stratify_on not in model.dataset.columns:
        raise ValueError(f'{stratify_on} column does not exist in dataset.')


def _calculate_vpc(
    model, simulations, binning: str, nbins: int, qi: float, ci: float, query=None, stratify_on=None
):
    dv = model.datainfo.dv_column.name
    nrows = len(model.dataset)
    nsim = int(len(simulations) / nrows)
    observations = get_observations(model, keep_index=True)
    obstab = model.dataset.loc[observations.index]
    if query is not None:
        obstab = obstab.query(query)
    obstab = obstab[dv]

    bincol, boundaries = bin_observations(model, binning, nbins)

    if len(bincol.unique()) != bincol.unique().max() + 1:
        raise ValueError("Some bins are empty, please choose a different number of bins.")

    obsgroup = obstab.groupby(bincol)

    simtab = simulations
    if stratify_on is not None:
        simtab[stratify_on] = np.tile(model.dataset[stratify_on], nsim)
    simtab = simtab.reset_index(['SIM'])
    simtab = simtab.loc[observations.index]
    simtab["__BIN__"] = bincol
    if query is not None:
        simtab = simtab.query(query)
    simser = simtab[[dv, "SIM"]]
    simser.index = simtab["__BIN__"]
    simgroup = simser.groupby("__BIN__")

    lower_quantile = (1 - qi) / 2
    upper_quantile = 1 - lower_quantile

    median_cis = simgroup.apply(
        partial(_calculate_confidence_interval_of_quantile, quantile=0.5, ci=ci, dv=dv),
        include_groups=False,
    )
    sim_central_lower = [lower for lower, _ in median_cis]
    sim_central_upper = [upper for _, upper in median_cis]

    lowerpi_cis = simgroup.apply(
        partial(_calculate_confidence_interval_of_quantile, quantile=lower_quantile, ci=ci, dv=dv),
        include_groups=False,
    )
    sim_lower_lower = [lower for lower, _ in lowerpi_cis]
    sim_lower_upper = [upper for _, upper in lowerpi_cis]

    upperpi_cis = simgroup.apply(
        partial(_calculate_confidence_interval_of_quantile, quantile=upper_quantile, ci=ci, dv=dv),
        include_groups=False,
    )
    sim_upper_lower = [lower for lower, _ in upperpi_cis]
    sim_upper_upper = [upper for _, upper in upperpi_cis]

    midpoints = (boundaries[1::] + boundaries[0:-1]) / 2

    simgroup = simser.drop(columns=["SIM"]).squeeze().groupby("__BIN__")

    df = pd.DataFrame(
        {
            'obs_central': obsgroup.apply(
                partial(_get_quantile, quantile=0.5), include_groups=False
            ),
            'obs_lower': obsgroup.apply(
                partial(_get_quantile, quantile=lower_quantile), include_groups=False
            ),
            'obs_upper': obsgroup.apply(
                partial(_get_quantile, quantile=upper_quantile), include_groups=False
            ),
            'sim_central': simgroup.median(),
            'sim_central_lower': sim_central_lower,
            'sim_central_upper': sim_central_upper,
            'sim_lower': simgroup.apply(
                partial(_get_quantile, quantile=lower_quantile), include_groups=False
            ),
            'sim_lower_lower': sim_lower_lower,
            'sim_lower_upper': sim_lower_upper,
            'sim_upper': simgroup.apply(
                partial(_get_quantile, quantile=upper_quantile), include_groups=False
            ),
            'sim_upper_lower': sim_upper_lower,
            'sim_upper_upper': sim_upper_upper,
            'bin_midpoint': midpoints,
            'bin_edges_right': boundaries[1::],
            'bin_edges_left': boundaries[0:-1],
            'n_data_points': obsgroup.count(),
        }
    )
    return df


def _calculate_confidence_interval_of_quantile(data, quantile, ci, dv):
    data_sim = data.groupby("SIM")
    quantiles = data_sim.apply(
        partial(_get_quantile, quantile=quantile, sort_by=dv), include_groups=False
    )
    sorted_quantiles = quantiles.sort_values().to_list()
    n = len(sorted_quantiles)
    alpha = (1 - ci) / 2
    i = int(np.floor(alpha * (n - 1) + 0.5))
    lower = sorted_quantiles[i]
    upper = sorted_quantiles[n - i - 1]
    return lower, upper


def _get_quantile(data, quantile, sort_by=None):
    n = len(data)
    if sort_by is None:
        sorted_data = data.sort_values().to_list()
    else:
        sorted_data = data[sort_by].sort_values().to_list()
    return sorted_data[int(np.floor(quantile * (n - 1) + 0.5))]


def _vpc_plot(model, simulations, binning, nbins, qi, ci, query=None, title='', stratify_on=None):
    obs = get_observations(model, keep_index=True)
    idv = model.datainfo.idv_column.name
    idname = model.datainfo.id_column.name
    data = model.dataset.loc[obs.index]
    if query is not None:
        data = data.query(query)

    df = _calculate_vpc(
        model,
        simulations,
        binning=binning,
        nbins=nbins,
        qi=qi,
        ci=ci,
        query=query,
        stratify_on=stratify_on,
    )

    scatter = (
        alt.Chart(data)
        .mark_circle(color='blue', filled=False)
        .encode(
            x=alt.X(
                f'{idv}',
                title=f'{idv}',
                scale=alt.Scale(domain=[-1, data[idv].max() * 1.01]),
            ),
            y=alt.Y(
                'DV',
                title='DV',
                scale=alt.Scale(domain=[data['DV'].min() * 0.9, data['DV'].max() * 1.05]),
            ),
            tooltip=[idv, 'DV', idname],
        )
        .properties(width=700, height=500, title=title)
        .interactive()
    )

    obs_mid = alt.Chart(df).mark_line(color='red').encode(x='bin_midpoint', y='obs_central')
    obs_lower = (
        alt.Chart(df)
        .mark_line(color='red', strokeDash=[8, 4])
        .encode(x='bin_midpoint', y='obs_lower')
    )
    obs_upper = (
        alt.Chart(df)
        .mark_line(color='red', strokeDash=[8, 4])
        .encode(x='bin_midpoint', y='obs_upper')
    )

    central_ci = (
        alt.Chart(df)
        .mark_rect(opacity=0.3, color='red')
        .encode(
            x='bin_edges_left', x2='bin_edges_right', y='sim_central_lower', y2='sim_central_upper'
        )
    )
    lower_ci = (
        alt.Chart(df)
        .mark_rect(opacity=0.3, color='blue')
        .encode(x='bin_edges_left', x2='bin_edges_right', y='sim_lower_lower', y2='sim_lower_upper')
    )
    upper_ci = (
        alt.Chart(df)
        .mark_rect(opacity=0.3, color='blue')
        .encode(x='bin_edges_left', x2='bin_edges_right', y='sim_upper_lower', y2='sim_upper_upper')
    )

    chart = scatter + obs_mid + obs_lower + obs_upper + central_ci + upper_ci + lower_ci
    return chart


def plot_vpc(
    model: Model,
    simulations: Union[Path, pd.DataFrame, str],
    binning: Literal["equal_width", "equal_number"] = "equal_number",
    nbins: int = 8,
    qi: float = 0.95,
    ci: float = 0.95,
    stratify_on: Optional[str] = None,
):
    """Creates a VPC plot for a model

    Parameters
    ----------
    model : Model
        Pharmpy model
    simulations : Path or pd.DataFrame
        DataFrame containing the simulation data or path to dataset.
        The dataset has to have one (index) column named "SIM" containing
        the simulation number, one (index) column named "index" containing the data indices and one dv column.
        See below for more information.
    binning : ["equal_number", "equal_width"]
        Binning method. Can be "equal_number" or "equal_width". The default is "equal_number".
    nbins : int
        Number of bins. Default is 8.
    qi : float
        Upper quantile. Default is 0.95.
    ci : float
        Confidence interval. Default is 0.95.
    stratify_on : str
        Parameter to use for stratification. Optional.

    Returns
    -------
    alt.Chart
        Plot


    The simulation data should have the following format:

    +-----+-------+--------+
    | SIM | index | DV     |
    +=====+=======+========+
    | 1   | 0     | 0.000  |
    +-----+-------+--------+
    | 1   | 1     | 34.080 |
    +-----+-------+--------+
    | 1   | 2     | 28.858 |
    +-----+-------+--------+
    | 1   | 3     | 0.000  |
    +-----+-------+--------+
    | 1   | 4     | 12.157 |
    +-----+-------+--------+
    | 2   | 0     | 23.834 |
    +-----+-------+--------+
    | 2   | 1     | 0.000  |
    +-----+-------+--------+
    | ... | ...   | ...    |
    +-----+-------+--------+
    | 20  | 2     | 0.000  |
    +-----+-------+--------+
    | 20  | 3     | 31.342 |
    +-----+-------+--------+
    | 20  | 4     | 29.983 |
    +-----+-------+--------+


    Examples
    --------
    >>> from pharmpy.modeling import set_simulation, plot_vpc, load_example_model
    >>> from pharmpy.tools import run_simulation
    >>> model = load_example_model("pheno")
    >>> sim_model = set_simulation(model, n=100)
    >>> sim_data = run_simulation(sim_model) # doctest: +SKIP
    >>> plot_vpc(model, sim_data) # doctest: +SKIP
    """
    if isinstance(simulations, str) or isinstance(simulations, Path):
        simulations = pd.read_table(simulations, delimiter=r'\s+|,', engine='python')
        if 'SIM' not in simulations.columns:
            raise ValueError('No column named "SIM" found in dataset.')
        if 'index' not in simulations.columns:
            raise ValueError('No column named "index" found in dataset.')
        simulations = simulations.set_index(['SIM', 'index'])

    if stratify_on is not None:
        df = get_and_check_dataset(model)
        if f'{stratify_on}' not in df.columns:
            raise ValueError(f'{stratify_on} column does not exist in dataset.')
        charts = []
        unique_values = df[f'{stratify_on}'].unique()
        n_unique = len(unique_values)
        if n_unique > 8:
            bin_stratification = np.linspace(df[stratify_on].min(), df[stratify_on].max(), 9)
            for i in range(len(bin_stratification) - 1):
                query = f'{stratify_on} >= {bin_stratification[i] and {stratify_on} < {bin_stratification[i+1]}}'
                charts.append(
                    _vpc_plot(
                        model,
                        simulations,
                        binning=binning,
                        nbins=nbins,
                        qi=qi,
                        ci=ci,
                        query=query,
                        title=f'{stratify_on} {bin_stratification[i]} - {bin_stratification[i+1]}',
                        stratify_on=stratify_on,
                    )
                )
        else:
            for value in unique_values:
                query = f'{stratify_on} == {value}'
                charts.append(
                    _vpc_plot(
                        model,
                        simulations,
                        binning=binning,
                        nbins=nbins,
                        qi=qi,
                        ci=ci,
                        query=query,
                        title=f'{stratify_on} {value}',
                        stratify_on=stratify_on,
                    )
                )
        chart = _concat(charts)
    else:
        chart = _vpc_plot(model, simulations, binning=binning, nbins=nbins, qi=qi, ci=ci)

    return chart


def _concat(charts):
    # Concatenate charts up to 2x4
    n = len(charts)
    if n == 1:
        return charts[0]
    else:
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
    return chart
