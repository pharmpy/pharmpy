from typing import Optional, Union

from pharmpy.deps import pandas as pd
from pharmpy.model import Model
from pharmpy.modeling import (
    add_iiv,
    fix_parameters_to,
    set_baseline_effect,
    set_initial_estimates,
    set_lower_bounds,
    set_name,
    unconstrain_parameters,
)
from pharmpy.tools.mfl.helpers import funcs, structsearch_pd_features

from ..mfl.parse import parse as mfl_parse


def create_baseline_pd_model(model: Model, ests: pd.Series, b_init: Optional[float] = None):
    """Create baseline pkpd model

    Parameters
    ----------
    model : Model
        Pharmpy PK model
    ests : pd.Series
       List of estimated PK parameters
    b_init : float
        Initial estimate for baseline

    Returns
    -------
    Baseline PD model
    """
    baseline_model = set_baseline_effect(model, expr='const')
    baseline_model = set_name(baseline_model, "baseline_model")
    baseline_model = baseline_model.replace(description="baseline_model")
    baseline_model = fix_parameters_to(baseline_model, ests)
    baseline_model = add_iiv(baseline_model, ["B"], "exp")
    if b_init is not None:
        baseline_model = set_initial_estimates(baseline_model, {'POP_B': b_init})
    return baseline_model


def create_pkpd_models(
    model: Model,
    search_space: str,
    b_init: Optional[Union[int, float]] = None,
    ests: Optional[pd.Series] = None,
    emax_init: Optional[Union[int, float]] = None,
    ec50_init: Optional[Union[int, float]] = None,
    met_init: Optional[Union[int, float]] = None,
):
    """Create pkpd models

    Parameters
    ----------
    model : Model
        Pharmpy PK model
    search_space : str
        search space for pkpd models
    b_init : float
       Initial estimate for baseline
    ests : pd.Series
       List of estimated PK parameters
    emax_init : float
        Initial estimate for E_max
    ec50_init : float
        Initial estimate for EC_50
    met_init : float
        Initial estimate for MET (mean equilibration time)

    Returns
    -------
    List of pharmpy models
    """
    mfl_statements = mfl_parse(search_space)
    functions = funcs(model, mfl_statements, structsearch_pd_features)

    models = []
    for key, func in functions.items():
        pkpd_model = func(model)
        description = '_'.join(key)
        pkpd_model = pkpd_model.replace(description=description)
        models.append(pkpd_model)

    final_models = []
    for index, pkpd_model in enumerate(models, 1):
        pkpd_model = set_name(pkpd_model, f"structsearch_run{index}")

        # Initial values
        if b_init is not None:
            pkpd_model = set_initial_estimates(pkpd_model, {'POP_B': b_init})
        if ests is not None:
            pkpd_model = fix_parameters_to(pkpd_model, ests)
        if emax_init is not None:
            pkpd_model = set_initial_estimates(pkpd_model, {'POP_E_MAX': emax_init})
        if ec50_init is not None:
            pkpd_model = set_initial_estimates(pkpd_model, {'POP_EC_50': ec50_init})
        if emax_init is not None and ec50_init is not None:
            pkpd_model = set_initial_estimates(pkpd_model, {'POP_SLOPE': emax_init / ec50_init})
        if met_init is not None:
            pkpd_model = set_initial_estimates(pkpd_model, {'POP_MET': met_init})

        pkpd_model = unconstrain_parameters(pkpd_model, ['POP_SLOPE'])
        pkpd_model = set_lower_bounds(pkpd_model, {'POP_E_MAX': -1.0})

        # Set iiv
        for parameter in ["E_MAX", "SLOPE"]:
            try:
                pkpd_model = add_iiv(pkpd_model, [parameter], "prop", initial_estimate=0.1)
            except ValueError:
                pass
        try:
            pkpd_model = add_iiv(pkpd_model, 'B', "exp", initial_estimate=0.1)
        except ValueError:
            pass

        final_models.append(pkpd_model)

    return final_models


def create_pk_model(model: Model):
    # Create copy of model with filtered dataset.
    # FIXME: This function needs to be removed later
    pk_dataset = model.dataset
    pk_dataset = pk_dataset[pk_dataset["DVID"] != 2]
    pk_model = model.replace(
        dataset=pk_dataset,
        description=model.description + ". Removed rows with DVID=2.",
        name="PK_" + model.name,
    )
    return pk_model
