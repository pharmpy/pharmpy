from typing import Optional, Union

from pharmpy.deps import pandas as pd
from pharmpy.model import Model
from pharmpy.modeling import (
    add_iiv,
    fix_parameters_to,
    set_baseline_effect,
    set_initial_estimates,
    set_name,
)

from ..mfl.parse import ModelFeatures
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
    search_space: Union[str, ModelFeatures],
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
    search_space : Union[str, ModelFeatures]
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
    if isinstance(search_space, str):
        mfl_statements = mfl_parse(search_space, True)
    else:
        mfl_statements = search_space
    functions = mfl_statements.convert_to_funcs(model=model, subset_features="pd")

    models = []
    for index, (key, func) in enumerate(functions.items(), 1):
        pkpd_model = func(model)
        description = '_'.join(key)
        pkpd_model = pkpd_model.replace(description=description)

        pkpd_model = set_name(pkpd_model, f"structsearch_run{index}")

        # Initial values
        if 'INDIRECTEFFECT' in key and 'DEGRADATION' in key:
            cur_emax_init = (1.0 / (emax_init + 1.0)) - 1.0
        else:
            cur_emax_init = emax_init

        if b_init is not None:
            pkpd_model = set_initial_estimates(pkpd_model, {'POP_B': b_init}, strict=False)
        if ests is not None:
            pkpd_model = fix_parameters_to(pkpd_model, ests, strict=False)
        if emax_init is not None:
            pkpd_model = set_initial_estimates(
                pkpd_model, {'POP_E_MAX': cur_emax_init}, strict=False
            )
        if ec50_init is not None:
            pkpd_model = set_initial_estimates(pkpd_model, {'POP_EC_50': ec50_init}, strict=False)
        if emax_init is not None and ec50_init is not None:
            pkpd_model = set_initial_estimates(
                pkpd_model, {'POP_SLOPE': cur_emax_init / ec50_init}, strict=False
            )
        if met_init is not None:
            pkpd_model = set_initial_estimates(pkpd_model, {'POP_MET': met_init}, strict=False)

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

        models.append(pkpd_model)

    return models


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
