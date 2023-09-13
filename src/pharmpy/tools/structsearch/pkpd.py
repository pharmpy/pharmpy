from itertools import product
from typing import List, Optional, Union

from pharmpy.deps import pandas as pd
from pharmpy.model import Model
from pharmpy.modeling import (
    add_effect_compartment,
    add_iiv,
    add_indirect_effect,
    fix_parameters_to,
    set_direct_effect,
    set_initial_estimates,
    set_lower_bounds,
    set_name,
    unconstrain_parameters,
)


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
    baseline_model = set_direct_effect(model, expr='baseline')
    baseline_model = set_name(baseline_model, "baseline_model")
    baseline_model = baseline_model.replace(parent_model='baseline_model')
    baseline_model = baseline_model.replace(description="baseline_model")
    baseline_model = fix_parameters_to(baseline_model, ests)
    baseline_model = add_iiv(baseline_model, ["B"], "exp")
    if b_init is not None:
        baseline_model = set_initial_estimates(baseline_model, {'POP_B': b_init})
    return baseline_model


def create_pkpd_models(
    model: Model,
    b_init: Optional[Union[int, float]] = None,
    ests: pd.Series = None,
    emax_init: Optional[Union[int, float]] = None,
    ec50_init: Optional[Union[int, float]] = None,
    mat_init: Optional[Union[int, float]] = None,
    response_type: Union[str, List] = 'all',
):
    """Create pkpd models

    Parameters
    ----------
    model : Model
        Pharmpy PK model
    b_init : float
       Initial estimate for baseline
    ests : pd.Series
       List of estimated PK parameters
    emax_init : float
        Initial estimate for E_max
    ec50_init : float
        Initial estimate for EC_50
    mat_init : float
        Initial estimate for MAT (mean equilibration time)
    response_type : str
        type of model (e.g. direct effect)

    Returns
    -------
    List of pharmpy models
    """

    if response_type == 'all':
        models_list = ['direct', 'effect_compartment', 'indirect']
    elif isinstance(response_type, str):
        models_list = [response_type]
    elif isinstance(response_type, list):
        models_list = response_type

    pd_types = ["linear", "Emax", "sigmoid"]

    models = []
    for modeltype in models_list:
        if modeltype in ['direct', 'effect_compartment']:
            for pd_type in pd_types:
                pkpd_model = _create_model(model, modeltype, expr=pd_type)
                pkpd_model = pkpd_model.replace(description=f"{modeltype}_{pd_type}")
                models.append(pkpd_model)
        elif modeltype == 'indirect':
            for argument in list(product(pd_types, [True, False])):
                pkpd_model = add_indirect_effect(model, *argument)
                pkpd_model = pkpd_model.replace(
                    description=f"{modeltype}_{argument[0]}_prod={argument[1]}"
                )
                models.append(pkpd_model)

        final_models = []
        index = 1
        for pkpd_model in models:
            pkpd_model = set_name(pkpd_model, f"structsearch_run{index}")
            index += 1

            # initial values
            if b_init is not None:
                pkpd_model = set_initial_estimates(pkpd_model, b_init)
            if ests is not None:
                pkpd_model = fix_parameters_to(pkpd_model, ests)

            if emax_init is not None and ec50_init is not None:
                pkpd_model = set_initial_estimates(pkpd_model, {'POP_E_MAX': emax_init})
                pkpd_model = set_initial_estimates(pkpd_model, {'POP_EC_50': ec50_init})
                pkpd_model = set_initial_estimates(pkpd_model, {'POP_SLOPE': emax_init / ec50_init})
            if mat_init is not None:
                pkpd_model = set_initial_estimates(
                    pkpd_model, {'POP_KE0': 1 / mat_init, 'POP_K_OUT': 1 / mat_init}
                )

            pkpd_model = unconstrain_parameters(pkpd_model, ['SLOPE'])
            pkpd_model = set_lower_bounds(pkpd_model, {'POP_E_MAX': -1})

            # set iiv
            for parameter in ["B", "SLOPE"]:
                try:
                    pkpd_model = add_iiv(pkpd_model, [parameter], "exp")
                except ValueError:
                    pass
            try:
                pkpd_model = add_iiv(pkpd_model, ['E_MAX'], "prop")
            except ValueError:
                pass

            pkpd_model = pkpd_model.replace(parent_model='baseline_model')
            final_models.append(pkpd_model)

    return final_models


def _create_model(model, modeltype, expr):
    if modeltype == 'direct':
        model = set_direct_effect(model, expr)
    elif modeltype == 'effect_compartment':
        model = add_effect_compartment(model, expr)
    return model


def create_pk_model(model: Model):
    # Create copy of model with filtered dataset.
    # FIXME: this function needs to be removed later
    pk_dataset = model.dataset
    pk_dataset = pk_dataset[pk_dataset["DVID"] != 2]
    pk_model = model.replace(
        dataset=pk_dataset,
        description=model.description + ". Removed rows with DVID=2.",
        name="PK_" + model.name,
    )
    return pk_model
