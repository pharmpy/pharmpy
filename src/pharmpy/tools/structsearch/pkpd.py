from pharmpy.deps import pandas as pd
from pharmpy.model import Model
from pharmpy.modeling import (
    add_effect_compartment,
    add_iiv,
    fix_parameters_to,
    set_direct_effect,
    set_initial_estimates,
    set_name,
)


def create_baseline_pd_model(model: Model, ests: pd.Series):
    """Create baseline pkpd model

    Parameters
    ----------
    model : Model
        Pharmpy PK model
    ests : pd.Series
       List of estimated PK parameters

    Returns
    -------
    Baseline PD model
    """
    baseline_model = set_direct_effect(model, expr='baseline')
    baseline_model = set_name(baseline_model, "baseline_pd_model")
    baseline_model = baseline_model.replace(description="direct_effect_baseline")
    baseline_model = fix_parameters_to(baseline_model, ests)
    baseline_model = add_iiv(baseline_model, ["E0"], "exp")
    return baseline_model


def create_pkpd_models(model: Model, e0_init: pd.Series, ests: pd.Series):
    """Create pkpd models

    Parameters
    ----------
    model : Model
        Pharmpy PK model
    ests : pd.Series
       List of estimated PK parameters

    Returns
    -------
    List of pharmpy models
    """
    models = []
    index = 1
    pd_types = ["linear", "Emax", "sigmoid", "step"]
    for model_type in ["direct_effect", "effect_compartment"]:
        for pd_type in pd_types:
            if model_type == "direct_effect":
                pkpd_model = set_direct_effect(model, expr=pd_type)
            elif model_type == "effect_compartment":
                pkpd_model = add_effect_compartment(model, expr=pd_type)
            pkpd_model = set_name(pkpd_model, f"structsearch_run{index}")
            pkpd_model = pkpd_model.replace(description=f"{model_type}_{pd_type}")
            index += 1
            pkpd_model = add_iiv(pkpd_model, ["E0"], "exp")
            pkpd_model = set_initial_estimates(pkpd_model, e0_init)
            pkpd_model = fix_parameters_to(pkpd_model, ests)
            for parameter in ["Slope", "E_max"]:
                try:
                    pkpd_model = add_iiv(pkpd_model, [parameter], "exp")
                except ValueError:
                    pass
            models.append(pkpd_model)
    return models


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
