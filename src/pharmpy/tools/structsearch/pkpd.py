from pharmpy.deps import pandas as pd
from pharmpy.model import Model
from pharmpy.modeling import (
    add_effect_compartment,
    add_iiv,
    fix_parameters_to,
    set_direct_effect,
    set_name,
)


def create_pkpd_models(model: Model, ests: pd.Series):
    """Create pkpd models

    Parameters
    ----------
    model_type : str
        Type of PD model. Currently either 'direct_effect' or 'effect_compartment'
    model : Model
        Pharmpy PK model
    ests : pd.Series
       List of estimated PK parameters

    Returns
    -------
    List of pharmpy models
    """
    models = []
    pd_types = ["baseline", "linear", "Emax", "sigmoid", "step", "loglin"]
    for model_type in ["direct_effect", "effect_compartment"]:
        for pd_type in pd_types:
            if model_type == "direct_effect":
                pkpd_model = set_direct_effect(model, expr=pd_type)
            elif model_type == "effect_compartment":
                pkpd_model = add_effect_compartment(model, expr=pd_type)
            pkpd_model = set_name(pkpd_model, f"{model_type}_{pd_type}")
            pkpd_model = fix_parameters_to(pkpd_model, ests)
            pkpd_model = add_iiv(pkpd_model, ["E0"], "exp")
            for parameter in ["S", "E_max", "m"]:
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
