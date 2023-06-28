from typing import List

from pharmpy.model import Model
from pharmpy.modeling import add_effect_compartment, fix_parameters_to, set_direct_effect, set_name


def create_pkpd_models(model_type: str, model: Model, ests: List):
    """Create pkpd models

    Parameters
    ----------
    model_type : str
        Type of PD model. Currently either 'direct_effect' or 'effect_compartment'
    model : Model
        Pharmpy PK model
    ests : list
       List of estimated PK parameters

    Returns
    -------
    Pharmpy model
    """
    models = []
    params = model.parameters.symbols
    params_dict = {f"{params[p]}": ests[p] for p in range(len(params))}
    pd_types = ["baseline", "linear", "Emax", "sigmoid", "step", "loglin"]
    for m in pd_types:
        if model_type == "direct_effect":
            pkpd_model = set_direct_effect(model, expr=m)
        elif model_type == "effect_compartment":
            pkpd_model = add_effect_compartment(model, expr=m)
        pkpd_model = set_name(pkpd_model, f"{model_type}_{m}")
        pkpd_model = fix_parameters_to(pkpd_model, params_dict)
        models.append(pkpd_model)
    return models
