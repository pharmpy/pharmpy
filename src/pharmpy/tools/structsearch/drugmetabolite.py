from typing import List

from pharmpy.model import Model
from pharmpy.modeling import add_metabolite, add_peripheral_compartment, set_name


def create_base_metabolite(model: Model) -> Model:
    """
    Create a plain metabolite models from an input model. This is used
    as the base model when searching for candidate drug metabolite model

    Parameters
    ----------
    model : Model
        Input model for search

    Returns
    -------
    Model
        Base model for drug metabolite model search

    """
    model = add_metabolite(model)
    model = set_name(model, "base_metabolite")
    model = model.replace(parent_model="base_metabolite")
    return model


def create_drug_metabolite_models(model: Model, route: str) -> List[Model]:
    """
    Create candidate models for drug metabolite model structsearch.
    Currently applies a PLAIN metabolite model with and without a connected
    peripheral compartment.

    Parameters
    ----------
    model : Model
        Base model for search.
    route : str
        Type of administration. Currently 'oral', 'iv' and 'ivoral'

    Returns
    -------
    List[Model]
        A list of candidate models.
    """
    models = []
    if route not in ('oral', 'ivoral'):
        presystemic_option = [False]
    else:
        presystemic_option = [False, True]
    for presystemic in presystemic_option:
        candidate_model = add_metabolite(model, presystemic=presystemic)
        candidate_model = set_name(candidate_model, 'presystemic')

        candidate_model = candidate_model.replace(parent_model='base_metabolite')
        if presystemic:
            models.append(candidate_model)

        candidate_model = add_peripheral_compartment(candidate_model, "METABOLITE")
        candidate_model = set_name(
            candidate_model, f'{"presystemic" if presystemic else "base_metabolite"}_peripheral'
        )
        if presystemic:
            candidate_model = candidate_model.replace(parent_model='presystemic')

        models.append(candidate_model)

    return models
