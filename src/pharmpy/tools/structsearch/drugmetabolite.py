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


def create_drug_metabolite_models(model: Model) -> List[Model]:
    """
    Create candidate models for drug metabolite model structsearch.
    Currently applies a PLAIN metabolite model with and without a connected
    peripheral compartment.

    Parameters
    ----------
    model : Model
        Base model for search.

    Returns
    -------
    List[Model]
        A list of candidate models.
    """
    models = []
    for presystemic in [False, True]:
        candidate_model = add_metabolite(model, presystemic=presystemic)
        candidate_model = set_name(candidate_model, 'presystemic')

        candidate_model = candidate_model.replace(parent_model='base_metabolite')
        if presystemic:
            models.append(candidate_model)

        candidate_model = add_peripheral_compartment(candidate_model, "METABOLITE")
        candidate_model = set_name(
            candidate_model, f'{"presystemic" if presystemic else "base_metabolite"}_peripheral'
        )

        models.append(candidate_model)

    return models
