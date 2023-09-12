from typing import List

from pharmpy.model import Model
from pharmpy.modeling import add_metabolite, add_peripheral_compartment, set_name


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
        candidate_model = set_name(candidate_model, f'{"presystemic" if presystemic else "plain"}')

        models.append(candidate_model)

        candidate_model = add_peripheral_compartment(candidate_model, "METABOLITE")
        candidate_model = set_name(
            candidate_model, f'{"presystemic" if presystemic else "plain"}_peripheral'
        )

        models.append(candidate_model)

    return models
