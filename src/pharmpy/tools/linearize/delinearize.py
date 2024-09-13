from typing import Optional

from pharmpy.model import Model
from pharmpy.modeling import (
    add_iiv,
    create_joint_distribution,
    get_individual_parameters,
    get_omegas,
    remove_iiv,
    remove_unused_parameters_and_rvs,
    set_initial_estimates,
    unfix_parameters,
)


def delinearize_model(
    linearized_model: Model, base_model: Model, param_mapping: Optional[dict] = None
):
    """
    Delinearize a model given a base_model to linearize to. If param_mapping is
    set, then the new model will get new ETAs based on this mapping.
    E.g param_mapping = {"ETA_1": "CL", "ETA_2": "V"}
    Otherwise, all ETAs are assumed to be the same in the both models and
    only the initial estimates will be updated.

    Parameters
    ----------
    linearized_model : Model
        Linearized model
    base_model : Model
        Model to use for the different
    param_mapping : None, dict
        Use special mapping, given as a dict. The default is None.

    Returns
    -------
    Model.

    """
    if param_mapping:

        for param in param_mapping.values():
            # Assert all mapping parameters are in the model
            if param not in get_individual_parameters(base_model):
                raise ValueError(f'Parameter "{param}" is not part of base model')

        for eta in linearized_model.random_variables.etas.names:
            # Assert all ETAs in linearized model is in param_mapping
            if eta not in param_mapping.keys():
                raise ValueError(f'{eta} is missing from param_mapping')

        dl_model = remove_iiv(base_model)  # Remove all IIV and then add based on linearized model
        for block in linearized_model.random_variables.etas:
            if len(block) > 1:
                # Add diagonal elements
                for eta in block:
                    omega = eta.variance
                    parameter = param_mapping[eta.names[0]]
                    initial_estimate = linearized_model.parameters[omega].init
                    dl_model = add_iiv(
                        dl_model,
                        parameter,
                        "exp",
                        initial_estimate=initial_estimate,
                        eta_names=eta.names[0],
                    )
                added_etas = dl_model.random_variables.etas[-len(block) :]
                added_etas_names = [eta.names[0] for eta in added_etas]

                # Create the join_normal_distribution
                dl_model = create_joint_distribution(dl_model, added_etas_names)
                new_matrix = dl_model.random_variables.etas[-1].variance
                new_initial_matrix = block.variance
                off_diagonal_updates = {}
                for row in range(1, len(added_etas)):
                    for col in range(1, row + 1):
                        param_name = new_matrix[row, col]
                        param_value = new_initial_matrix[row, col]
                        off_diagonal_updates[param_name] = param_value
            else:
                # Single ETA
                eta_name = block.names[0]
                eta_variance = block.variance
                parameter = param_mapping[eta_name]
                initial_estimate = linearized_model.parameters[eta_variance].init
                dl_model = add_iiv(
                    dl_model,
                    parameter,
                    "exp",
                    initial_estimate=initial_estimate,
                    eta_names=eta_name,
                )

        dl_model = unfix_parameters(dl_model, "DUMMYOMEGA")
        dl_model = remove_iiv(dl_model, "eta_dummy")
    else:
        if not all(o in get_omegas(base_model).names for o in get_omegas(linearized_model).names):
            raise ValueError(
                "Cannot de-linearize model with different set"
                " of random variables without param_mapping"
            )
        dl_model = set_initial_estimates(base_model, get_omegas(linearized_model).inits)
    dl_model = remove_unused_parameters_and_rvs(dl_model)
    return dl_model.update_source()
