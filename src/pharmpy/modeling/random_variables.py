from __future__ import annotations

from pharmpy.basic import Expr
from pharmpy.model import Model, Parameters, RandomVariables


def replace_non_random_rvs(model: Model):
    """Replace all random variables that are not actually random

    Some random variables are constant. For example a normal
    distribution with the variance parameter fixed to 0 will always
    yield a single value when sampled. This function will find all such
    random variables and replace them with their constant value in the model.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    Model
        A new model
    """

    dists = model.random_variables
    keep = []  # Dists
    remove = []  # Pop-params
    d = {}

    for dist in dists:
        for parname in dist.parameter_names:
            param = model.parameters[parname]
            if not (param.init == 0.0 and param.fix):
                keep.append(dist)
                break
        else:
            for parname in dist.parameter_names:
                remove.append(parname)
                d[Expr.symbol(parname)] = Expr.integer(0)
            for name in dist.names:
                d[Expr.symbol(name)] = Expr.integer(0)

    new_parameters = Parameters(tuple(p for p in model.parameters if p.name not in remove))
    new_statements = model.statements.subs(d)
    new_rvs = RandomVariables(
        tuple(keep), eta_levels=dists.eta_levels, epsilon_levels=dists.epsilon_levels
    )

    model = model.replace(
        statements=new_statements, parameters=new_parameters, random_variables=new_rvs
    )
    model = model.update_source()
    return model
