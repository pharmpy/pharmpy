from typing import Iterable, Union

from pharmpy.deps import numpy as np
from pharmpy.deps.scipy import stats
from pharmpy.model import Model
from pharmpy.workflows import ModelEntry


def degrees_of_freedom(parent: Union[Model, ModelEntry], child: Union[Model, ModelEntry]) -> int:
    if isinstance(child, ModelEntry):
        child_parameters = child.model.parameters
    else:
        child_parameters = child.parameters

    if isinstance(child, ModelEntry):
        parent_parameters = parent.model.parameters
    else:
        parent_parameters = parent.parameters
    return len(child_parameters) - len(parent_parameters)


def cutoff(parent: Model, child: Model, alpha: float) -> float:
    df = degrees_of_freedom(parent, child)
    return (
        0
        if df == 0
        else (
            float(stats.chi2.isf(q=alpha, df=df))
            if df > 0
            else -float(stats.chi2.isf(q=alpha, df=-df))
        )
    )


def p_value(reduced: Model, extended: Model, reduced_ofv, extended_ofv) -> float:
    dofv = reduced_ofv - extended_ofv
    df = degrees_of_freedom(reduced, extended)
    return float(stats.chi2.sf(x=dofv, df=df))


def test(parent: Model, child: Model, parent_ofv, child_ofv, alpha: float) -> bool:
    dofv = parent_ofv - child_ofv
    return dofv >= cutoff(parent, child, alpha)


def best_of_two(parent: Model, child: Model, parent_ofv, child_ofv, alpha: float) -> Model:
    return child if test(parent, child, parent_ofv, child_ofv, alpha) else parent


def best_of_many(
    parent: Model, models: Iterable[Model], parent_ofv, model_ofvs, alpha: float
) -> Model:
    # NOTE: numpy.nanargmin ignores NaN values and raises a ValueError when all
    # values are NaN.
    # See https://numpy.org/doc/stable/reference/generated/numpy.nanargmin.html
    try:
        best_index = np.nanargmin(model_ofvs)
    except ValueError:
        return parent
    return best_of_two(parent, list(models)[best_index], parent_ofv, model_ofvs[best_index], alpha)
