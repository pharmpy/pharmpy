from itertools import chain
from typing import Iterable, List, Union

import numpy as np
from scipy.stats import chi2

from pharmpy.model import Model


def _ofv(model: Model) -> float:
    return np.nan if model.modelfit_results is None else model.modelfit_results.ofv


def _dofv(parent: Model, model: Model) -> float:
    return _ofv(parent) - _ofv(model)


def _best_by_ofv(models: Iterable[Model], default: Union[None, Model] = None) -> Model:
    if default is None:
        return min(models, key=_ofv)
    else:
        return min(models, key=_ofv, default=default)


def _df(parent: Model, child: Model) -> int:
    return len(child.parameters) - len(parent.parameters)


def cutoff(parent: Model, child: Model, alpha: float) -> float:
    df = _df(parent, child)
    return float(chi2.isf(q=alpha, df=df))


def p_value(parent: Model, child: Model) -> float:
    x = _dofv(parent, child)
    df = _df(parent, child)
    return float(chi2.sf(x=x, df=df))


def test(parent: Model, child: Model, alpha: float) -> bool:
    return _dofv(parent, child) >= cutoff(parent, child, alpha)


def best_of_two(parent: Model, child: Model, alpha: float) -> Model:
    return child if test(parent, child, alpha) else parent


def best_of_many(parent: Model, models: Iterable[Model], alpha: float) -> Model:
    best_candidate = _best_by_ofv(models)
    return best_of_two(parent, best_candidate, alpha)


def best_of_subtree(root: Model, nodes: List[Model], alpha: float) -> Model:
    models_dict = {model.name: model for model in chain((root,), nodes)}

    def parent(model: Model):
        return models_dict[model.parent_model]

    return _best_by_ofv(
        filter(lambda model: test(parent(model), model, alpha), nodes),
        default=root,
    )
