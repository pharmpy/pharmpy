import sympy
import sympy.stats as stats
from sympy import Eq, Piecewise

from pharmpy.parameter import Parameter
from pharmpy.random_variables import VariabilityLevel
from pharmpy.statements import Assignment, ModelStatements
from pharmpy.symbols import symbol as S


def add_iov(model, occ, list_of_etas=None):
    rvs, pset, sset = model.random_variables, model.parameters, model.statements
    etas = _get_etas(rvs, list_of_etas)
    iovs, etais = ModelStatements(), ModelStatements()

    categories = _get_occ_levels(model.dataset, occ)

    for i, eta in enumerate(etas, 1):
        omega_name = str(next(iter(eta.pspace.distribution.free_symbols)))
        omega = S(f'OMEGA_IOV_{i}')  # TODO: better name
        pset.add(Parameter(str(omega), init=pset[omega_name].init * 0.1))

        iov = S(f'IOV_{i}')

        values, conditions = [], []

        for j, cat in enumerate(categories, 1):
            eta_new = stats.Normal(f'ETA_IOV_{i}{j}', 0, sympy.sqrt(omega))
            eta_new.variability_level = VariabilityLevel.IOV

            rvs.add(eta_new)

            values += [S(eta_new.name)]
            conditions += [Eq(cat, S(occ))]

        expression = Piecewise(*zip(values, conditions))

        iovs.append(Assignment(iov, sympy.sympify(0)))
        iovs.append(Assignment(iov, expression))
        etais.append(Assignment(S(f'ETAI{i}'), eta + iov))

        sset.subs({eta.name: S(f'ETAI{i}')})

    iovs.extend(etais)
    iovs.extend(sset)

    model.random_variables, model.parameters, model.statements = rvs, pset, iovs

    return model


def _get_etas(rvs, list_of_etas):
    if list_of_etas is None:
        return rvs.etas
    else:
        etas = []
        for eta in list_of_etas:
            etas.append(rvs[eta.upper()])
        return etas


def _get_occ_levels(df, occ):
    levels = df[occ].unique()
    return _round_categories(levels)


def _round_categories(categories):
    categories_rounded = []
    for c in categories:
        if c.is_integer():
            categories_rounded.append(int(c))
        else:
            categories_rounded.append(c)
    categories_rounded.sort()
    return categories_rounded
