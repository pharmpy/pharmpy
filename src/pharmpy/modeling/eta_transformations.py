import re
import warnings

from sympy import exp

from pharmpy.parameter import Parameter
from pharmpy.statements import Assignment, ModelStatements
from pharmpy.symbols import symbol as S


def boxcox(model, list_of_etas):
    """
    Applies a boxcox transformation to specified etas from a :class:`pharmpy.model`.

    Parameters
    ----------
    model : Model
        Pharmpy model to apply boxcox transformation to.
    list_of_etas : list
        List of etas to transform. If None, all etas will be transformed.
    """
    etas = _get_etas(model, list_of_etas)
    eta_transformation = EtaTransformation.boxcox(len(etas))
    _transform_etas(model, eta_transformation, etas)


def _get_etas(model, list_of_etas):
    rvs = model.random_variables

    if list_of_etas is None:
        return rvs.etas
    else:
        etas = []
        for eta in list_of_etas:
            try:
                etas.append(rvs[eta.upper()])
            except KeyError:
                warnings.warn(f'Random variable "{eta}" does not exist')
        return etas


def _transform_etas(model, eta_transformation, etas):
    etas_assignment, etas_subs = _create_new_etas(etas)
    thetas = _create_new_thetas(model, eta_transformation.name, len(etas))

    eta_transformation.apply(etas_assignment, thetas)
    statements_new = eta_transformation.assignments
    sset = model.statements
    sset.subs(etas_subs)

    statements_new.extend(sset)

    model.statements = statements_new


def _create_new_etas(etas_original):
    etas_subs = dict()
    etas_assignment = dict()

    for i, eta in enumerate(etas_original):
        eta_no = int(re.findall(r'\d', eta.name)[0])
        etas_subs[eta.name] = f'ETAB{eta_no}'
        etas_assignment[f'etab{i + 1}'] = f'ETAB{eta_no}'
        etas_assignment[f'eta{i + 1}'] = f'ETA({eta_no})'

    return etas_assignment, etas_subs


def _create_new_thetas(model, transformation, no_of_thetas):
    pset = model.parameters
    thetas = dict()
    theta_name = str(model.create_symbol(stem=transformation, force_numbering=True))

    if no_of_thetas == 1:
        pset.add(Parameter(theta_name, 0.01, -3, 3))
        thetas['theta1'] = theta_name
    else:
        theta_no = int(re.findall(r'\d', theta_name)[0])

        for i in range(1, no_of_thetas+1):
            pset.add(Parameter(theta_name, 0.01, -3, 3))
            thetas[f'theta{i}'] = theta_name
            theta_name = f'{transformation}{theta_no + i}'

    model.parameters = pset

    return thetas


class EtaTransformation:
    def __init__(self, name, assignments):
        self.name = name
        self.assignments = assignments

    def apply(self, etas, thetas):
        for assignment in self.assignments:
            assignment.subs(etas)
            assignment.subs(thetas)

    @classmethod
    def boxcox(cls, no_of_etas):
        assignments = ModelStatements()
        for i in range(1, no_of_etas + 1):
            symbol = S(f'etab{i}')
            expression = ((exp(S(f'eta{i}'))**S(f'theta{i}')-1) /
                          (S(f'theta{i}')))

            assignment = Assignment(symbol, expression)
            assignments.append(assignment)

        return cls('BOXCOX', assignments)

    @classmethod
    def tdist(cls, no_of_etas):
        assignments = ModelStatements()
        for i in range(1, no_of_etas + 1):
            symbol = S(f'etat{i}')

            eta = S(f'eta{i}')
            theta = S(f'theta{i}')

            num_1 = eta**2 + 1
            denom_1 = 4 * theta

            num_2 = (5 * eta**4) + (16 * eta**2 + 3)
            denom_2 = 96 * theta**2

            num_3 = (3 * eta**6) + (19 * eta**4) + (17 * eta**2) - 15
            denom_3 = 384 * theta**3

            expression = eta * (1 + (num_1/denom_1) + (num_2/denom_2) +
                                (num_3/denom_3))

            assignment = Assignment(symbol, expression)
            assignments.append(assignment)

        return cls('TDIST', assignments)

    def __str__(self):
        return str(self.assignments)
