from sympy import exp

from pharmpy.statements import Assignment
from pharmpy.symbols import real


def transform_etas(model, transformation, etas):
    pass


def create_equation(transformation, no_of_etas):
    if transformation == 'boxcox':
        return EtaTransformation.boxcox(no_of_etas)


class EtaTransformation:
    def __init__(self, assignments):
        self.assignments = assignments

    def apply(self, eta, theta):
        self.equation.subs({'etab': eta, 'theta': theta})

    @classmethod
    def boxcox(cls, no_of_etas):
        assignments = []
        for i in range(1, no_of_etas + 1):
            symbol = S(f'etab{i}')
            expression = ((exp(S(f'etab{i}'))**(S(f'theta{i}')-1)) /
                          (S(f'theta{i}')))

            assignment = Assignment(symbol, expression)
            assignments.append(assignment)

        return cls(assignments)


def S(x):
    return real(x)
