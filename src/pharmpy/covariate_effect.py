from sympy import Symbol, exp

from pharmpy.statements import Assignment


def S(x):
    return Symbol(x, real=True)


class CovariateEffect:
    def __init__(self, template):
        self.template = template

    def apply(self, parameter, covariate, theta_name, statistic):
        self.template.symbol = f'{parameter}{covariate}'

        self.template.subs(S('theta'), S(theta_name))
        self.template.subs(S('cov'), S(covariate))
        self.template.subs(S('stat'), statistic)

    @classmethod
    def exponential(cls):
        symbol = S('symbol')
        expression = exp(S('theta') * (S('cov') - S('mean')))
        template = Assignment(symbol, expression)

        return cls(template)

    def __str__(self):
        return str(self.template)
