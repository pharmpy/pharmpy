from operator import add, mul

from sympy import Symbol, exp

from pharmpy.statements import Assignment


def S(x):
    return Symbol(x, real=True)


class CovariateEffect:
    def __init__(self, template):
        self.template = template

    def apply(self, parameter, covariate, theta_name, mean, median, operation_str='*'):
        effect_name = f'{parameter}{covariate}'
        self.template.symbol = S(effect_name)

        self.template.subs(S('theta'), S(theta_name))
        self.template.subs(S('cov'), S(covariate))

        template_str = [str(symbol) for symbol in self.template.free_symbols]
        if 'mean' in template_str:
            self.template.subs(S('mean'), mean)
        else:
            self.template.subs(S('median'), median)

        operation = self._get_operation(operation_str)
        effect = Assignment(parameter, operation(S(parameter), S(effect_name)))
        return effect

    @staticmethod
    def _get_operation(operation_str):
        if operation_str == '*':
            return mul
        elif operation_str == '+':
            return add

    @classmethod
    def exponential(cls):
        symbol = S('symbol')
        expression = exp(S('theta') * (S('cov') - S('median')))
        template = Assignment(symbol, expression)

        return cls(template)

    def __str__(self):
        return str(self.template)
