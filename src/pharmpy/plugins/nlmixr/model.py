import sympy
from sympy.printing.str import StrPrinter

import pharmpy.model
from pharmpy.statements import Assignment


class CodeGenerator:
    def __init__(self):
        self.indent_level = 0
        self.lines = []

    def indent(self):
        self.indent_level += 4

    def dedent(self):
        self.indent_level -= 4

    def add(self, line):
        self.lines.append(f'{" " * self.indent_level}{line}')

    def empty_line(self):
        self.lines.append('')

    def __str__(self):
        return '\n'.join(self.lines)


def convert_model(model):
    """Convert any model into an nlmixr model"""
    nlmixr_model = Model()
    generic_model = model.to_generic_model()
    nlmixr_model.__dict__ = generic_model.__dict__
    nlmixr_model.update_source()
    return nlmixr_model


def name_mangle(s):
    return s.replace('(', '').replace(')', '').replace(',', '_')


class ExpressionPrinter(StrPrinter):
    def __init__(self, amounts):
        self.amounts = amounts
        super().__init__()

    def _print_Symbol(self, expr):
        return name_mangle(expr.name)

    def _print_Derivative(self, expr):
        fn = expr.args[0]
        return f'd/dt({fn.name})'

    def _print_Function(self, expr):
        name = expr.func.__name__
        if name in self.amounts:
            return expr.func.__name__
        else:
            return expr.func.__name__ + f'({self.stringify(expr.args, ", ")})'


def create_dataset(cg, model):
    """Create dataset for nlmixr"""
    dataname = f'{model.name}.csv'
    # model.dataset.pharmpy.write_csv(dataname)
    cg.add(f'dataset <- read.csv("{dataname}")')


def create_ini(cg, model):
    """Create the nlmixr ini section code"""
    cg.add('ini({')
    cg.indent()

    thetas = [p for p in model.parameters if p.symbol not in model.random_variables.free_symbols]
    for theta in thetas:
        theta_name = name_mangle(theta.name)
        cg.add(f'{theta_name} <- {theta.init}')

    for rvs, dist in model.random_variables.etas.distributions():
        if len(rvs) == 1:
            omega = dist.std ** 2
            init = model.parameters[omega.name].init
            cg.add(f'{name_mangle(rvs[0].name)} ~ {init}')
        else:
            omega = dist.sigma
            inits = []
            for row in range(omega.rows):
                for col in range(row + 1):
                    inits.append(model.parameters[omega[row, col].name].init)
            cg.add(f'{" + ".join([name_mangle(rv.name) for rv in rvs])} ~ c({", ".join(inits)})')

    for rvs, dist in model.random_variables.epsilons.distributions():
        sigma = dist.std ** 2
        cg.add(f'{name_mangle(sigma.name)} <- {model.parameters[sigma.name].init}')

    cg.dedent()
    cg.add('})')


def create_model(cg, model):
    """Create the nlmixr model section code"""
    amounts = [am.name for am in list(model.statements.ode_system.amounts)]
    printer = ExpressionPrinter(amounts)

    cg.add('model({')
    cg.indent()
    for s in model.statements:
        if isinstance(s, Assignment):
            if s.symbol == model.dependent_variable:

                for rvs, dist in model.random_variables.epsilons.distributions():
                    sigma = dist.std ** 2
                # FIXME: Needs to be generalized
                cg.add('Y <- F')
                cg.add(f'{s.symbol.name} ~ prop({name_mangle(sigma.name)})')
            else:
                expr = s.expression
                if expr.is_Piecewise:
                    first = True
                    for value, cond in expr.args:
                        if cond is not sympy.S.true:
                            if first:
                                cg.add(f'if ({cond}) {{')
                                first = False
                            else:
                                cg.add(f'}} else if ({cond}) {{')
                        else:
                            cg.add('} else {')
                        cg.indent()
                        cg.add(f'{s.symbol.name} <- {printer.doprint(value)}')
                        cg.dedent()
                    cg.add('}')
                else:
                    cg.add(f'{s.symbol.name} <- {printer.doprint(expr)}')

        else:
            eqs, _ = s.to_explicit_odes()
            for eq in eqs[:-1]:
                cg.add(f'{printer.doprint(eq.lhs)} = {printer.doprint(eq.rhs)}')
    cg.dedent()
    cg.add('})')


def create_fit(cg, model):
    """Create the call to fit"""
    cg.add(f'fit <- nlmixr({model.name}, dataset, "focei")')
    cg.add('print(fit)')


class Model(pharmpy.model.Model):
    def update_source(self):
        cg = CodeGenerator()
        cg.add('library(nlmixr)')
        cg.empty_line()
        create_dataset(cg, self)
        cg.empty_line()
        cg.add(f'{self.name} <- function() {{')
        cg.indent()
        create_ini(cg, self)
        create_model(cg, self)
        cg.dedent()
        cg.add('}')
        cg.empty_line()
        create_fit(cg, self)
        self._src = str(cg)

    def __str__(self):
        return self._src
