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


def create_ini(cg, model):
    """Create the nlmixr ini section code"""
    cg.add('ini({')
    cg.indent()
    thetas = [p for p in model.parameters if p.symbol not in model.random_variables.free_symbols]
    for theta in thetas:
        theta_name = theta.name.replace('(', '').replace(')', '')
        cg.add(f'{theta_name} <- {theta.init}')
    cg.dedent()
    cg.add('})')


def create_model(cg, model):
    """Create the nlmixr model section code"""
    cg.add('model({')
    cg.indent()
    for s in model.statements:
        if isinstance(s, Assignment):
            cg.add(f'{s.symbol.name} <- {str(s.expression)}')
    cg.dedent()
    cg.add('})')


def create_fit(cg, model):
    """Create the call to fit"""
    cg.add(f'fit <- nlmixr({model.name}, dataset, "saem"')
    cg.add('print(fit)')


class Model(pharmpy.model.Model):
    def update_source(self):
        cg = CodeGenerator()
        cg.add('library(nlmixr)')
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
