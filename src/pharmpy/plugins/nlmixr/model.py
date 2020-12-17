import pharmpy.model


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
    cg.dedent()
    cg.add('})')


class Model(pharmpy.model.Model):
    def update_source(self):
        cg = CodeGenerator()
        cg.add('library(nlmixr)')
        cg.empty_line()
        cg.add(f'{self.name} <- function() {{')
        cg.indent()
        create_ini(cg, self)
        cg.dedent()
        cg.add('}')
        self._src = str(cg)

    def __str__(self):
        return self._src
