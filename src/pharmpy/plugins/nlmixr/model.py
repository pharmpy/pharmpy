import os
import subprocess
import uuid
import warnings
from pathlib import Path

import pandas as pd
import sympy
from sympy.printing.str import StrPrinter

import pharmpy.model
from pharmpy.modeling import write_csv
from pharmpy.results import ModelfitResults
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
    from pharmpy.modeling import convert_model

    generic_model = convert_model(model, 'generic')
    nlmixr_model.__dict__ = generic_model.__dict__
    nlmixr_model.filename_extension = '.R'
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


def create_dataset(cg, model, path=None):
    """Create dataset for nlmixr"""
    dataname = f'{model.name}.csv'
    if path is None:
        path = ""
    path = Path(path) / dataname
    cg.add(f'dataset <- read.csv("{path}")')


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
            omega = dist.std**2
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
        sigma = dist.std**2
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
                    sigma = dist.std**2
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
            eqs = s.to_explicit_system().odes
            for eq in eqs[:-1]:
                cg.add(f'{printer.doprint(eq.lhs)} = {printer.doprint(eq.rhs)}')
    cg.dedent()
    cg.add('})')


def create_fit(cg, model):
    """Create the call to fit"""
    cg.add(f'fit <- nlmixr({model.name}, dataset, "focei")')


class Model(pharmpy.model.Model):
    def update_source(self, path=None):
        cg = CodeGenerator()
        cg.add('library(nlmixr)')
        cg.empty_line()
        create_dataset(cg, self, path)
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
        self._path = None

    @property
    def model_code(self):
        self.update_source(path=self._path)
        return self._src

    def read_modelfit_results(self):
        try:
            rdata_path = self.database.retrieve_file(self.name, self.name + '.RDATA')
        except FileNotFoundError:
            self.modelfit_results = None
            return None
        if rdata_path is not None:
            read_modelfit_results(self, rdata_path)
            return self.modelfit_results


def read_modelfit_results(model, rdata_path):
    with warnings.catch_warnings():
        # Supress a numpy deprecation warning
        warnings.simplefilter("ignore")
        import pyreadr
    rdata = pyreadr.read_r(rdata_path)
    ofv = rdata['ofv']['ofv'][0]
    omegas_sigmas = dict()
    omega = model.random_variables.etas.covariance_matrix
    for i in range(0, omega.rows):
        for j in range(0, omega.cols):
            symb = omega.row(i)[j]
            if symb != 0:
                omegas_sigmas[symb.name] = rdata['omega'].values[i, j]
    sigma = model.random_variables.epsilons.covariance_matrix
    for i in range(len(sigma)):
        omegas_sigmas[sigma[i]] = rdata['sigma']['sigma'][i]
    thetas_index = 0
    pe = dict()
    for param in model.parameters:
        if param.fix:
            continue
        elif param.name in omegas_sigmas:
            pe[param.name] = omegas_sigmas[param.name]
        else:
            pe[param.name] = rdata['thetas']['thetas'][thetas_index]
            thetas_index += 1
    pe = pd.Series(pe)
    res = ModelfitResults(ofv=ofv, parameter_estimates=pe)
    model.modelfit_results = res


def execute_model(model):
    database = model.database
    model = convert_model(model)
    path = Path.cwd() / f'nlmixr_run_{model.name}-{uuid.uuid1()}'
    model._path = path
    path.mkdir(parents=True, exist_ok=True)
    write_csv(model, path=path)

    code = model.model_code
    cg = CodeGenerator()
    cg.add('ofv <- fit$ofv')
    cg.add('thetas <- fit$theta')
    cg.add('omega <- fit$omega')
    cg.add('sigma <- fit$sigma')
    cg.add(f'save(file="{path}/{model.name}.RDATA", ofv, thetas, omega, sigma)')
    code += f'\n{str(cg)}'
    with open(path / f'{model.name}.R', 'w') as fh:
        fh.write(code)

    from pharmpy.plugins.nlmixr import conf

    rpath = conf.rpath / 'bin' / 'Rscript'
    newenv = os.environ
    # Reset environment variables incase started from R
    # and calling other R version.
    newenv['R_LIBS_USERS'] = ''
    newenv['R_LIBS_SITE'] = ''
    subprocess.run([rpath, path / (model.name + '.R')], env=newenv)
    rdata_path = path / f'{model.name}.RDATA'
    database.store_local_file(model, path / f'{model.name}.R')
    database.store_local_file(model, rdata_path)
    read_modelfit_results(model, rdata_path)
    return model
