import json
import os
import subprocess
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

import pharmpy.model
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy, sympy_printing
from pharmpy.model import Assignment
from pharmpy.modeling import (
    get_evid,
    get_sigmas,
    get_thetas,
    set_evaluation_step,
    update_inits,
    write_csv,
    translate_nmtran_time,
    drop_dropped_columns,
    has_additive_error_model,
    has_proportional_error_model,
    has_combined_error_model,
)
from pharmpy.results import ModelfitResults


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
    if isinstance(model, Model):
        return model.copy()

    nlmixr_model = Model()
    from pharmpy.modeling import convert_model

    generic_model = convert_model(model, 'generic')
    nlmixr_model.__dict__ = generic_model.__dict__
    nlmixr_model.internals = NLMIXRModelInternals()
    nlmixr_model.filename_extension = '.R'

    # Update dataset to lowercase and add evid
    nlmixr_model = modify_dataset(nlmixr_model)
    
    # Drop all dropped columns so it does not interfere with nlmixr
    drop_dropped_columns(nlmixr_model)

    # Update dataset
    translate_nmtran_time(nlmixr_model)
    nlmixr_model.datainfo = nlmixr_model.datainfo.replace(path = None)

    nlmixr_model.update_source()
    return nlmixr_model


def name_mangle(s):
    return s.replace('(', '').replace(')', '').replace(',', '_')


class ExpressionPrinter(sympy_printing.str.StrPrinter):
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

    for dist in model.random_variables.etas:
        omega = dist.variance
        if len(dist.names) == 1:
            init = model.parameters[omega.name].init
            cg.add(f'{name_mangle(dist.names[0])} ~ {init}')
        else:
            inits = []
            for row in range(omega.rows):
                for col in range(row + 1):
                    inits.append(model.parameters[omega[row, col].name].init)
            cg.add(
                f'{" + ".join([name_mangle(name) for name in dist.names])} ~ c({", ".join([str(x) for x in inits])})'
            )

    for dist in model.random_variables.epsilons:
        sigma = dist.variance
        cg.add(f'{name_mangle(sigma.name)} <- {model.parameters[sigma.name].init}')

    cg.dedent()
    cg.add('})')


def create_model(cg, model):
    """Create the nlmixr model section code"""
    if model.statements.ode_system:
        amounts = [am.name for am in list(model.statements.ode_system.amounts)]
        printer = ExpressionPrinter(amounts)

    cg.add('model({')
    cg.indent()
    for s in model.statements:
        if isinstance(s, Assignment):
            if s.symbol == model.dependent_variable:
                sigma = None
                for dist in model.random_variables.epsilons:
                    sigma = dist.variance
                assert sigma is not None
                #cg.add('Y <- F')
                #cg.add(f'{s.symbol.name} ~ prop({name_mangle(sigma.name)})')
                # FIXME: Needs to be generalized
                if has_additive_error_model(model):
                    # Find the term with NO EPS in it
                    expr, error = find_term(model, s.expression)
                    cg.add(f'{s.symbol.name} <- {expr}')
                    cg.add(f'{s.symbol.name} ~ add({name_mangle(sigma.name)})')
                elif has_proportional_error_model(model):
                    # Find the term with NO EPS in it
                    expr, error = find_term(model, s.expression)
                    cg.add(f'{s.symbol.name} <- {expr}')
                    cg.add(f'{s.symbol.name} ~ prop({name_mangle(sigma.name)})')
                elif has_combined_error_model(model):
                    # Find the termwith NO EPS in it
                    pass
                else:
                    # TODO: Implement special case if not additive but
                    # sigma is 1 with a scaling theta factor
                    if model.parameters[sigma.name].init == 1:
                        raise Warning("In its current state the error model cannot be handled")
                    raise ValueError("Error model cannot be handled by nlmixr")
            else:
                expr = s.expression
                if expr.is_Piecewise:
                    first = True
                    for value, cond in expr.args:
                        if cond is not sympy.S.true:
                            if cond.atoms(sympy.Eq):
                                cond = convert_eq(cond)
                            if first:
                                cg.add(f'if ({cond}) {{')
                                first = False
                            else:
                                cg.add(f'}} else if ({cond}) {{')
                        else:
                            cg.add('} else {')
                        cg.indent()
                        cg.add(f'{s.symbol.name} <- {value}')
                        cg.dedent()
                    cg.add('}')
                else:
                    cg.add(f'{s.symbol.name} <- {expr}')

        else:
            for eq in s.eqs:
                # Should remove piecewise from these equations in nlmixr
                if eq.atoms(sympy.Piecewise):
                    lhs = convert_piecewise(printer.doprint(eq.lhs))
                    rhs = convert_piecewise(printer.doprint(eq.rhs))
                    cg.add(f'{lhs} = {rhs}')
                else:
                    cg.add(f'{printer.doprint(eq.lhs)} = {printer.doprint(eq.rhs)}')
    cg.dedent()
    cg.add('})')


def create_fit(cg, model):
    """Create the call to fit"""
    if [s.evaluation for s in model.estimation_steps._steps][0] is False:
        cg.add(f'fit <- nlmixr2({model.name}, dataset, "focei")')
    else:
        cg.add(f'fit <- nlmixr2({model.name}, dataset, "focei",control=foceiControl(maxOuterIterations=0))')


@dataclass
class NLMIXRModelInternals:
    src: Optional[str] = None
    path: Optional[Path] = None


class Model(pharmpy.model.Model):
    def __init__(self):
        self.internals = NLMIXRModelInternals()

    def update_source(self, path=None):
        cg = CodeGenerator()
        cg.add('library(nlmixr2)')
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
        # Create lowercase id, time and amount symbols for nlmixr to be able
        # to run
        self.internals.src = (
            str(cg).replace("AMT", "amt").replace("TIME", "time").replace("ID", "id")
        )
        self.internals.path = None

    @property
    def model_code(self):
        self.update_source(path=self.internals.path)
        code = self.internals.src
        assert code is not None
        return code


def parse_modelfit_results(model, path):
    rdata_path = path / (model.name + '.RDATA')
    with warnings.catch_warnings():
        # Supress a numpy deprecation warning
        warnings.simplefilter("ignore")
        import pyreadr
    try:
        rdata = pyreadr.read_r(rdata_path)
    except (FileNotFoundError, OSError):
        return None

    rdata["thetas"] = rdata["thetas"].loc[get_thetas(model).names]
    rdata["sigma"] = rdata["sigma"].loc[get_sigmas(model).names]

    ofv = rdata['ofv']['ofv'][0]
    omegas_sigmas = {}
    omega = model.random_variables.etas.covariance_matrix
    for i in range(0, omega.rows):
        for j in range(0, omega.cols):
            symb = omega.row(i)[j]
            if symb != 0: 
                omegas_sigmas[symb.name] = rdata['omega'].values[i, j]
    sigma = model.random_variables.epsilons.covariance_matrix
    for i in range(len(sigma)):
        omegas_sigmas[sigma[i].name] = rdata['sigma']['fit$theta'][i]
    thetas_index = 0
    pe = {}
    for param in model.parameters:
        if param.fix:
            continue
        elif param.name in omegas_sigmas:
            pe[param.name] = omegas_sigmas[param.name]
        else:
            pe[param.name] = rdata['thetas']['fit$theta'][thetas_index]
            thetas_index += 1

    name = model.name
    description = model.description
    pe = pd.Series(pe)
    predictions = rdata['pred'].set_index(["ID", "TIME"])
    predictions.index = predictions.index.set_levels(
        predictions.index.levels[0].astype("float64"), level=0
    )

    res = ModelfitResults(
        name=name, description=description, ofv=ofv, parameter_estimates=pe, predictions=predictions
    )
    return res


def execute_model(model, db):
    database = db.model_database
    model = convert_model(model)
    path = Path.cwd() / f'nlmixr_run_{model.name}-{uuid.uuid1()}'
    model.internals.path = path
    meta = path / '.pharmpy'
    meta.mkdir(parents=True, exist_ok=True)
    # This csv file need to have lower case time and amt in order to function
    # otherwise the model will not run for nlmixr2.
    # column with eventID is CRUCIAL for nlmixr, added in write_csv file
    # if not yet existing
    # DONE (within write_csv)
    write_csv(model, path=path)

    code = model.model_code
    cg = CodeGenerator()
    cg.add('ofv <- fit$objDf$OBJF')
    cg.add('thetas <- as.data.frame(fit$theta)')
    cg.add('omega <- fit$omega')
    cg.add('sigma <- as.data.frame(fit$theta)')
    cg.add('log_likelihood <- fit$objDf$`Log-likelihood`')
    cg.add('runtime_total <- sum(fit$time)')
    cg.add('pred <- as.data.frame(fit[c("ID", "TIME", "PRED")])')

    cg.add(
        f'save(file="{path}/{model.name}.RDATA",ofv, thetas, omega, sigma, log_likelihood, runtime_total, pred)'
    )
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

    stdout = path / 'stdout'
    stderr = path / 'stderr'

    args = [str(rpath), str(path / (model.name + '.R'))]

    with open(stdout, "wb") as out, open(stderr, "wb") as err:
        result = subprocess.run(args, stdin=subprocess.DEVNULL, stderr=err, stdout=out, env=newenv)

    rdata_path = path / f'{model.name}.RDATA'

    metadata = {
        'plugin': 'nlmixr',
        'path': str(path),
    }

    plugin = {
        'rpath': str(rpath),
        'commands': [
            {
                'args': args,
                'returncode': result.returncode,
                'stdout': 'stdout',
                'stderr': 'stderr',
            }
        ],
    }

    with database.transaction(model) as txn:

        txn.store_local_file(path / f'{model.name}.R')
        txn.store_local_file(rdata_path)

        txn.store_local_file(stdout)
        txn.store_local_file(stderr)

        plugin_path = path / 'nlmixr.json'
        with open(plugin_path, 'w') as f:
            json.dump(plugin, f, indent=2)

        txn.store_local_file(plugin_path)

        txn.store_metadata(metadata)
        txn.store_modelfit_results()

    res = parse_modelfit_results(model, path)
    model.modelfit_results = res
    return model


def verification(model, db_name, error=10**-3, return_comp=False):

    nonmem_model = model.copy()

    # Save results from the nonmem model
    nonmem_results = nonmem_model.modelfit_results.predictions.iloc[:, [0]]

    # Check that evaluation step is set to True
    if [s.evaluation for s in nonmem_model.estimation_steps._steps][0] is False:
        nonmem_model = set_evaluation_step(nonmem_model)

    # Update the nonmem model with new estimates
    # and convert to nlmixr
    nlmixr_model = convert_model(
        update_inits(nonmem_model, nonmem_model.modelfit_results.parameter_estimates)
    )
    # Execute the nlmixr model
    import pharmpy.workflows

    db = pharmpy.workflows.LocalDirectoryToolDatabase(db_name)
    nlmixr_model = execute_model(nlmixr_model, db)

    nlmixr_results = nlmixr_model.modelfit_results.predictions

    with warnings.catch_warnings():
        # Supress a numpy deprecation warning
        warnings.simplefilter("ignore")
        nonmem_results.rename(columns={"PRED": "PRED_NONMEM"}, inplace=True)
        nlmixr_results.rename(columns={"PRED": "PRED_NLMIXR"}, inplace=True)

    # Combine the two based on ID and time
    combined_result = pd.merge(nonmem_results, nlmixr_results, left_index=True, right_index=True)

    # Add difference between the models
    combined_result["DIFF"] = abs(combined_result["PRED_NONMEM"] - combined_result["PRED_NLMIXR"])

    combined_result["PASS/FAIL"] = "PASS"
    combined_result.loc[combined_result["DIFF"] > error, "PASS/FAIL"] = "FAIL"

    if return_comp is True:
        return combined_result
    else:
        if all(combined_result["PASS/FAIL"] == "PASS"):
            return True
        else:
            return False


def modify_dataset(model):
    temp_model = model.copy()
    temp_model.dataset["evid"] = get_evid(temp_model)
    return temp_model

def convert_piecewise(expr: str, if_statement = False):
    """
    Return a string where each piecewise expression has been changed to if
    elseif else statements in R
    """
    all_piecewise = find_piecewise(expr)
    
    #Go into each piecewise found
    for p in all_piecewise:
        
        if if_statement is False:
            expr = piecewise_replace(expr, p, "")
        else:
            #Find start point for all arguments
            p_d = find_parentheses(p)
            p_start = [list(p_d.keys())[0]]
            for start in p_d:
                if start > p_d[p_start[-1]]:
                    p_start.append(start)
            
            #go through all arguments
            #Add the first condition as an if statement
            p_arg = p[p_start[0]+1:p_d[p_start[0]]].split(",")
            elseif = f'if ({p_arg[1].strip()}) {{{p_arg[0].strip()}}}'
            
            for start in p_start[1:]:
                p_arg = p[start+1:p_d[start]]
                p_arg = p_arg.split(",")
                # Add all others as else if
                elseif += f'else if({p_arg[1].strip()}) {{{p_arg[0].strip()}}}'
            
            # Replace the piecewise in the expression
            expr = piecewise_replace(expr, p, elseif)
            expr = expr.replace("True", "TRUE")
        
    return expr

def piecewise_replace(expr, piecewise, s):
    if s == "":
        expr = re.sub(r'([\+\-\/\*]\s*)(Piecewise)', r'\2', expr)
        return expr.replace(f'Piecewise({piecewise})', s)
    else:
        return expr.replace(f'Piecewise({piecewise})', s)
    
def find_piecewise(expr):
    
    d = find_parentheses(expr)
        
    piecewise_start = [m.start() + len("Piecewise") for m in re.finditer("Piecewise", expr)]
    
    all_piecewise = []
    for p in piecewise_start:
        if p in d:
            all_piecewise.append(expr[p+1:d[p]])
    return all_piecewise

def find_parentheses(s):
    start = [] # all opening parentheses
    d = {}
    
    for i, c in enumerate(s):
        if c == '(':
             start.append(i)
        if c == ')':
            try:
                d[start.pop()] = i
            except IndexError:
                print('Too many closing parentheses')
    if start:  # check if stack is empty afterwards
        print('Too many opening parentheses')
        
    return d

def convert_eq(cond):
    cond = sympy.pretty(cond)
    cond = cond.replace("=", "==")
    cond = cond.replace("∧", "&")
    cond = cond.replace("∨", "|")
    cond = re.sub(r'(ID\s*==\s*)(\d+)', r"\1'\2'", cond)
    return cond

def find_term(model, expr):
    first_error = True
    first_res = True
    
    terms = sympy.Add.make_args(expr)
    for term in terms:
        error_term = False
        for symbol in term.free_symbols:
            if str(symbol) in model.random_variables.epsilons.names:
                if first_error:
                    error = term
                    first_error = False
                    error_term = True
                else:
                    error = error + term
                    error_term = True
        if not error_term:
            if first_res:
                res = term
                first_res = False
            else:
                res = res + term
            
    return res, error