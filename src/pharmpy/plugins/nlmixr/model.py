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
    drop_columns,
    has_additive_error_model,
    has_proportional_error_model,
    has_combined_error_model,
)
from pharmpy.results import ModelfitResults
from pharmpy.tools import fit


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
    
    # Check data structure of doses
    if not check_doses(nlmixr_model):
        print_warning("The connected model data contains mixed dosage types. Nlmixr cannot handle this \nConverted model will not run on associated data")

    
    # Drop all dropped columns so it does not interfere with nlmixr
    drop_dropped_columns(nlmixr_model)
    if all(x in nlmixr_model.dataset.columns for x in ["RATE", "DUR"]):
        nlmixr_model = drop_columns(nlmixr_model, ["DUR"])

    # Update dataset
    nlmixr_model = translate_nmtran_time(nlmixr_model)
    nlmixr_model.datainfo = nlmixr_model.datainfo.replace(path = None)

    # FIXME : Redundant?? Seem to be produced during NLMIXRModeInternals()
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
        add_theta(cg, theta)

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
                # FIXME : Find another way to assert that a sigma exist
                sigma = None
                for dist in model.random_variables.epsilons:
                    sigma = dist.variance
                assert sigma is not None
                
                if has_additive_error_model(model):
                    expr, error = find_term(model, s.expression)
                    add_error_model(cg, expr, error, s.symbol.name, force_add = True)
                    add_error_relation(cg, error, s.symbol)
                elif has_proportional_error_model(model):
                    if len(sympy.Add.make_args(s.expression)) == 1:
                        expr, error = find_term(model, sympy.expand(s.expression))
                    else:
                        expr, error = find_term(model, s.expression)
                    add_error_model(cg, expr, error, s.symbol.name, force_prop = True)
                    add_error_relation(cg, error, s.symbol)
                elif has_combined_error_model(model):
                    pass
                else:
                    print_warning("Format of error model is unknown. Will try to translate either way")
                    if s.expression.is_Piecewise:
                        # Convert eps to sigma name
                        #piecewise = convert_eps_to_sigma(s, model)
                        convert_piecewise(s, cg, model)
                    else:         
                        expr, error = find_term(model, s.expression)
                        add_error_model(cg, expr, error, s.symbol.name)
                        add_error_relation(cg, error, s.symbol)
                    
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
                    lhs = remove_piecewise(printer.doprint(eq.lhs))
                    rhs = remove_piecewise(printer.doprint(eq.rhs))
                    
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
            str(cg).replace("AMT", "amt").replace("TIME", "time")
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
        if sigma[i] != 0:
            omegas_sigmas[sigma[i].name] = rdata['sigma']['fit$theta'][sigma[i].name]
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
    if nonmem_model.modelfit_results is None:
        nonmem_model.modelfit_results = fit(nonmem_model)
        nonmem_results = nonmem_model.modelfit_results.predictions.iloc[:, [0]]
    else:
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
        nonmem_results.rename(columns={nonmem_results.columns[0]: "PRED_NONMEM"}, inplace=True)
        nlmixr_results.rename(columns={"PRED": "PRED_NLMIXR"}, inplace=True)

    # Combine the two based on ID and time
    if "evid" not in nonmem_model.dataset.columns.str.lower():
        nonmem_model = modify_dataset(nonmem_model)
    nonmem_results = nonmem_results.reset_index()
    nonmem_results = nonmem_results.drop(nonmem_model.dataset[nonmem_model.dataset["EVID"] != 0].index.to_list())
    nonmem_results = nonmem_results.set_index(["ID","TIME"])
    combined_result = nonmem_results
    combined_result["PRED_NLMIXR"] = nlmixr_results["PRED_NLMIXR"].to_list()

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
    if "EVID" not in temp_model.dataset.columns:
        temp_model.dataset["EVID"] = get_evid(temp_model)
    return temp_model

def convert_piecewise(piecewise, cg, model):
    """
    Return a string where each piecewise expression has been changed to if
    elseif else statements in R
    """
    first = True
    for expr, cond in piecewise.expression.args:
        if first:
            cg.add(f'if ({cond}){{')
            expr, error = find_term(model, expr)
            add_error_model(cg, expr, error, piecewise.symbol)
            cg.add('}')
            first = False
        else:
            if cond is not sympy.S.true:
                cg.add(f'else if ({cond}){{')
                expr, error = find_term(model, expr)
                add_error_model(cg, expr, error, piecewise.symbol)
                cg.add('}')
            else:
                cg.add('else {')
                expr, error = find_term(model, expr)
                add_error_model(cg, expr, error, piecewise.symbol)
                cg.add('}')
    
    add_error_relation(cg, error, piecewise.symbol)

def piecewise_replace(expr, piecewise, s):
    if s == "":
        expr = re.sub(r'([\+\-\/\*]\s*)(Piecewise)', r'\2', expr)
        return expr.replace(f'Piecewise({piecewise})', s)
    else:
        return expr.replace(f'Piecewise({piecewise})', s)

def remove_piecewise(expr:str):
    all_piecewise = find_piecewise(expr)
    #Go into each piecewise found
    for p in all_piecewise:
        expr = piecewise_replace(expr, p, "")
    return expr
    
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
    errors = []
    
    terms = sympy.Add.make_args(expr)
    for term in terms:
        error_term = False
        for symbol in term.free_symbols:
            if str(symbol) in model.random_variables.epsilons.names:
                error_term = True
            
        if error_term:
            errors.append(term)
        else:
            if "res"  not in locals():
                res = term
            else:
                res = res + term
    
    errors_add_prop = {"add": None, "prop": None}
    
    prop = False
    res_alias = find_aliases(res, model)
    for term in errors:
        for symbol in term.free_symbols:
            for ali in find_aliases(symbol, model):
                if ali in res_alias:
                    prop = True
                    # Remove the symbol that was found
                    # and substitute res to that symbol to avoid confusion
                    term = term.subs(symbol,1)
                    res = symbol
            
        if prop:
            if errors_add_prop["prop"] is None:
                errors_add_prop["prop"] = term    
            else:
                raise ValueError("Proportional term already added. Check format of error model")
        else:
            if errors_add_prop["add"] is None:
                errors_add_prop["add"] = term
            else:
                raise ValueError("Additive term already added. Check format of error model")
    
    for pair in errors_add_prop.items():
        key = pair[0]
        term = pair[1]
        if term != None:
            term = convert_eps_to_sigma(term, model)
        errors_add_prop[key] = term    
        
    return res, errors_add_prop

def convert_eps_to_sigma(expr, model):
    eps_to_sigma = {sympy.Symbol(eps.names[0]): sympy.Symbol(str(eps.variance)) for eps in model.random_variables.epsilons}
    return expr.subs(eps_to_sigma)

def add_error_model(cg, expr, error, symbol, force_add = False, force_prop = False, force_comb = False):
    cg.add(f'{symbol} <- {expr}')
    
    if force_add:
        if error["add"]:
            cg.add(f'add_error <- {error["add"]}')
        else:
            if error["prop"]:
                cg.add(f'add_error <- {error["prop"]}')
            else:
                raise ValueError("Model should have additive error but no error was found")
    elif force_prop:
        if error["prop"]:
            cg.add(f'prop_error <- {error["prop"]}')
        else:
            if error["add"]:
                cg.add(f'prop_error <- {error["add"]}')
            else:
                raise ValueError("Model should have additive error but no error was found")
    elif force_comb:
        pass
    else:
        # Add term for the additive and proportional error (if exist)
        # as solution for nlmixr error model handling
        if error["add"]:
            cg.add(f'add_error <- {error["add"]}')
        else:
            cg.add('add_error <- 0')
        if error["prop"]:
            cg.add(f'prop_error <- {error["prop"]}')
        else:
            cg.add('prop_error <- 0')
        
def add_error_relation(cg, error, symbol):
    # Add the actual error model depedent on the previously
    # defined variable add_error and prop_error
    if error["add"] and error["prop"]:
        cg.add(f'{symbol} ~ add(add_error) + prop(prop_error)')
    elif error["add"] and not error["prop"]:
        cg.add(f'{symbol} ~ add(add_error)')
    elif not error["add"] and error["prop"]:
        cg.add(f'{symbol} ~ prop(prop_error)')
        
def find_aliases(symbol:str, model):
    aliases = [symbol]
    for expr in model.statements.after_odes:
        if symbol == expr.symbol and isinstance(expr.expression, sympy.Symbol):
            aliases.append(expr.expression)
        if symbol == expr.symbol and expr.expression.is_Piecewise:
            for e, c in expr.expression.args:
                if isinstance(e, sympy.Symbol):
                    aliases.append(e)
    return aliases

def check_doses(model):
    dataset = model.dataset
    if "RATE" in dataset.columns:
        no_bolus = len(dataset[(dataset["RATE"] == 0) & (dataset["EVID"] != 0)])
        if no_bolus != 0:
            return False
        else:
            return True
    else:
        return True
    
def add_theta(cg, theta):
    theta_name = name_mangle(theta.name)
    limit = 1000000.0
    if theta.lower > -limit and theta.upper < limit:
        cg.add(f'{theta_name} <- c({theta.lower}, {theta.init}, {theta.upper})')
    elif theta.lower == -limit and theta.upper < limit:
        cg.add(f'{theta_name} <- c(-Inf, {theta.init}, {theta.upper})')
    elif theta.lower > -limit and theta.upper == limit:
        cg.add(f'{theta_name} <- c({theta.lower}, {theta.init}, Inf)')
    else:
        cg.add(f'{theta_name} <- {theta.init}')
        
def print_warning(warning):
    print(f'-------\nWARNING : \n{warning}\n-------')