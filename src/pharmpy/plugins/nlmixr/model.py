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
    append_estimation_step_options,
)
from pharmpy.results import ModelfitResults
from pharmpy.tools import fit
from .sanity_checks import check_model
from .modify_code import (
    CodeGenerator,
    name_mangle,
    add_ini_parameter,
    find_term,
    add_error_model,
    add_error_relation,
    remove_piecewise,
    convert_piecewise,
    convert_eq,
    )

def convert_model(model: pharmpy.model, keep_etas: bool = False) -> pharmpy.model:
    """
    Convert any model into an nlmixr model

    Parameters
    ----------
    model : pharmpy.model
        A pharmpy model object.

    Returns
    -------
    pharmpy.model
        A model converted to nlmixr format.

    """
    
    if isinstance(model, Model):
        return model.copy()

    nlmixr_model = Model()
    from pharmpy.modeling import convert_model

    generic_model = convert_model(model, 'generic')
    nlmixr_model.__dict__ = generic_model.__dict__
    nlmixr_model.internals = NLMIXRModelInternals()
    nlmixr_model.filename_extension = '.R'
    
    # Update dataset
    if keep_etas == True:
        nlmixr_model.modelfit_results = ModelfitResults(
            individual_estimates = model.modelfit_results.individual_estimates
        )
    nlmixr_model = translate_nmtran_time(nlmixr_model)
    drop_dropped_columns(nlmixr_model)
    if all(x in nlmixr_model.dataset.columns for x in ["RATE", "DUR"]):
        nlmixr_model = drop_columns(nlmixr_model, ["DUR"])
    nlmixr_model.datainfo = nlmixr_model.datainfo.replace(path = None)
    nlmixr_model.dataset = nlmixr_model.dataset.reset_index(drop=True)
    
    # Add evid
    nlmixr_model = add_evid(nlmixr_model)
    
    # Check model for warnings regarding data structure or model contents
    nlmixr_model = check_model(nlmixr_model)

    nlmixr_model.update_source()
    
    return nlmixr_model


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


def create_dataset(cg: CodeGenerator, model: pharmpy.model, path=None) -> None:
    """
    Create dataset for nlmixr

    Parameters
    ----------
    cg : CodeGenerator
        A code object associated with the model.
    model : pharmpy.model
        A pharmpy.model object.
    path : TYPE, optional
        Path to add file to. The default is None.

    Returns
    -------
    None
        Modification of code object and creation of files.

    """
    dataname = f'{model.name}.csv'
    if path is None:
        path = ""
    path = Path(path) / dataname
    cg.add(f'dataset <- read.csv("{path}")')


def create_ini(cg: CodeGenerator, model: pharmpy.model) -> None:
    """
    Create the nlmixr ini section code

    Parameters
    ----------
    cg : CodeGenerator
        A code object associated with the model.
    model : pharmpy.model
        A pharmpy.model object.

    Returns
    -------
    None
        Modification of code object.

    """
    cg.add('ini({')
    cg.indent()

    thetas = [p for p in model.parameters if p.symbol not in model.random_variables.free_symbols]
    for theta in thetas:
        if model.estimation_steps[0].method not in ["SAEM", "NLME"]:
            add_ini_parameter(cg, theta, boundary = True)
        else:
            add_ini_parameter(cg, theta)

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
            
    for sigma in get_sigmas(model):
        if model.estimation_steps[0].method not in ["SAEM", "NLME"]:
            add_ini_parameter(cg, sigma, boundary = True)
        else:
            add_ini_parameter(cg, sigma)

        
    cg.dedent()
    cg.add('})')


def create_model(cg: CodeGenerator, model: pharmpy.model) -> None:
    """
    Create the nlmixr model section code

    Parameters
    ----------
    cg : CodeGenerator
        A code object associated with the model.
    model : pharmpy.model
        A pharmpy.model object.

    Returns
    -------
    None
        Modification of code object.

    """
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
                    # FIXME : Combine with the unknown case
                    expr, error = find_term(model, s.expression)
                    add_error_model(cg, expr, error, s.symbol.name, force_comb = True)
                    add_error_relation(cg, error, s.symbol)
                else:
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


def create_fit(cg: CodeGenerator, model: pharmpy.model, path=None) -> None:
    """
    Create the call to fit

    Parameters
    ----------
    cg : CodeGenerator
        A code object associated with the model.
    model : pharmpy.model
        A pharmpy.model object.

    Returns
    -------
    None
        Modification of code object.

    """
    # FIXME : rasie error if the method does not match when evaluating
    estimation_steps = model.estimation_steps[0]
    if "fix_eta" in estimation_steps.tool_options:
        fix_eta = True
    else:
        fix_eta = False
    
    if [s.evaluation for s in model.estimation_steps._steps][0] is True:
        max_eval = 0
    else:
        max_eval = estimation_steps.maximum_evaluations
    
    method = estimation_steps.method
    interaction = estimation_steps.interaction
    
    nonmem_method_to_nlmixr = {
        "FOCE" : "foce",
        "FO" : "fo",
        "SAEM": "saem"
        }
    
    nlmixr_method = nonmem_method_to_nlmixr[method]
    if interaction and nlmixr_method != "saem":
        nlmixr_method += "i"

    if max_eval != None:
        if max_eval == 0 and nlmixr_method not in ["fo", "foi", "foce", "focei"]:
            nlmixr_method = "posthoc"
            cg.add(f'fit <- nlmixr2({model.name}, dataset, est = "{nlmixr_method}"')
        else:
            if fix_eta:
                eta_file = 'fix_eta.csv'
                if path is None:
                    path = ""
                path = Path(path) / eta_file
                cg.add(f'etas <- as.matrix(read.csv("{path}"))')
                cg.add(f'fit <- nlmixr2({model.name}, dataset, est = "{nlmixr_method}", control=foceiControl(maxOuterIterations={max_eval}, maxInnerIterations=0, etaMat = etas))')
            else:
                cg.add(f'fit <- nlmixr2({model.name}, dataset, est = "{nlmixr_method}", control=foceiControl(maxOuterIterations={max_eval}))')
    else:
        cg.add(f'fit <- nlmixr2({model.name}, dataset, est = "{nlmixr_method}")')
                
def add_evid(model):
    temp_model = model.copy()
    if "EVID" not in temp_model.dataset.columns:
        temp_model.dataset["EVID"] = get_evid(temp_model)
    return temp_model


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
        create_fit(cg, self, path)
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


def parse_modelfit_results(model: pharmpy.model, path):
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
    if "fix_eta" in model.estimation_steps[0].tool_options:
        write_fix_eta(model, path=path)
    
    code = model.model_code
    cg = CodeGenerator()
    cg.add('ofv <- fit$objDf$OBJF')
    cg.add('thetas <- as.data.frame(fit$theta)')
    cg.add('omega <- fit$omega')
    cg.add('sigma <- as.data.frame(fit$theta)')
    cg.add('log_likelihood <- fit$objDf$`Log-likelihood`')
    cg.add('runtime_total <- sum(fit$time)')
    cg.add('pred <- as.data.frame(fit[c("ID", "TIME", "PRED", "IPRED")])')
    
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
        
        txn.store_local_file(model.datainfo.path)

        plugin_path = path / 'nlmixr.json'
        with open(plugin_path, 'w') as f:
            json.dump(plugin, f, indent=2)

        txn.store_local_file(plugin_path)

        txn.store_metadata(metadata)
        txn.store_modelfit_results()
    
    res = parse_modelfit_results(model, path)
    model.modelfit_results = res
    return model


def verification(model: pharmpy.model,
                 db_name: str,
                 error: float = 10**-3,
                 return_comp : bool = False,
                 fix_eta = True,
                 ipred = True
                 ) -> bool or pd.DataFrame:
    """
    Verify that a model inputet in NONMEM format can be correctly translated to 
    nlmixr as well as verify that the predictions of the two models are the same 
    given a user specified error margin (defailt is 0.001).
    If return_comp = True, return a table of comparisons and differences in 
    predictions instead of a boolean indicating if they are the same or not

    Parameters
    ----------
    model : Model
        pharmpy Model object in NONMEM format
    db_name : str
        a string with given name of database folder for created files
    error : float, optional
        Allowed error margins for predictions. The default is 10**-3.
    return_comp : bool, optional
        Choose to return table of predictions. The default is False.

    Returns
    -------
    bool or pd.DataFrame
        Boolean indicating likeness or table of predictions from the two models.

    """
 
    nonmem_model = model.copy()

    # Save results from the nonmem model
    if nonmem_model.modelfit_results.predictions is None:
        print_step("Calculating NONMEM predictions... (this might take a while)")
        nonmem_model.modelfit_results = fit(nonmem_model)
        nonmem_results = nonmem_model.modelfit_results.predictions
    else:
        nonmem_results = nonmem_model.modelfit_results.predictions
    
    # Check that evaluation step is set to True
    if [s.evaluation for s in nonmem_model.estimation_steps._steps][0] is False:
        nonmem_model = set_evaluation_step(nonmem_model)
        
    # Set a tool option to fix theta values when running nlmixr
    if fix_eta:
        print_step("Creating .csv file with fixed ETAs...")
        nonmem_model = fixate_eta(nonmem_model)
    
    # Update the nonmem model with new estimates
    # and convert to nlmixr
    print_step("Converting NONMEM model to nlmixr2...")
    nlmixr_model = convert_model(
        update_inits(nonmem_model, nonmem_model.modelfit_results.parameter_estimates),
        keep_etas = True
    )
    
    # Execute the nlmixr model
    print_step("Executing nlmixr2 model... (this might take a while)")
    import pharmpy.workflows

    db = pharmpy.workflows.LocalDirectoryToolDatabase(db_name)
    nlmixr_model = execute_model(nlmixr_model, db)
    nlmixr_results = nlmixr_model.modelfit_results.predictions
    
    if ipred == True:
        pred = "IPRED"
    else:
        pred = "PRED"
    
    with warnings.catch_warnings():
        # Supress a numpy deprecation warning
        warnings.simplefilter("ignore")
        nonmem_results.rename(columns={pred: f'{pred}_NONMEM'}, inplace=True)
        nlmixr_results.rename(columns={pred: f'{pred}_NLMIXR'}, inplace=True)
    # Combine the two based on ID and time
    print_step("Creating result comparison table...")
    nonmem_model.dataset = nonmem_model.dataset.reset_index()
    
    if "EVID" not in nonmem_model.dataset.columns:
        nonmem_model = add_evid(nonmem_model)
    nonmem_results = nonmem_results.reset_index()
    nonmem_results = nonmem_results.drop(nonmem_model.dataset[nonmem_model.dataset["EVID"] != 0].index.to_list())
    nonmem_results = nonmem_results.set_index(["ID","TIME"])
    combined_result = nonmem_results
    combined_result[f'{pred}_NLMIXR'] = nlmixr_results[f'{pred}_NLMIXR'].to_list()

    # Add difference between the models
    combined_result[f'{pred}_DIFF'] = abs(combined_result[f'{pred}_NONMEM'] - combined_result[f'{pred}_NLMIXR'])

    combined_result["PASS/FAIL"] = "PASS"
    combined_result.loc[combined_result[f'{pred}_DIFF'] > error, "PASS/FAIL"] = "FAIL"
    
    print("Differences in population predicted values")
    print(combined_result[f'{pred}_DIFF'].describe()[["mean", "75%", "max"]].to_string(), end ="\n\n")
    
    print_step("DONE")
    if return_comp is True:
        return combined_result
    else:
        if all(combined_result["PASS/FAIL"] == "PASS"):
            return True
        else:
            return False

def print_step(s):
    print("***** ", s, " *****")

def fixate_eta(model):
    opts = {"fix_eta": True}
    model = append_estimation_step_options(model, tool_options = opts, idx=0)
    return model
            
def write_fix_eta(model, path=None, force = True):
    from pharmpy.model import data
    from pharmpy.internals.fs.path import path_absolute
    
    filename = "fix_eta.csv"
    path = path / filename
    if not force and path.exists():
        raise FileExistsError(f'File at {path} already exists.')
        
    path = path_absolute(path)
    model.modelfit_results.individual_estimates.to_csv(path, na_rep=data.conf.na_rep, index=False)
    model.datainfo = model.datainfo.replace(path=path)
    return path
            
            
            
            