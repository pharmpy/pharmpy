import json
import os
import subprocess
import uuid
import warnings
from dataclasses import dataclass, replace
from pathlib import Path, PosixPath
from typing import Optional, Union

import pharmpy.model
from pharmpy.deps import pandas as pd
from pharmpy.internals.code_generator import CodeGenerator
from pharmpy.modeling import (
    append_estimation_step_options,
    drop_columns,
    get_evid,
    get_sigmas,
    get_thetas,
    set_evaluation_step,
    translate_nmtran_time,
    update_inits,
    write_csv,
)
from pharmpy.results import ModelfitResults
from pharmpy.tools import fit
from pharmpy.workflows.log import Log

from .error_model import res_error_term
from .ini import add_eta, add_sigma, add_theta
from .model_block import add_bioavailability, add_lag_times, add_ode, add_statements
from .sanity_checks import check_model


def convert_model(
    model: pharmpy.model.Model,
    keep_etas: bool = False,
    skip_check: bool = False,
    updated_estimates: bool = False,
) -> pharmpy.model.Model:
    """
    Convert a NONMEM model into an nlmixr model

    Parameters
    ----------
    model : pharmpy.model.Model
        A NONMEM pharmpy model object
    keep_etas : bool, optional
        Decide if NONMEM estimated thetas are to be used. The default is False.
    skip_check : bool, optional
        Skip determination of error model type. Could speed up conversion. The default is False.

    Returns
    -------
    pharmpy.model.Model
        A model converted to nlmixr format.

    """

    if isinstance(model, Model):
        return model

    if updated_estimates:
        model = update_inits(model, model.modelfit_results.parameter_estimates)

    nlmixr_model = Model(
        internals=NLMIXRModelInternals(),
        parameters=model.parameters,
        random_variables=model.random_variables,
        statements=model.statements,
        dependent_variables=model.dependent_variables,
        estimation_steps=model.estimation_steps,
        filename_extension='.R',
        datainfo=model.datainfo,
        dataset=model.dataset,
        name=model.name,
        description=model.description,
    )

    # Update dataset
    if model.dataset is not None:
        if keep_etas is True:
            nlmixr_model = nlmixr_model.replace(
                modelfit_results=ModelfitResults(
                    individual_estimates=model.modelfit_results.individual_estimates
                )
            )
            nlmixr_model = fixate_eta(nlmixr_model)

        nlmixr_model = translate_nmtran_time(nlmixr_model)
        # FIXME: dropping columns runs update source which becomes redundant.
        # drop_dropped_columns(nlmixr_model)
        if all(x in nlmixr_model.dataset.columns for x in ["RATE", "DUR"]):
            nlmixr_model = drop_columns(nlmixr_model, ["DUR"])
        nlmixr_model = nlmixr_model.replace(
            datainfo=nlmixr_model.datainfo.replace(path=None),
            dataset=nlmixr_model.dataset.reset_index(drop=True),
        )

        # Add evid
        nlmixr_model = add_evid(nlmixr_model)

    # Check model for warnings regarding data structure or model contents
    nlmixr_model = check_model(nlmixr_model, skip_error_model_check=skip_check)

    nlmixr_model.update_source()

    return nlmixr_model


def create_dataset(cg: CodeGenerator, model: pharmpy.model.Model, path=None) -> None:
    """
    Create dataset for nlmixr

    Parameters
    ----------
    cg : CodeGenerator
        A code object associated with the model.
    model : pharmpy.model.Model
        A pharmpy.model object.
    path : TYPE, optional
        Path to add file to. The default is None.

    """
    dataname = f'{model.name}.csv'
    if path is None:
        path = ""
    path = Path(path) / dataname
    cg.add(f'dataset <- read.csv("{path}")')


def create_ini(cg: CodeGenerator, model: pharmpy.model.Model) -> None:
    """
    Create the nlmixr ini block code

    Parameters
    ----------
    cg : CodeGenerator
        A code object associated with the model.
    model : pharmpy.model.Model
        A pharmpy.model object.

    """
    cg.add('ini({')
    cg.indent()

    add_theta(model, cg)

    add_eta(model, cg)

    add_sigma(model, cg)

    cg.dedent()
    cg.add('})')


def create_model(cg: CodeGenerator, model: pharmpy.model.Model) -> None:
    """
    Create the nlmixr model block code

    Parameters
    ----------
    cg : CodeGenerator
        A code object associated with the model.
    model : pharmpy.model.Model
        A pharmpy.model object.

    """

    cg.add('model({')

    # Add statements before ODEs
    cg.indent()
    if len(model.statements.after_odes) != 0:
        add_statements(model, cg, model.statements.before_odes)

    # Add the ODEs
    cg.add("")
    cg.add("# --- DIFF EQUATIONS ---")
    if model.statements.ode_system:
        add_ode(model, cg)
    cg.add("")

    # Find what kind of error model we are looking at
    dv = list(model.dependent_variables.keys())[0]
    dv_statement = model.statements.find_assignment(dv)

    only_piecewise = False
    if dv_statement.expression.is_Piecewise:
        only_piecewise = True
        dependencies = set()
        res_alias = set()
        for s in model.statements.after_odes:
            if s.symbol == dv:
                if s.expression.is_Piecewise:
                    for value, cond in s.expression.args:
                        if value != dv:
                            dv_term = res_error_term(model, value)
                            dependencies.update(dv_term.dependencies())

                            dv_term.create_res_alias()
                            res_alias.update(dv_term.res_alias)
                else:
                    dv_term = res_error_term(model, s.expression)
                    dependencies.update(dv_term.dependencies())

                    dv_term.create_res_alias()
                    res_alias.update(dv_term.res_alias)
    else:
        dv_term = res_error_term(model, dv_statement.expression)
        dependencies = dv_term.dependencies()
        dv_term.create_res_alias()
        res_alias = dv_term.res_alias

    # Add bioavailability statements
    if model.statements.ode_system is not None:
        add_bioavailability(model, cg)
        add_lag_times(model, cg)

    # Add statements after ODEs
    if len(model.statements.after_odes) == 0:
        statements = model.statements
    else:
        statements = model.statements.after_odes
    add_statements(
        model, cg, statements, only_piecewise, dependencies=dependencies, res_alias=res_alias
    )

    cg.dedent()
    cg.add('})')


def create_fit(cg: CodeGenerator, model: pharmpy.model.Model) -> None:
    """
    Create the call to fit for the nlmixr model with appropriate methods and datasets

    Parameters
    ----------
    cg : CodeGenerator
        A code object associated with the model.
    model : pharmpy.model
        A pharmpy.model.Model object.

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

    nonmem_method_to_nlmixr = {"FOCE": "foce", "FO": "fo", "SAEM": "saem"}

    if method not in nonmem_method_to_nlmixr.keys():
        nlmixr_method = "focei"
    else:
        nlmixr_method = nonmem_method_to_nlmixr[method]

    if interaction and nlmixr_method != "saem":
        nlmixr_method += "i"

    if max_eval is not None:
        if max_eval == 0 and nlmixr_method not in ["fo", "foi", "foce", "focei"]:
            nlmixr_method = "posthoc"
            cg.add(f'fit <- nlmixr2({model.name}, dataset, est = "{nlmixr_method}"')
        else:
            f = f'fit <- nlmixr2({model.name}, dataset, est = "{nlmixr_method}", '
            if fix_eta:
                f += f'control=foceiControl(maxOuterIterations={max_eval}, maxInnerIterations=0, etaMat = etas))'
            else:
                f += f'control=foceiControl(maxOuterIterations={max_eval}))'
            cg.add(f)
    else:
        cg.add(f'fit <- nlmixr2({model.name}, dataset, est = "{nlmixr_method}")')


def add_evid(model: pharmpy.model.Model) -> pharmpy.model.Model:
    temp_model = model
    if "EVID" not in temp_model.dataset.columns:
        temp_model.dataset["EVID"] = get_evid(temp_model)
    return temp_model


@dataclass
class NLMIXRModelInternals:
    src: Optional[str] = None
    path: Optional[Path] = None


class Model(pharmpy.model.Model):
    def __init__(self, **kwargs):
        super().__init__(
            **kwargs,
        )

    def update_source(self):
        cg = CodeGenerator()
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
        self.internals.src = str(cg).replace("AMT", "amt").replace("TIME", "time")
        self.internals.path = None
        code = str(cg).replace("AMT", "amt").replace("TIME", "time")
        internals = replace(self.internals, src=code)
        model = self.replace(internals=internals)
        return model

    @property
    def model_code(self):
        model = self.update_source()
        code = model.internals.src
        assert code is not None
        return code


def parse_modelfit_results(
    model: pharmpy.model.Model, path: PosixPath
) -> Union[None, ModelfitResults]:
    """
    Create ModelfitResults object for given model object taken from values saved in executed Rdata file

    Parameters
    ----------
    model : pharmpy.model.Model
        An nlmixr pharmpy model object.
    path : PosixPath
        A path to folder with model and data files.

    Returns
    -------
    Union[None, ModelfitResults]
        Either return ModelfitResult object or None if Rdata file not found.

    """
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
    s = []
    for sigma in get_sigmas(model):
        if sigma.init != 1 and not sigma.fix:
            s.append(sigma.name)
    rdata["sigma"] = rdata["sigma"].loc[s]

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
            s = sigma[i]
            if model.parameters[s].init != 1 and not model.parameters[s].fix:
                omegas_sigmas[sigma[i].name] = rdata['sigma']['fit$theta'][sigma[i].name]
    thetas_index = 0
    pe = {}
    for param in model.parameters:
        if param.fix:
            continue
        elif param.name in omegas_sigmas:
            pe[param.name] = omegas_sigmas[param.name]
        else:
            pe[param.name] = rdata['thetas']['fit$theta'][param.name]
            thetas_index += 1

    name = model.name
    description = model.description
    pe = pd.Series(pe)
    predictions = rdata['pred'].set_index(["ID", "TIME"])
    predictions.index = predictions.index.set_levels(
        predictions.index.levels[0].astype("float64"), level=0
    )

    res = ModelfitResults(
        name=name,
        description=description,
        ofv=ofv,
        minimization_successful=True,  # FIXME: parse minimization status
        parameter_estimates=pe,
        predictions=predictions,
        log=Log(),
    )
    return res


def execute_model(model: pharmpy.model.Model, db: str, evaluate=False) -> pharmpy.model.Model:
    """
    Executes a model using nlmixr2 estimation.

    Parameters
    ----------
    model : pharmpy.model.Model
        An pharmpy model object.
    db : str
        Name of folder in home directory to store resulting files in.

    Returns
    -------
    model : pharmpy.model.Model
        Model with accompanied results.

    """
    if evaluate:
        if [s.evaluation for s in model.estimation_steps._steps][0] is False:
            model = set_evaluation_step(model)

    db = pharmpy.workflows.LocalDirectoryToolDatabase(db)
    database = db.model_database
    model = convert_model(model)
    path = Path.cwd() / f'nlmixr_run_{model.name}-{uuid.uuid1()}'
    model.internals.path = path
    meta = path / '.pharmpy'
    meta.mkdir(parents=True, exist_ok=True)
    if model.datainfo.path is not None:
        model = model.replace(datainfo=model.datainfo.replace(path=None))
    write_csv(model, path=path)
    model = model.replace(datainfo=model.datainfo.replace(path=path))

    dataname = f'{model.name}.csv'
    pre = f'library(nlmixr2)\n\ndataset <- read.csv("{path / dataname}")\n'

    if "fix_eta" in model.estimation_steps[0].tool_options:
        write_fix_eta(model, path=path)
        pre += f'etas <- as.matrix(read.csv("{path}/fix_eta.csv"))'
    pre += "\n"

    code = pre + model.model_code
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
    model = model.replace(modelfit_results=res)
    return model


def verification(
    model: pharmpy.model.Model,
    db_name: str,
    error: float = 10**-3,
    return_comp: bool = False,
    fix_eta: bool = True,
    force_ipred: bool = False,
    force_pred: bool = False,
    ignore_print=False,
) -> Union[bool, pd.DataFrame]:
    """
    Verify that a model inputet in NONMEM format can be correctly translated to
    nlmixr as well as verify that the predictions of the two models are the same
    given a user specified error margin (defailt is 0.001).

    Comparison will be done on PRED values as default unless only IPRED values
    are present or if force_ipred is set to True. Can also force to use pred values
    which will cause an error if only ipred values are present.

    Parameters
    ----------
    model : pharmpy.model.Model
        pharmpy Model object in NONMEM format.
    db_name : str
        a string with given name of database folder for created files.
    error : float, optional
        Allowed error margins for predictions. The default is 10**-3.
    return_comp : bool, optional
        Choose to return table of predictions. The default is False.
    fix_eta : bool, optional
        Decide if NONMEM estimated ETAs are to be used. The default is True.
    force_ipred : bool, optional
        Force to use IPRED for calculating differences instead of PRED. The default is False.
    force_pred : bool, optional
        Force to use PRED for calculating differences. The default is False.

    Returns
    -------
    Union[bool, pd.Dataframe]
        If return_comp = True, return a table of comparisons and differences in
        predictions instead of a boolean indicating if they are the same or not

    """

    nonmem_model = model

    # Save results from the nonmem model
    if nonmem_model.modelfit_results is None:
        if not ignore_print:
            print_step("Calculating NONMEM predictions... (this might take a while)")
        nonmem_model = nonmem_model.replace(modelfit_results=fit(nonmem_model))
    else:
        if nonmem_model.modelfit_results.predictions is None:
            if not ignore_print:
                print_step("Calculating NONMEM predictions... (this might take a while)")
            nonmem_model = nonmem_model.replace(modelfit_results=fit(nonmem_model))

    # Set a tool option to fix theta values when running nlmixr
    if fix_eta:
        nonmem_model = fixate_eta(nonmem_model)

    # Check that evaluation step is set to True
    if [s.evaluation for s in nonmem_model.estimation_steps._steps][0] is False:
        nonmem_model = set_evaluation_step(nonmem_model)

    # Update the nonmem model with new estimates
    # and convert to nlmixr
    if not ignore_print:
        print_step("Converting NONMEM model to nlmixr2...")
    if fix_eta is True:
        nlmixr_model = convert_model(
            update_inits(nonmem_model, nonmem_model.modelfit_results.parameter_estimates),
            keep_etas=True,
        )
    else:
        nlmixr_model = convert_model(
            update_inits(nonmem_model, nonmem_model.modelfit_results.parameter_estimates)
        )

    # Execute the nlmixr model
    if not ignore_print:
        print_step("Executing nlmixr2 model... (this might take a while)")

    nlmixr_model = execute_model(nlmixr_model, db_name)

    # Combine the two based on ID and time
    if not ignore_print:
        print_step("Creating result comparison table...")
    combined_result = compare_models(
        nonmem_model,
        nlmixr_model,
        error=error,
        force_ipred=force_ipred,
        force_pred=force_pred,
        ignore_print=ignore_print,
    )

    if not ignore_print:
        print_step("DONE")
    if return_comp is True:
        return combined_result
    else:
        if all(combined_result["PASS/FAIL"] == "PASS"):
            return True
        else:
            return False


def compare_models(
    model_1, model_2, error=10**-3, force_ipred=False, force_pred=False, ignore_print=False
):
    assert model_1.modelfit_results.predictions is not None
    assert model_2.modelfit_results.predictions is not None

    mod1 = model_1
    mod1_type = str(type(mod1)).split(".")[2]
    mod2 = model_2
    mod2_type = str(type(mod2)).split(".")[2]

    nm_to_r = False
    if (
        mod1_type == "nonmem"
        and mod2_type != "nonmem"
        or mod2_type == "nonmem"
        and mod1_type != "nonmem"
    ):
        nm_to_r = True

    mod1 = mod1.replace(dataset=mod1.dataset.reset_index())
    mod2 = mod2.replace(dataset=mod2.dataset.reset_index())

    for mod in [mod1, mod2]:
        if "EVID" not in mod.dataset.columns:
            mod = add_evid(mod)

    if nm_to_r:
        if mod1_type == "nonmem":
            predictions = mod1.modelfit_results.predictions.reset_index()
            predictions = predictions.drop(
                mod1.dataset[~mod1.dataset["EVID"].isin([0, 2])].index.to_list()
            )
            predictions = predictions.set_index(["ID", "TIME"])
            mod1 = mod1.replace(modelfit_results=ModelfitResults(predictions=predictions))
        if mod2_type == "nonmem":
            predictions = mod2.modelfit_results.predictions.reset_index()
            predictions = predictions.drop(
                mod2.dataset[~mod2.dataset["EVID"].isin([0, 2])].index.to_list()
            )
            predictions = predictions.set_index(["ID", "TIME"])
            mod2 = mod2.replace(modelfit_results=ModelfitResults(predictions=predictions))

    mod1_results = mod1.modelfit_results.predictions.copy()

    mod2_results = mod2.modelfit_results.predictions.copy()

    pred = False
    ipred = False
    # ---
    if force_pred:
        if "PRED" in mod1_results.columns:
            pred = True
            p = "PRED"
            assert p in mod2_results.columns
            mod1_results.rename(columns={p: f'PRED_{mod1_type}'}, inplace=True)
            mod2_results.rename(columns={p: f'PRED_{mod2_type}'}, inplace=True)
    elif force_ipred:
        if "IPRED" in mod1_results.columns:
            p = "IPRED"
            ipred = True
            assert p in mod2_results.columns
            mod1_results.rename(columns={p: f'{p}_{mod1_type}'}, inplace=True)
            mod2_results.rename(columns={p: f'IPRED_{mod2_type}'}, inplace=True)
        elif "CIPREDI" in mod1_results.columns:
            p = "CIPREDI"
            ipred = True
            assert p in mod2_results.columns
            mod1_results.rename(columns={p: f'{p}_{mod1_type}'}, inplace=True)
            mod2_results.rename(columns={p: f'IPRED_{mod2_type}'}, inplace=True)
    else:
        for p in mod1_results.columns:
            if p == "PRED":
                pred = True
                assert p in mod2_results.columns
                mod1_results.rename(columns={p: f'PRED_{mod1_type}'}, inplace=True)
                mod2_results.rename(columns={p: f'PRED_{mod2_type}'}, inplace=True)
            elif (p == "IPRED" or p == "CIPREDI") and "PRED" not in mod1_results.columns:
                ipred = True
                assert "IPRED" in mod2_results.columns or "CIPREDI" in mod2_results.columns
                mod1_results.rename(columns={p: f'{p}_{mod1_type}'}, inplace=True)
                if "IPRED" in mod2_results.columns:
                    mod2_results.rename(columns={p: f'IPRED_{mod2_type}'}, inplace=True)
                else:
                    mod2_results.rename(columns={p: f'CIPREDI_{mod2_type}'}, inplace=True)
    # ---

    if not (pred or ipred):
        print("No comparable prediction value was found. Please use 'PRED' or 'IPRED")
        return False

    combined_result = mod1_results
    if pred:
        combined_result[f'PRED_{mod2_type}'] = mod2_results[f'PRED_{mod2_type}'].to_list()
        # Add difference between the models
        combined_result['PRED_DIFF'] = abs(
            combined_result[f'PRED_{mod1_type}'] - combined_result[f'PRED_{mod2_type}']
        )
    if ipred:
        combined_result[f'IPRED_{mod2_type}'] = mod2_results[f'IPRED_{mod2_type}'].to_list()
        combined_result['IPRED_DIFF'] = abs(
            combined_result[f'IPRED_{mod1_type}'] - combined_result[f'IPRED_{mod2_type}']
        )

    combined_result["PASS/FAIL"] = "PASS"
    if not ignore_print:
        print("Differences in population predicted values")
    if (pred and ipred) or (pred and not ipred):
        if force_ipred:
            if not ignore_print:
                print("Using IPRED values for final comparison")
            final = "IPRED"
        else:
            if not ignore_print:
                print("Using PRED values for final comparison")
            final = "PRED"
    elif ipred and not pred:
        if force_ipred:
            if not ignore_print:
                print("Using IPRED values for final comparison")
        else:
            if not ignore_print:
                print("Using IPRED values instead")
        final = "IPRED"

    combined_result.loc[combined_result[f'{final}_DIFF'] > error, "PASS/FAIL"] = "FAIL"
    if not ignore_print:
        print(
            combined_result[f'{final}_DIFF'].describe()[["mean", "75%", "max"]].to_string(),
            end="\n\n",
        )

    return combined_result


def print_step(s: str) -> None:
    """
    Print step currently being performed. Used during verification

    Parameters
    ----------
    s : str
        Information to print.

    See also
    --------
    verification : verify conversion of model to nlmixr2

    """
    print("***** ", s, " *****")


def fixate_eta(model: pharmpy.model.Model) -> pharmpy.model.Model:
    """
    Used during verification to give information to model to fixate etas to
    NONMEM estimates. Add the information to the models tool_options

    Parameters
    ----------
    model : pharmpy.model.Model
        An nlmixr2 pharmpy model object to fixate etas for.

    Returns
    -------
    model : TYPE
        Model with modified tool options to fixate etas during verification.

    See also
    --------
    verification : verify conversion of model to nlmixr2

    """
    opts = {"fix_eta": True}
    model = append_estimation_step_options(model, tool_options=opts, idx=0)
    return model


def write_fix_eta(model: pharmpy.model.Model, path=None, force=True) -> str:
    """
    Writes ETAs to be fixated during verification to a csv file to be read by
    nlmixr2

    Parameters
    ----------
    model : pharmpy.model.Model
        A pharmpy model object.
    path : TYPE, optional
        Path to write csv file to. The default is None.
    force : TYPE, optional
        Force overwrite the file if exist. The default is True.

    Raises
    ------
    FileExistsError
        If csv file exist and force is False, raise error.

    Returns
    -------
    str
        Return the path to fixated ETAs csv file.

    """
    from pharmpy.internals.fs.path import path_absolute
    from pharmpy.model import data

    filename = "fix_eta.csv"
    path = path / filename
    if not force and path.exists():
        raise FileExistsError(f'File at {path} already exists.')

    path = path_absolute(path)
    model.modelfit_results.individual_estimates.to_csv(path, na_rep=data.conf.na_rep, index=False)
    return path


def verify_param(model1, model2, est=False):
    tol = 0.01

    if est:
        param1 = model1.modelfit_results.parameter_estimates
        param2 = model2.modelfit_results.parameter_estimates

        passed = []
        failed = []
        for p1 in param1.index:
            if p1 in param2.index:
                p1_value = param1[p1]
                p2_value = param2[p1]
                diff = p1_value - p2_value
                if abs(diff) > tol:
                    failed.append((p1, diff))
                else:
                    passed.append((p1, diff))

    else:
        param1 = model1.parameters
        param2 = model2.parameters

        passed = []
        failed = []
        for p1 in param1:
            if p1.name in param2.names:
                p2 = param2[p1.name]
                diff = p1.init - p2.init
                if abs(diff) > tol:
                    failed.append((p1.name, diff))
                else:
                    passed.append((p1.name, diff))
    return passed, failed
