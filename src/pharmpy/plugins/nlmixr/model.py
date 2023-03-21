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

from .ini import add_eta, add_sigma, add_theta
from .model_block import add_ode, add_statements
from .sanity_checks import check_model, print_warning


def convert_model(
    model: pharmpy.model.Model, keep_etas: bool = False, skip_check: bool = False
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

    if model.internals.control_stream.get_records("DES"):
        des = model.internals.control_stream.get_records("DES")[0]
    else:
        des = None

    nlmixr_model = Model(
        internals=NLMIXRModelInternals(DES=des),
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
    if model.dataset is not None or len(model.dataset) != 0:
        if keep_etas is True:
            nlmixr_model = nlmixr_model.replace(
                modelfit_results=ModelfitResults(
                    individual_estimates=model.modelfit_results.individual_estimates
                )
            )
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
    cg.indent()

    add_statements(model, cg, model.statements.before_odes)

    if model.statements.ode_system:
        add_ode(model, cg)

    add_statements(model, cg, model.statements.after_odes)

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
    DES: Optional = None


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
        # Replace all instances of EPS with sigma instead
        for eps in self.random_variables.epsilons:
            code = code.replace(eps.names[0], eps.variance.name)
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


def execute_model(model: pharmpy.model.Model, db: str) -> pharmpy.model.Model:
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
    database = db.model_database
    model = convert_model(model)
    path = Path.cwd() / f'nlmixr_run_{model.name}-{uuid.uuid1()}'
    model.internals.path = path
    meta = path / '.pharmpy'
    meta.mkdir(parents=True, exist_ok=True)
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
    ipred_diff: bool = False,
) -> Union[bool, pd.DataFrame]:
    """
    Verify that a model inputet in NONMEM format can be correctly translated to
    nlmixr as well as verify that the predictions of the two models are the same
    given a user specified error margin (defailt is 0.001).

    Comparison will be done on PRED values as default unless only IPRED values
    are present or if ipred_diff is set to True.

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
    ipred_diff : bool, optional
        Force to use IPRED for calculating differences instead of PRED. The default is False.

    Returns
    -------
    Union[bool, pd.Dataframe]
        If return_comp = True, return a table of comparisons and differences in
        predictions instead of a boolean indicating if they are the same or not

    """

    nonmem_model = model

    # Save results from the nonmem model
    if nonmem_model.modelfit_results is None:
        print_step("Calculating NONMEM predictions... (this might take a while)")
        nonmem_model = nonmem_model.replace(modelfit_results=fit(nonmem_model))
        nonmem_results = nonmem_model.modelfit_results.predictions.copy()
    else:
        if nonmem_model.modelfit_results.predictions is None:
            print_step("Calculating NONMEM predictions... (this might take a while)")
            nonmem_model = nonmem_model.replace(modelfit_results=fit(nonmem_model))
            nonmem_results = nonmem_model.modelfit_results.predictions.copy()
        else:
            nonmem_results = nonmem_model.modelfit_results.predictions.copy()

    # Set a tool option to fix theta values when running nlmixr
    if fix_eta:
        nonmem_model = fixate_eta(nonmem_model)

    # Check that evaluation step is set to True
    if [s.evaluation for s in nonmem_model.estimation_steps._steps][0] is False:
        nonmem_model = set_evaluation_step(nonmem_model)

    # Update the nonmem model with new estimates
    # and convert to nlmixr
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
    print_step("Executing nlmixr2 model... (this might take a while)")
    import pharmpy.workflows

    db = pharmpy.workflows.LocalDirectoryToolDatabase(db_name)
    nlmixr_model = execute_model(nlmixr_model, db)
    nlmixr_results = nlmixr_model.modelfit_results.predictions

    pred = False
    ipred = False
    for p in nonmem_model.modelfit_results.predictions.columns:
        if p == "PRED":
            pred = True
            nonmem_results.rename(columns={p: 'PRED_NONMEM'}, inplace=True)
            nlmixr_results.rename(columns={p: 'PRED_NLMIXR'}, inplace=True)
        elif p == "IPRED":
            ipred = True
            nonmem_results.rename(columns={p: 'IPRED_NONMEM'}, inplace=True)
            nlmixr_results.rename(columns={p: 'IPRED_NLMIXR'}, inplace=True)
        else:
            print(
                f"Unknown prediction value {p}. Currently only 'PRED' and 'IPRED' are supported and this is ignored"
            )

    if not (pred or ipred):
        print("No known prediction value was found. Please use 'PRED' or 'IPRED")
        return False

    # Combine the two based on ID and time
    print_step("Creating result comparison table...")
    nonmem_model = nonmem_model.replace(dataset=nonmem_model.dataset.reset_index())

    if "EVID" not in nonmem_model.dataset.columns:
        nonmem_model = add_evid(nonmem_model)
    nonmem_results = nonmem_results.reset_index()
    nonmem_results = nonmem_results.drop(
        nonmem_model.dataset[nonmem_model.dataset["EVID"] != 0].index.to_list()
    )
    nonmem_results = nonmem_results.set_index(["ID", "TIME"])

    combined_result = nonmem_results
    if pred:
        combined_result['PRED_NLMIXR'] = nlmixr_results['PRED_NLMIXR'].to_list()
        # Add difference between the models
        combined_result['PRED_DIFF'] = abs(
            combined_result['PRED_NONMEM'] - combined_result['PRED_NLMIXR']
        )
    if ipred:
        combined_result['IPRED_NLMIXR'] = nlmixr_results['IPRED_NLMIXR'].to_list()
        combined_result['IPRED_DIFF'] = abs(
            combined_result['IPRED_NONMEM'] - combined_result['IPRED_NLMIXR']
        )

    combined_result["PASS/FAIL"] = "PASS"
    print("Differences in population predicted values")
    if (pred and ipred) or (pred and not ipred):
        if ipred_diff:
            print("Using PRED values for final comparison")
            final = "IPRED"
        else:
            print("Using PRED values for final comparison")
            final = "PRED"
    elif ipred and not pred:
        print("Using IPRED values for final comparison")
        final = "IPRED"
    combined_result.loc[combined_result[f'{final}_DIFF'] > error, "PASS/FAIL"] = "FAIL"
    print(
        combined_result[f'{final}_DIFF'].describe()[["mean", "75%", "max"]].to_string(), end="\n\n"
    )

    print_step("DONE")
    if return_comp is True:
        return combined_result
    else:
        if all(combined_result["PASS/FAIL"] == "PASS"):
            return True
        else:
            return False


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
