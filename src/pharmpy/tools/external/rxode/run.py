import json
import os
import subprocess
import uuid
import warnings
from pathlib import Path
from typing import Union

import pharmpy.model
from pharmpy.deps import pandas as pd
from pharmpy.internals.code_generator import CodeGenerator
from pharmpy.model.external.rxode import convert_model
from pharmpy.modeling import get_omegas, get_sigmas, write_csv
from pharmpy.results import ModelfitResults


def execute_model(model: pharmpy.model.Model, db: str) -> pharmpy.model.Model:
    """
    Executes a model using rxode2.

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
    db = pharmpy.workflows.LocalDirectoryToolDatabase(db)
    database = db.model_database
    model = convert_model(model)
    path = Path.cwd() / f'rxode_run_{model.name}-{uuid.uuid1()}'
    model.internals.path = path
    meta = path / '.pharmpy'
    meta.mkdir(parents=True, exist_ok=True)
    if model.datainfo.path is not None:
        model = model.replace(datainfo=model.datainfo.replace(path=None))
    write_csv(model, path=path)
    model = model.replace(datainfo=model.datainfo.replace(path=path))

    dataname = f'{model.name}.csv'
    pre = f'library(rxode2)\n\nev <- read.csv("{path / dataname}")\n'

    pre += "\n"

    code = pre + model.model_code
    cg = CodeGenerator()

    dv = list(model.dependent_variables.keys())[0]
    cg.add(f"res <- as.data.frame(fit[c('id', 'time', '{dv}')])")
    cg.add("sigma <- as.data.frame(sigmas)")
    cg.add("omegas <- as.data.frame(omegas)")
    cg.add("params <- as.data.frame(fit$params)")

    cg.add(f'save(file="{path}/{model.name}.RDATA", res, params)')

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
        txn.store_local_file(path / f'{model.name}.csv')

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


def parse_modelfit_results(model: pharmpy.model.Model, path: Path) -> Union[None, ModelfitResults]:
    rdata_path = path / (model.name + '.RDATA')
    with warnings.catch_warnings():
        # Supress a numpy deprecation warning
        warnings.simplefilter("ignore")
        import pyreadr
    try:
        rdata = pyreadr.read_r(rdata_path)
    except (FileNotFoundError, OSError):
        return None

    dv = list(model.dependent_variables.keys())[0]
    pred = rdata["res"][["id", "time", f"{dv}"]]
    pred.rename(columns={f"{dv}": 'PRED', "id": "ID", "time": "TIME"}, inplace=True)
    pred = pred.set_index(["ID", "TIME"])

    # TODO : extract thetas, omegas and sigmas

    predictions = pred
    predictions.index = predictions.index.set_levels(
        predictions.index.levels[0].astype("float64"), level=0
    )

    # TODO : Add more variables such as name and description and parameter estimates
    res = ModelfitResults(predictions=predictions)
    return res


def verification(
    model: pharmpy.model.Model,
    db_name: str,
    error: float = 10**-3,
    return_comp: bool = False,
    ignore_print=False,
) -> Union[bool, pd.DataFrame]:
    nonmem_model = model

    from pharmpy.modeling import update_inits
    from pharmpy.plugins.nlmixr.model import print_step
    from pharmpy.tools import fit

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

    param_estimates = nonmem_model.modelfit_results.parameter_estimates

    omega_names = get_omegas(nonmem_model).names
    for name in omega_names:
        param_estimates[name] = 0

    sigma_names = get_sigmas(model).names
    for name in sigma_names:
        param_estimates[name] = 0

    # Update the nonmem model with new estimates
    # and convert to nlmixr
    if not ignore_print:
        print_step("Converting NONMEM model to RxODE...")
    rxode_model = convert_model(update_inits(nonmem_model, param_estimates))

    # Execute the nlmixr model
    if not ignore_print:
        print_step("Executing RxODE model... (this might take a while)")

    rxode_model = execute_model(rxode_model, db_name)

    from pharmpy.plugins.nlmixr.model import compare_models

    combined_result = compare_models(
        nonmem_model, rxode_model, error=error, force_pred=True, ignore_print=ignore_print
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
