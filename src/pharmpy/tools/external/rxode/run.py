import json
import os
import subprocess
import sys
import uuid
import warnings
from pathlib import Path
from typing import Optional, Union

import pharmpy.model
from pharmpy.deps import pandas as pd
from pharmpy.internals.code_generator import CodeGenerator
from pharmpy.model.external.rxode import convert_model
from pharmpy.modeling import get_omegas, get_sigmas, set_initial_estimates, write_csv
from pharmpy.tools import fit
from pharmpy.tools.external.nlmixr.run import compare_models, print_step
from pharmpy.workflows import ModelEntry, default_context
from pharmpy.workflows.results import ModelfitResults


def execute_model(model_entry, db):
    assert isinstance(model_entry, ModelEntry)
    model = model_entry.model

    database = db.model_database
    model = convert_model(model)
    path = Path.cwd() / f'rxode_run_{model.name}-{uuid.uuid1()}'
    model = model.replace(internals=model.internals.replace(path=path))
    meta = path / '.pharmpy'
    meta.mkdir(parents=True, exist_ok=True)
    if model.datainfo.path is not None:
        model = model.replace(datainfo=model.datainfo.replace(path=None))
    write_csv(model, path=path)
    model = model.replace(datainfo=model.datainfo.replace(path=path))

    dataname = f'{model.name}.csv'
    if sys.platform == 'win32':
        dataset_path = f"{path / dataname}".replace("\\", "\\\\")
    else:
        dataset_path = f"{path / dataname}"
    pre = f'library(rxode2)\n\nev <- read.csv("{dataset_path}")\n'

    pre += "\n"

    code = pre + model.code
    cg = CodeGenerator()

    dv = list(model.dependent_variables.keys())[0]
    cg.add(f"res <- as.data.frame(fit[c('id', 'time', '{dv}')])")
    cg.add("sigma <- as.data.frame(sigmas)")
    cg.add("omegas <- as.data.frame(omegas)")
    cg.add("params <- as.data.frame(fit$params)")

    if sys.platform == 'win32':
        p = f"{path / model.name}.RDATA".replace("\\", "\\\\")
    else:
        p = f"{path / model.name}.RDATA"
    cg.add(f'save(file="{p}", res, params)')

    code += f'\n{str(cg)}'

    with open(path / f'{model.name}.R', 'w') as fh:
        fh.write(code)

    from pharmpy.tools.external.nlmixr import conf

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
    log = res.log if res else None
    model_entry = model_entry.attach_results(modelfit_results=res, log=log)

    return model_entry


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

    # TODO: Extract thetas, omegas and sigmas

    predictions = pred
    predictions.index = predictions.index.set_levels(
        predictions.index.levels[0].astype("float64"), level=0
    )

    # TODO: Add more variables such as name and description and parameter estimates
    res = ModelfitResults(predictions=predictions)
    return res


def verification(
    model: pharmpy.model.Model,
    modelfit_results: Optional[ModelfitResults] = None,
    error: float = 10**-3,
    return_comp: bool = False,
    ignore_print=False,
) -> Union[bool, pd.DataFrame]:
    nonmem_model = model

    # Save results from the nonmem model
    if modelfit_results is None:
        if not ignore_print:
            print_step("Calculating NONMEM predictions... (this might take a while)")
        nonmem_res = fit(nonmem_model)
    else:
        if modelfit_results.predictions is None:
            if not ignore_print:
                print_step("Calculating NONMEM predictions... (this might take a while)")
            nonmem_res = fit(nonmem_model)
        else:
            nonmem_res = modelfit_results

    param_estimates = nonmem_res.parameter_estimates

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
    rxode_model = convert_model(set_initial_estimates(nonmem_model, param_estimates))

    # Execute the rxode model
    db = default_context("comparison")
    if not ignore_print:
        print_step("Executing RxODE model... (this might take a while)")

    rxode_model_entry = ModelEntry(rxode_model, modelfit_results=None)
    rxode_model_entry = execute_model(rxode_model_entry, db)

    combined_result = compare_models(
        nonmem_model,
        nonmem_res,
        rxode_model,
        rxode_model_entry.modelfit_results,
        error=error,
        force_pred=True,
        ignore_print=ignore_print,
    )

    combined_result.to_csv(db.path / "comparison.csv", index=False)

    if not ignore_print:
        print_step("DONE")
    if return_comp is True:
        return combined_result
    else:
        if all(combined_result["PASS/FAIL"] == "PASS"):
            return True
        else:
            return False
