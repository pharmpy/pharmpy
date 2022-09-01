import json
import os
import subprocess
import uuid
from pathlib import Path

from pharmpy.estimation import EstimationSteps
from pharmpy.modeling import write_csv, write_model
from pharmpy.plugins.nonmem import conf, convert_model


def execute_model(model):
    database = model.database
    parent_model = model.parent_model
    model = convert_model(model)
    path = Path.cwd() / f'NONMEM_run_{model.name}-{uuid.uuid1()}'
    meta = path / '.pharmpy'
    meta.mkdir(parents=True, exist_ok=True)
    model = model.copy()
    model.parent_model = parent_model
    write_csv(model, path=path, force=True)
    # Set local path
    model.datainfo = model.datainfo.derive(
        path=model.datainfo.path.relative_to(model.datainfo.path.parent)
    )
    model._dataset_updated = True  # Hack to get update_source to update IGNORE
    write_model(model, path=path, force=True)
    basepath = Path(model.name)
    args = nmfe(
        model.name + model.filename_extension,
        'results.lst',
    )

    stdout = path / 'stdout'
    stderr = path / 'stderr'

    with open(stdout, "wb") as out, open(stderr, "wb") as err:
        result = subprocess.run(
            args, stdin=subprocess.DEVNULL, stderr=err, stdout=out, cwd=str(path)
        )

    metadata = {
        'plugin': 'nonmem',
        'path': str(path),
    }

    plugin = {
        'commands': [
            {
                'args': args,
                'returncode': result.returncode,
            }
        ]
    }

    with database.transaction(model) as txn:

        txn.store_model()
        txn.store_local_file((path / 'results.lst'), new_filename=basepath.with_suffix('.lst'))
        txn.store_local_file((path / basepath).with_suffix('.ext'))
        txn.store_local_file((path / basepath).with_suffix('.phi'))
        txn.store_local_file((path / basepath).with_suffix('.cov'))
        txn.store_local_file((path / basepath).with_suffix('.cor'))
        txn.store_local_file((path / basepath).with_suffix('.coi'))

        for rec in model.control_stream.get_records('TABLE'):
            txn.store_local_file(path / rec.path)

        txn.store_local_file(stdout)
        txn.store_local_file(stderr)

        plugin_path = path / 'nonmem.json'
        with open(plugin_path, 'w') as f:
            json.dump(plugin, f, indent=2)

        txn.store_local_file(plugin_path)

        txn.store_metadata(metadata)
        if len(model.estimation_steps) > 0:
            txn.store_modelfit_results()

            # Read in results for the server side
            # FIXME: this breaks through abstraction
            model.read_modelfit_results(database.path / model.name)

    # FIXME: the database path is changed in write
    model.database = database

    return model


def nmfe_path():
    if os.name == 'nt':
        nmfe_candidates = ['nmfe74.bat', 'nmfe75.bat', 'nmfe73.bat']
    else:
        nmfe_candidates = ['nmfe74', 'nmfe75', 'nmfe73']
    path = conf.default_nonmem_path
    if path != Path(''):
        path /= 'run'
    for nmfe in nmfe_candidates:
        candidate_path = path / nmfe
        if candidate_path.is_file():
            path = candidate_path
            break
    else:
        raise FileNotFoundError(f'Cannot find nmfe script for NONMEM ({path})')
    return str(path)


def nmfe(*args):
    conf_args = []
    if conf.licfile is not None:
        conf_args.append(f'-licfile={str(conf.licfile)}')

    return [
        nmfe_path(),
        *args,
        *conf_args,
    ]


def evaluate_design(model):
    # Prepare and run model for design evaluation
    model = model.copy()
    model.name = '_design_model'

    model.estimation_steps = EstimationSteps()
    estrecs = model.control_stream.get_records('ESTIMATION')
    model.control_stream.remove_records(estrecs)

    design_code = '$DESIGN APPROX=FOCEI MODE=1 NELDER FIMDIAG=0 DATASIM=1 GROUPSIZE=32 OFVTYPE=0'
    model.control_stream.insert_record(design_code)

    execute_model(model)

    from pharmpy.tools.evaldesign import EvalDesignResults

    res = EvalDesignResults(
        ofv=model.modelfit_results.ofv,
        individual_ofv=model.modelfit_results.individual_ofv,
        information_matrix=model.modelfit_results.information_matrix,
    )
    return res
