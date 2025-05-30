import os
import os.path
import shutil
import subprocess
import time
import uuid
import warnings
from itertools import repeat
from pathlib import Path
from tempfile import mkdtemp
from typing import Optional

import pharmpy.config as config
from pharmpy.model.external.nonmem import convert_model
from pharmpy.modeling import get_config_path, write_csv, write_model
from pharmpy.tools.external.nonmem import conf, parse_modelfit_results, parse_simulation_results
from pharmpy.workflows import ModelEntry

from .parafile import create_parafile

PARENT_DIR = f'..{os.path.sep}'


def execute_model(model_entry, context):
    assert isinstance(model_entry, ModelEntry)
    model = model_entry.model

    database = context.model_database
    model = convert_model(model)
    path = Path.cwd() / f'NONMEM_run_{model.name}-{uuid.uuid1()}'

    if len(context.dispatcher.get_hosts()) > 1 and context.get_ncores_for_execution() > 1:
        tmp_dir = Path.home() / '.tmp'
        if not tmp_dir.is_dir():
            tmp_dir.mkdir()
        tmp_path = mkdtemp(dir=tmp_dir)
    else:
        tmp_path = None

    # NOTE: This deduplicates the dataset before running NONMEM, so we know which
    # filename to give to this dataset.
    database.store_model(model)
    # NOTE: We cannot reuse model_with_correct_datapath as the model object
    # later because it might have lost some of the ETA names mapping due to the
    # current incomplete implementation of serialization of Pharmpy Model
    # objects through the NONMEM plugin. Hopefully we can get rid of this
    # hack later.
    model_with_correct_datapath = database.retrieve_model(model)
    stream = model_with_correct_datapath.internals.control_stream
    data_record = stream.get_records('DATA')[0]
    relative_dataset_path = data_record.filename

    # NOTE: We set up a directory tree that replicates the structure generated by
    # the database so that NONMEM writes down the correct relative paths in
    # generated files such as results.lst.
    # NOTE: It is important that we do this in a DB-agnostic way so that we do
    # not depent on its implementation.
    depth = relative_dataset_path.count(PARENT_DIR)
    # NOTE: This creates an FS tree branch x/x/x/x/...
    model_path = path.joinpath(*repeat('x', depth))
    meta = model_path / '.pharmpy'
    meta.mkdir(parents=True, exist_ok=True)
    # NOTE: This removes the leading ../
    relative_dataset_path_suffix = relative_dataset_path[len(PARENT_DIR) * depth :]
    # NOTE: We do not support non-leading ../, e.g. a/b/../c
    assert PARENT_DIR not in relative_dataset_path_suffix
    dataset_path = path / Path(relative_dataset_path_suffix)
    datasets_path = dataset_path.parent
    datasets_path.mkdir(parents=True, exist_ok=True)

    # NOTE: Write dataset and model files so they can be used by NONMEM.
    model = write_csv(model, path=dataset_path, force=True)
    model = write_model(model, path=model_path / "model.ctl", force=True)

    parafile_option = create_parafile_and_option(context, model_path / 'parafile.pnm', tmp_path)
    try:
        args = nmfe("model.ctl", "model.lst", parafile_option)
    except FileNotFoundError as e:
        context.abort_workflow(str(e))

    stdout = model_path / 'stdout'
    stderr = model_path / 'stderr'

    with open(stdout, "wb") as out, open(stderr, "wb") as err:
        result = subprocess.run(
            args, stdin=subprocess.DEVNULL, stderr=err, stdout=out, cwd=str(model_path)
        )

    basename = Path("model")

    results_path = model_path / 'model.lst'
    start = time.time()
    timeout = 5

    while not results_path.is_file():
        elapsed_time = time.time() - start
        if elapsed_time >= timeout:
            context.log_warning(
                f'UNEXPECTED Could not find .lst-file after waiting {elapsed_time}s',
                model_entry.model,
            )
            break
        else:
            time.sleep(1)

    metadata = {
        'plugin': 'nonmem',
        'path': str(model_path),
    }

    if model.internals.control_stream.get_records('ESTIMATION'):
        modelfit_results = parse_modelfit_results(model, model_path / basename)
    else:
        modelfit_results = None
    if model.internals.control_stream.get_records('SIMULATION'):
        simulation_results = parse_simulation_results(model, model_path / basename)
    else:
        simulation_results = None

    log = modelfit_results.log if modelfit_results else None
    model_entry = model_entry.attach_results(
        modelfit_results=modelfit_results, simulation_results=simulation_results, log=log
    )

    with database.transaction(model_entry) as txn:
        if (
            not (model_path / basename).with_suffix('.lst').is_file()
            or not (model_path / basename).with_suffix('.ext').is_file()
        ):
            context.log_warning(
                'Expected result files do not exist, copying everything', model=model_entry.model
            )
            for file in path.glob('x/*'):
                txn.store_local_file(file)
        else:
            for suffix in ('.lst', '.ext', '.phi', '.cov', '.cor', '.coi', '.grd', '.ets'):
                file_path = (model_path / basename).with_suffix(suffix)
                txn.store_local_file(file_path)

            for rec in model.internals.control_stream.get_records('TABLE'):
                txn.store_local_file(model_path / rec.path)

        txn.store_local_file(stdout)
        txn.store_local_file(stderr)

        commands = [
            {
                'args': args,
                'returncode': result.returncode,
                'stdout': 'stdout',
                'stderr': 'stderr',
            }
        ]
        metadata['commands'] = commands

        txn.store_metadata(metadata)

    context.store_model_entry(model_entry)

    return model_entry


def nmfe_path():
    if os.name == 'nt':
        nmfe_candidates = ('nmfe76.bat', 'nmfe75.bat', 'nmfe74.bat', 'nmfe73.bat')
    else:
        nmfe_candidates = ('nmfe76', 'nmfe75', 'nmfe74', 'nmfe73')
    default_path = conf.default_nonmem_path
    if default_path != Path(''):
        path = default_path
        if path.is_dir():
            if path.name != 'run':
                path = path / 'run'
            for nmfe in nmfe_candidates:
                candidate_path = path / nmfe
                if candidate_path.is_file():
                    path = candidate_path
                    break
    else:
        # Not in configuration file
        for nmfe in nmfe_candidates:
            candidate_path = shutil.which(nmfe)
            if candidate_path is not None:
                path = Path(candidate_path)
                break
        else:
            with warnings.catch_warnings():
                conf_path = get_config_path()
                if conf_path:
                    raise FileNotFoundError(
                        'No path to NONMEM configured in pharmpy.conf and nmfe not in PATH'
                    )
                else:
                    raise FileNotFoundError(
                        f'Cannot find pharmpy.conf: {config.user_config_path()}'
                    )

    if not path.is_file():
        raise FileNotFoundError(f'Cannot find nmfe script for NONMEM ({default_path})')

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


def create_parafile_and_option(context, path: Path, tmp_path: Optional[Path]) -> str:
    ncores = context.get_ncores_for_execution()
    if ncores > 1:
        nodedict = context.dispatcher.get_hosts()
        if context.dispatcher.get_hostname() == 'localhost':
            nodedict['localhost'] = ncores
        create_parafile(path, nodedict, tmp_path)
        return f"-parafile={path.name}"
    else:
        return ""
