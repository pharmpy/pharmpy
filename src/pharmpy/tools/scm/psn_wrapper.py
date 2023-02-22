import os
import re
import subprocess

from pharmpy.modeling import read_model, write_csv, write_model
from pharmpy.tools import read_results
from pharmpy.workflows import default_tool_database


def have_scm():
    # Check if scm is available through PsN
    try:
        out = subprocess.run(["scm", "--version"], capture_output=True)
    except FileNotFoundError:
        return False

    m = re.match(r"PsN version: (\d+)\.(\d+)\.(\d+)", out.stdout.decode("utf-8"))
    if m:
        major = int(m.group(1))
        minor = int(m.group(2))
        patch = int(m.group(3))
        return (
            major > 5 or (major == 5 and minor > 2) or (major == 5 and minor == 2 and patch >= 10)
        )
    else:
        return False


def run_scm(model, relations, continuous=None, categorical=None, path=None):
    if continuous is None:
        continuous = []
    if categorical is None:
        categorical = []

    if path is not None:
        path = path / 'scm'

    db = default_tool_database(toolname='scm', path=path)
    path = db.path / "psn-wrapper"
    path.mkdir()

    config = _create_config(model, path, relations, continuous, categorical)
    config_path = path / "config.scm"
    with open(config_path, 'w') as fh:
        fh.write(config)

    write_csv(model, path=path)
    write_model(model, path=path)

    os.system(f"scm {config_path} -directory={path / 'scmdir'} -auto_tv")

    res = read_results(path / 'scmdir' / 'results.json')
    final_model_path = path / 'scmdir' / 'final_models' / 'final_forward.mod'
    if final_model_path.is_file():
        final_model = read_model(final_model_path)
        res.final_model = final_model
    return res


def _create_config(model, path, relations, continuous, categorical):
    config = f"model={path}/{model.name}{model.filename_extension}\n"
    config += "search_direction=forward\n"
    config += "p_forward=0.01\n"
    if continuous:
        config += f"continuous_covariates={','.join(continuous)}\n"
    if categorical:
        config += f"categorical_covariates={','.join(categorical)}\n"
    config += f"do_not_drop={','.join(continuous + categorical)}\n"
    config += "\n"
    config += "[test_relations]\n"
    for param, covs in relations.items():
        config += f"{param}={','.join(covs)}\n"
    config += "\n"
    config += "[valid_states]\n"
    config += "continuous=1,4\n"
    config += "categorical=1,2\n"
    return config
