import os

from pharmpy.modeling import read_results, write_model
from pharmpy.workflows import default_tool_database


def run_scm(model, relations, continuous=None, categorical=None):
    if continuous is None:
        continuous = []
    if categorical is None:
        categorical = []

    db = default_tool_database(toolname='scm')
    path = db.path / "psn-wrapper"
    path.mkdir()

    config = _create_config(model, path, relations, continuous, categorical)
    config_path = path / "config.scm"
    with open(config_path, 'w') as fh:
        fh.write(config)

    model._dataset_updated = True  # Hack to get update_source to update IGNORE
    write_model(model, path=path)

    os.system(f"scm {config_path} -directory={path / 'scmdir'}")

    res = read_results(path / 'scmdir' / 'results.json')
    return res


def _create_config(model, path, relations, continuous, categorical):
    config = f"model={path}/{model.name}{model.filename_extension}\n"
    config += "search_direction=both\n"
    config += "p_forward=0.05\n"
    config += "p_backward=0.01\n"
    config += f"continuous_covariates={'.'.join(continuous)}\n"
    config += f"categorical_covariates={'.'.join(categorical)}\n"
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
