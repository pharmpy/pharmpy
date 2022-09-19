import importlib
import inspect
import os
from pathlib import Path

import pytest

import pharmpy
from pharmpy.results import read_results
from pharmpy.tools.run import (
    _create_metadata_common,
    _create_metadata_tool,
    _get_run_setup,
    retrieve_final_model,
    retrieve_models,
)
from pharmpy.utils import TemporaryDirectoryChanger
from pharmpy.workflows import LocalDirectoryToolDatabase, local_dask


def test_create_metadata_tool():
    name = 'modelsearch'
    tool = importlib.import_module(f'pharmpy.tools.{name}')
    tool_params = inspect.signature(tool.create_workflow).parameters

    metadata = _create_metadata_tool(
        tool_name=name,
        tool_params=tool_params,
        tool_options={'iiv_strategy': 0},
        args=('ABSORPTION(ZO)', 'exhaustive'),
    )

    assert metadata['pharmpy_version'] == pharmpy.__version__
    assert metadata['tool_name'] == 'modelsearch'
    assert metadata['tool_options']['rank_type'] == 'bic'

    metadata = _create_metadata_tool(
        tool_name=name,
        tool_params=tool_params,
        tool_options={'algorithm': 'exhaustive'},
        args=('ABSORPTION(ZO)',),
    )

    assert metadata['tool_name'] == 'modelsearch'
    assert metadata['tool_options']['algorithm'] == 'exhaustive'

    with pytest.raises(Exception, match='modelsearch: \'algorithm\' was not set'):
        _create_metadata_tool(
            tool_name=name,
            tool_params=tool_params,
            tool_options={},
            args=('ABSORPTION(ZO)',),
        )


def test_get_run_setup(tmp_path):
    with TemporaryDirectoryChanger(tmp_path):
        name = 'modelsearch'

        dispatcher, database = _get_run_setup(common_options={}, toolname=name)

        assert dispatcher.__name__.endswith('local_dask')
        assert database.path.stem == 'modelsearch_dir1'

        dispatcher, database = _get_run_setup(
            common_options={'path': 'tool_database_path'}, toolname=name
        )

        assert dispatcher.__name__.endswith('local_dask')
        assert database.path.stem == 'tool_database_path'


def test_create_metadata_common(tmp_path):
    with TemporaryDirectoryChanger(tmp_path):
        name = 'modelsearch'

        dispatcher = local_dask
        database = LocalDirectoryToolDatabase(name)

        metadata = _create_metadata_common(
            common_options={}, dispatcher=dispatcher, database=database, toolname=name
        )

        assert metadata['dispatcher'] == 'pharmpy.workflows.dispatchers.local_dask'
        assert metadata['database']['class'] == 'LocalDirectoryToolDatabase'
        path = Path(metadata['database']['path'])
        assert path.stem == 'modelsearch_dir1'
        assert 'path' not in metadata.keys()

        path = 'tool_database_path'

        dispatcher = local_dask
        database = LocalDirectoryToolDatabase(name, path)

        metadata = _create_metadata_common(
            common_options={'path': path}, dispatcher=dispatcher, database=database, toolname=name
        )

        path = Path(metadata['database']['path'])
        assert path.stem == 'tool_database_path'
        assert metadata['path'] == 'tool_database_path'


def test_retrieve_models(testdata):
    tool_database_path = testdata / 'results' / 'tool_databases' / 'modelsearch'

    model_to_retrieve = ['modelsearch_candidate1']

    models = retrieve_models(tool_database_path, names=model_to_retrieve)
    assert len(models) == 1
    assert models[0].name == model_to_retrieve[0]

    model_names_all = [
        'input_model',
        'modelsearch_candidate1',
        'modelsearch_candidate2',
        'modelsearch_candidate3',
        'modelsearch_candidate4',
    ]

    models = retrieve_models(tool_database_path)
    assert [model.name for model in models] == model_names_all

    with open(tool_database_path / 'results.json') as f:
        results_json = f.read()
        if os.name == 'nt':
            new_path = str(tool_database_path).replace('\\', '\\\\')
        else:
            new_path = str(tool_database_path)

        results_json_testpath = results_json.replace('/tmp/tool_results/modelsearch', new_path)

    res = read_results(results_json_testpath)
    models = retrieve_models(res, names=model_to_retrieve)
    assert models[0].name == model_to_retrieve[0]

    tool_db = LocalDirectoryToolDatabase('modelsearch', path=tool_database_path, exist_ok=True)
    models = retrieve_models(tool_db, names=model_to_retrieve)
    assert models[0].name == model_to_retrieve[0]

    models = retrieve_models(tool_db.model_database, names=model_to_retrieve)
    assert models[0].name == model_to_retrieve[0]

    with pytest.raises(ValueError, match='Models {\'x\'} not in database'):
        retrieve_models(res, names=['x'])

    tool_without_db_in_results = testdata / 'results' / 'qa_results.json'
    res = read_results(tool_without_db_in_results)
    with pytest.raises(
        ValueError, match='Results type \'QAResults\' does not serialize tool database'
    ):
        retrieve_models(res, names=model_to_retrieve)


def test_retrieve_final_model(testdata):
    tool_database_path = testdata / 'results' / 'tool_databases' / 'modelsearch'

    with open(tool_database_path / 'results.json') as f:
        results_json = f.read()
        if os.name == 'nt':
            new_path = str(tool_database_path).replace('\\', '\\\\')
        else:
            new_path = str(tool_database_path)

        results_json_testpath = results_json.replace('/tmp/tool_results/modelsearch', new_path)

    res = read_results(results_json_testpath)
    final_model = retrieve_final_model(res)
    assert final_model.name == 'modelsearch_candidate2'

    results_json_none = results_json_testpath.replace(
        '"final_model_name": "modelsearch_candidate2"', '"final_model_name": null'
    )
    res = read_results(results_json_none)
    with pytest.raises(ValueError, match='Attribute \'final_model_name\' is None'):
        retrieve_final_model(res)
