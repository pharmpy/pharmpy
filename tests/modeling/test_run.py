import importlib
import inspect
from pathlib import Path

import pytest

from pharmpy.modeling.run import _create_metadata_common, _create_metadata_tool, _get_run_setup
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

    assert metadata['tool_name'] == 'modelsearch'
    assert metadata['tool_options']['rankfunc'] == 'bic'

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
