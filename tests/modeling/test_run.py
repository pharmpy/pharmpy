import importlib
import inspect

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
        tool_options={'add_iivs': True},
        args=('exhaustive', 'ABSORPTION(ZO)'),
    )

    assert metadata['tool_name'] == 'modelsearch'
    assert metadata['tool_options']['rankfunc'] == 'ofv'

    metadata = _create_metadata_tool(
        tool_name=name,
        tool_params=tool_params,
        tool_options={'mfl': 'ABSORPTION(ZO)'},
        args=('exhaustive',),
    )

    assert metadata['tool_name'] == 'modelsearch'
    assert metadata['tool_options']['mfl'] == 'ABSORPTION(ZO)'

    with pytest.raises(Exception, match='modelsearch: \'mfl\' was not set'):
        _create_metadata_tool(
            tool_name=name,
            tool_params=tool_params,
            tool_options={},
            args=('exhaustive',),
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
        assert metadata['database']['path'].endswith('modelsearch_dir1')
        assert 'path' not in metadata.keys()

        path = 'tool_database_path'

        dispatcher = local_dask
        database = LocalDirectoryToolDatabase(name, path)

        metadata = _create_metadata_common(
            common_options={'path': path}, dispatcher=dispatcher, database=database, toolname=name
        )

        assert metadata['database']['path'].endswith('tool_database_path')
        assert metadata['database']['path'].endswith('tool_database_path')

        assert metadata['database']['path'].endswith('/tool_database_path')
