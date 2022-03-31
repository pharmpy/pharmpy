import importlib
import inspect

from pharmpy.modeling.run import _create_metadata


def test_create_metadata():
    print('\n')
    name = 'modelsearch'
    tool = importlib.import_module(f'pharmpy.tools.{name}')
    tool_params = inspect.signature(tool.create_workflow).parameters

    metadata = _create_metadata(
        name,
        tool_params,
        ('exhaustive', 'ABSORPTION(ZO)'),
        {'path': '/path/to/db/'},
        {'add_iivs': True},
    )

    assert metadata['tool_name'] == 'modelsearch'
    assert metadata['common_options']['path'] == '/path/to/db/'
    assert metadata['tool_options']['rankfunc'] == 'ofv'
