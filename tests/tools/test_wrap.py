import inspect

import pytest


def _is_iivsearch(obj):
    params = inspect.signature(obj).parameters
    assert 'algorithm' in params.keys()
    assert 'rank_type' in params.keys()
    assert params['rank_type'].default == 'bic'
    assert 'kwargs' in params.keys()
    assert 'kwargs' in obj.__doc__
    return True


def _is_modelsearch(obj):
    params = inspect.signature(obj).parameters
    assert 'search_space' in params.keys()
    assert 'rank_type' in params.keys()
    assert params['rank_type'].default == 'bic'
    assert 'kwargs' in params.keys()
    assert 'kwargs' in obj.__doc__
    return True


def _dynamic_tool_import_1(tool: str):
    mod = __import__('pharmpy.tools', globals(), locals(), [tool], 0)
    return getattr(mod, tool)


def _dynamic_tool_import_2(tool: str):
    import pharmpy.tools as mod

    return getattr(mod, tool)


@pytest.mark.parametrize(
    ('tool_import',),
    (
        (_dynamic_tool_import_1,),
        (_dynamic_tool_import_2,),
    ),
)
@pytest.mark.parametrize(
    ('tool', 'identity'),
    (
        ('run_iivsearch', _is_iivsearch),
        ('run_modelsearch', _is_modelsearch),
    ),
)
def test_import_tools_getattr(tool_import, tool, identity):
    obj = tool_import(tool)
    assert identity(obj)


def test_import_tools_import_run_iivsearch():
    from pharmpy.tools import run_iivsearch

    assert _is_iivsearch(run_iivsearch)


def test_import_tools_import_run_modelsearch():
    from pharmpy.tools import run_modelsearch

    assert _is_modelsearch(run_modelsearch)
