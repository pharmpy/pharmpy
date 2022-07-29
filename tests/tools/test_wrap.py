import inspect

import pytest


def _is_iivsearch(obj):
    params = inspect.signature(obj).parameters
    assert 'algorithm' in params.keys()
    assert 'rank_type' in params.keys()
    assert params['rank_type'].default == 'bic'
    return True


def _is_modelsearch(obj):
    params = inspect.signature(obj).parameters
    assert 'search_space' in params.keys()
    assert 'rank_type' in params.keys()
    assert params['rank_type'].default == 'bic'
    return True


@pytest.mark.parametrize(
    ('attr', 'identity'),
    (
        ('run_iivsearch', _is_iivsearch),
        ('run_modelsearch', _is_modelsearch),
    ),
)
def test_import_tools_getattr(attr, identity):
    import pharmpy.tools as tools

    obj = getattr(tools, attr, None)
    assert identity(obj)


def test_import_tools_import_run_iivsearch():
    from pharmpy.tools import run_iivsearch

    assert _is_iivsearch(run_iivsearch)


def test_import_tools_import_run_modelsearch():
    from pharmpy.tools import run_modelsearch

    assert _is_modelsearch(run_modelsearch)


@pytest.mark.parametrize(
    ('attr',),
    (
        ('run_allometry',),
        ('run_iivsearch',),
        ('run_modelsearch',),
        ('run_ruvsearch',),
    ),
)
def test_import_tools_attr(attr):
    import pharmpy.tools as tools

    _all = getattr(tools, '__all__', None)

    assert attr in dir() if _all is None else attr in _all
