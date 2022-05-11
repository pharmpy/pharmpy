import inspect


def test_import_tool_wrappers():
    import pharmpy.modeling.tool_wrappers as tool_wrappers

    assert 'run_iivsearch' in dir(tool_wrappers)


def test_import_modeling():
    import pharmpy.modeling as modeling

    assert 'run_iivsearch' in dir(modeling)

    from pharmpy.modeling import run_iivsearch

    params = inspect.signature(run_iivsearch).parameters
    assert 'algorithm' in params.keys()
    assert 'rankfunc' in params.keys()
    assert params['rankfunc'].default == 'bic'

    from pharmpy.modeling import run_modelsearch

    params = inspect.signature(run_modelsearch).parameters
    assert 'search_space' in params.keys()
    assert 'rankfunc' in params.keys()
    assert params['rankfunc'].default == 'bic'
