import pytest

exports = (
    ('create_results',),
    ('fit',),
    ('read_results',),
    ('retrieve_models',),
    ('run_allometry',),
    ('run_amd',),
    ('run_covsearch',),
    ('run_iivsearch',),
    ('run_iovsearch',),
    ('run_modelsearch',),
    ('run_ruvsearch',),
    ('run_tool',),
)


@pytest.mark.parametrize(('tool',), exports)
def test_import_tools_all(tool):
    import pharmpy.tools as tools

    assert tool in getattr(tools, '__all__')


@pytest.mark.parametrize(('tool',), exports)
def test_import_tools_dir(tool):
    import pharmpy.tools as tools

    assert tool in dir(tools)


@pytest.mark.parametrize(('tool',), exports)
def test_import_tools_attr(tool):
    import pharmpy.tools as tools

    assert callable(getattr(tools, tool))


@pytest.mark.parametrize(('tool',), exports)
def test_import_tools_import(tool):
    mod = __import__('pharmpy.tools', globals(), locals(), [tool], 0)
    assert callable(getattr(mod, tool))
