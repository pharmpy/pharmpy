import pytest

from pharmpy import Model
from pharmpy.modeling.help_functions import _as_integer, _get_etas


def test_get_etas(pheno_path, testdata):
    model = Model.create_model(pheno_path)

    etas = _get_etas(model, ['ETA(1)'])
    assert len(etas) == 1

    etas = _get_etas(model, ['ETA(1)', 'CL'], include_symbols=True)
    assert len(etas) == 1

    etas = _get_etas(model, ['ETA(1)', 'V'], include_symbols=True)
    assert len(etas) == 2

    with pytest.raises(KeyError):
        _get_etas(model, ['ETA(23)'])

    model = Model.create_model(testdata / 'nonmem' / 'pheno_block.mod')
    rvs = _get_etas(model, None)
    assert rvs[0].name == 'ETA(1)'

    model.parameters.fix = {'OMEGA(1,1)': True}
    rvs = _get_etas(model, None)
    assert rvs[0].name == 'ETA(2)'

    model = Model.create_model(testdata / 'nonmem' / 'pheno_block.mod')
    model.random_variables['ETA(1)'].level = 'IOV'
    rvs = _get_etas(model, None)
    assert rvs[0].name == 'ETA(2)'


def test_as_integer():
    n = _as_integer(1)
    assert isinstance(n, int) and n == 1
    n = _as_integer(1.0)
    assert isinstance(n, int) and n == 1
    with pytest.raises(TypeError):
        _as_integer(1.5)
