import pytest

from pharmpy import Model
from pharmpy.modeling.remove_iiv import _get_etas


def test_get_etas(testdata):
    model = Model(testdata / 'nonmem' / 'pheno_block.mod')

    etas = _get_etas(model, ['ETA(1)'])
    assert len(etas) == 1

    etas = _get_etas(model, ['ETA(1)', 'CL'])
    assert len(etas) == 1

    etas = _get_etas(model, ['ETA(1)', 'V'])
    assert len(etas) == 2

    etas = _get_etas(model, ['ETA(1)', 'S1'])
    assert len(etas) == 4

    with pytest.raises(KeyError):
        _get_etas(model, ['ETA(23)'])
