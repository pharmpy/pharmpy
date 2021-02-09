import pytest

from pharmpy import Model
from pharmpy.modeling.remove_iiv import _get_etas


def test_get_etas(pheno_path):
    model = Model(pheno_path)

    etas = _get_etas(model, ['ETA(1)'])
    assert len(etas) == 1

    etas = _get_etas(model, ['ETA(1)', 'CL'])
    assert len(etas) == 1

    etas = _get_etas(model, ['ETA(1)', 'V'])
    assert len(etas) == 2

    with pytest.raises(KeyError):
        _get_etas(model, ['ETA(23)'])
