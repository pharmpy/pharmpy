import pytest

from pharmpy.mfl.features import IIV
from pharmpy.mfl.features.symbols import Ref
from pharmpy.mfl.model_features import ModelFeatures


def test_init():
    iiv1 = IIV('CL', 'EXP', False)
    assert iiv1.args == ('CL', 'EXP', False)
    assert iiv1.args == (iiv1.parameter, iiv1.fp, iiv1.optional)

    iiv2 = IIV(Ref('PK'), 'EXP', False)
    assert iiv2.args == (Ref('PK'), 'EXP', False)
    assert iiv2.args == (iiv2.parameter, iiv2.fp, iiv2.optional)


def test_create():
    iiv1 = IIV.create('cl', 'exp', False)
    assert iiv1.args == ('CL', 'EXP', False)
    assert iiv1.args == (iiv1.parameter, iiv1.fp, iiv1.optional)

    iiv2 = IIV.create(Ref('PK'), 'EXP', False)
    assert iiv2.args == (Ref('PK'), 'EXP', False)
    assert iiv2.args == (iiv2.parameter, iiv2.fp, iiv2.optional)

    with pytest.raises(TypeError):
        IIV.create(1, 'exp', True)

    with pytest.raises(TypeError):
        IIV.create('cl', 'exp', 1)

    with pytest.raises(ValueError):
        IIV.create('cl', 'x', True)


def test_replace():
    iiv1 = IIV.create(parameter='CL', fp='EXP', optional=False)
    assert iiv1.args == ('CL', 'EXP', False)
    iiv2 = iiv1.replace(fp='add')
    assert iiv2.args == ('CL', 'ADD', False)

    with pytest.raises(TypeError):
        iiv1.replace(parameter=1)


def test_expand():
    expand_to = {Ref('PK'): ['CL', 'VC', 'MAT']}

    iiv1 = IIV.create('CL', 'EXP', False)
    iiv1_expanded = iiv1.expand(expand_to)
    assert iiv1_expanded == (iiv1,)

    iiv2 = IIV.create(Ref('PK'), 'EXP', False)
    iiv2_expanded = iiv2.expand(expand_to)
    assert len(iiv2_expanded) == 3
    assert iiv2_expanded[0].parameter == 'CL'

    assert iiv2.expand({Ref('PK'): tuple()}) == tuple()

    with pytest.raises(ValueError):
        iiv2.expand(dict())

    with pytest.raises(ValueError):
        iiv2.expand({'x': ('y',)})


def test_repr():
    iiv1 = IIV.create(parameter='CL', fp='EXP', optional=False)
    assert repr(iiv1) == 'IIV(CL,EXP)'
    iiv2 = IIV.create(parameter='CL', fp='EXP', optional=True)
    assert repr(iiv2) == 'IIV?(CL,EXP)'


def test_eq():
    iiv1 = IIV.create(parameter='CL', fp='EXP', optional=False)
    assert iiv1 == iiv1
    iiv2 = IIV.create(parameter='CL', fp='EXP', optional=False)
    assert iiv1 == iiv2
    iiv3 = IIV.create(parameter='CL', fp='ADD', optional=False)
    assert iiv3 != iiv1
    iiv4 = IIV.create(parameter='CL', fp='ADD', optional=True)
    assert iiv4 != iiv3

    assert iiv1 != 1


def test_lt():
    iiv1 = IIV.create(parameter='CL', fp='EXP', optional=False)
    assert not iiv1 < iiv1
    iiv2 = IIV.create(parameter='CL', fp='EXP', optional=True)
    assert iiv1 < iiv2
    iiv3 = IIV.create(parameter='VC', fp='EXP', optional=False)
    assert iiv1 < iiv3
    iiv4 = IIV.create(parameter='CL', fp='ADD', optional=False)
    assert iiv4 < iiv1

    with pytest.raises(TypeError):
        iiv1 < 1


@pytest.mark.parametrize(
    'list_of_kwargs, expected',
    [
        (
            [
                {'parameter': 'CL', 'fp': 'EXP', 'optional': True},
            ],
            'IIV?(CL,EXP)',
        ),
        (
            [
                {'parameter': 'CL', 'fp': 'EXP', 'optional': True},
                {'parameter': 'VC', 'fp': 'EXP', 'optional': True},
            ],
            'IIV?([CL,VC],EXP)',
        ),
        (
            [
                {'parameter': 'CL', 'fp': 'EXP', 'optional': True},
                {'parameter': 'VC', 'fp': 'EXP', 'optional': False},
                {'parameter': 'MAT', 'fp': 'ADD', 'optional': True},
            ],
            'IIV(VC,EXP);IIV?(CL,EXP);IIV?(MAT,ADD)',
        ),
        (
            [
                {'parameter': 'CL', 'fp': 'EXP', 'optional': True},
                {'parameter': 'VC', 'fp': 'EXP', 'optional': True},
                {'parameter': 'MAT', 'fp': 'EXP', 'optional': True},
            ],
            'IIV?([CL,MAT,VC],EXP)',
        ),
        (
            [
                {'parameter': Ref('PK'), 'fp': 'EXP', 'optional': True},
            ],
            'IIV?(@PK,EXP)',
        ),
    ],
)
def test_repr_many(list_of_kwargs, expected):
    features = []
    for kwargs in list_of_kwargs:
        iiv = IIV.create(**kwargs)
        features.append(iiv)
    mfl1 = ModelFeatures.create(features)
    assert IIV.repr_many(mfl1) == expected
    mfl2 = ModelFeatures.create([features[0]])
    assert IIV.repr_many(mfl2) == repr(features[0])
