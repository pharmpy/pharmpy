import pytest

from pharmpy.mfl.features import IIV, IOV
from pharmpy.mfl.features.symbols import Ref
from pharmpy.mfl.model_features import ModelFeatures


@pytest.mark.parametrize(
    'var_type',
    [IIV, IOV],
)
def test_init(var_type):
    var1 = var_type('CL', 'EXP', False)
    assert var1.args == ('CL', 'EXP', False)
    assert var1.args == (var1.parameter, var1.fp, var1.optional)

    var2 = var_type(Ref('PK'), 'EXP', False)
    assert var2.args == (Ref('PK'), 'EXP', False)
    assert var2.args == (var2.parameter, var2.fp, var2.optional)


@pytest.mark.parametrize(
    'var_type',
    [IIV, IOV],
)
def test_create(var_type):
    var1 = var_type.create('cl', 'exp', False)
    assert var1.args == ('CL', 'EXP', False)
    assert var1.args == (var1.parameter, var1.fp, var1.optional)

    var2 = var_type.create(Ref('PK'), 'EXP', False)
    assert var2.args == (Ref('PK'), 'EXP', False)
    assert var2.args == (var2.parameter, var2.fp, var2.optional)

    with pytest.raises(TypeError):
        var_type.create(1, 'exp', True)

    with pytest.raises(TypeError):
        var_type.create('cl', 'exp', 1)

    with pytest.raises(ValueError):
        var_type.create('cl', 'x', True)


@pytest.mark.parametrize(
    'var_type',
    [IIV, IOV],
)
def test_replace(var_type):
    var1 = var_type.create(parameter='CL', fp='EXP', optional=False)
    assert var1.args == ('CL', 'EXP', False)
    var2 = var1.replace(fp='add')
    assert var2.args == ('CL', 'ADD', False)

    with pytest.raises(TypeError):
        var1.replace(parameter=1)


@pytest.mark.parametrize(
    'var_type',
    [IIV, IOV],
)
def test_expand(var_type):
    expand_to = {Ref('PK'): ['CL', 'VC', 'MAT']}

    var1 = var_type.create('CL', 'EXP', False)
    var1_expanded = var1.expand(expand_to)
    assert var1_expanded == (var1,)

    var2 = var_type.create(Ref('PK'), 'EXP', False)
    var2_expanded = var2.expand(expand_to)
    assert len(var2_expanded) == 3
    assert var2_expanded[0].parameter == 'CL'

    assert var2.expand({Ref('PK'): tuple()}) == tuple()

    with pytest.raises(ValueError):
        var2.expand(dict())

    with pytest.raises(ValueError):
        var2.expand({'x': ('y',)})


def test_repr():
    iiv1 = IIV.create(parameter='CL', fp='EXP', optional=False)
    assert repr(iiv1) == 'IIV(CL,EXP)'
    iiv2 = IIV.create(parameter='CL', fp='EXP', optional=True)
    assert repr(iiv2) == 'IIV?(CL,EXP)'

    iov1 = IOV.create(parameter='CL', fp='EXP', optional=False)
    assert repr(iov1) == 'IOV(CL,EXP)'
    iov2 = IOV.create(parameter='CL', fp='EXP', optional=True)
    assert repr(iov2) == 'IOV?(CL,EXP)'


@pytest.mark.parametrize(
    'var_type',
    [IIV, IOV],
)
def test_eq(var_type):
    var1 = var_type.create(parameter='CL', fp='EXP', optional=False)
    assert var1 == var1
    var2 = var_type.create(parameter='CL', fp='EXP', optional=False)
    assert var1 == var2
    var3 = var_type.create(parameter='CL', fp='ADD', optional=False)
    assert var3 != var1
    var4 = var_type.create(parameter='CL', fp='ADD', optional=True)
    assert var4 != var3

    assert var1 != 1


def test_eq_iiv_iov():
    iiv = IIV.create(parameter='CL', fp='EXP', optional=False)
    iov = IOV.create(parameter='CL', fp='EXP', optional=False)
    assert iiv != iov


@pytest.mark.parametrize(
    'var_type',
    [IIV, IOV],
)
def test_lt(var_type):
    var1 = var_type.create(parameter='CL', fp='EXP', optional=False)
    assert not var1 < var1
    var2 = var_type.create(parameter='CL', fp='EXP', optional=True)
    assert var1 < var2
    var3 = var_type.create(parameter='VC', fp='EXP', optional=False)
    assert var1 < var3
    var4 = var_type.create(parameter='CL', fp='ADD', optional=False)
    assert var4 < var1

    with pytest.raises(TypeError):
        var1 < 1


def test_lt_iiv_iov():
    iiv = IIV.create(parameter='CL', fp='EXP', optional=False)
    iov = IOV.create(parameter='CL', fp='EXP', optional=False)
    with pytest.raises(TypeError):
        iiv < iov


@pytest.mark.parametrize(
    'var_type, list_of_kwargs, expected',
    [
        (
            IIV,
            [
                {'parameter': 'CL', 'fp': 'EXP', 'optional': True},
            ],
            'IIV?(CL,EXP)',
        ),
        (
            IIV,
            [
                {'parameter': 'CL', 'fp': 'EXP', 'optional': True},
                {'parameter': 'VC', 'fp': 'EXP', 'optional': True},
            ],
            'IIV?([CL,VC],EXP)',
        ),
        (
            IIV,
            [
                {'parameter': 'CL', 'fp': 'EXP', 'optional': True},
                {'parameter': 'VC', 'fp': 'EXP', 'optional': False},
                {'parameter': 'MAT', 'fp': 'ADD', 'optional': True},
            ],
            'IIV(VC,EXP);IIV?(CL,EXP);IIV?(MAT,ADD)',
        ),
        (
            IIV,
            [
                {'parameter': 'CL', 'fp': 'EXP', 'optional': True},
                {'parameter': 'VC', 'fp': 'EXP', 'optional': True},
                {'parameter': 'MAT', 'fp': 'EXP', 'optional': True},
            ],
            'IIV?([CL,MAT,VC],EXP)',
        ),
        (
            IIV,
            [
                {'parameter': Ref('PK'), 'fp': 'EXP', 'optional': True},
            ],
            'IIV?(@PK,EXP)',
        ),
        (
            IOV,
            [
                {'parameter': 'CL', 'fp': 'EXP', 'optional': True},
            ],
            'IOV?(CL,EXP)',
        ),
        (
            IOV,
            [
                {'parameter': 'CL', 'fp': 'EXP', 'optional': True},
                {'parameter': 'VC', 'fp': 'EXP', 'optional': True},
            ],
            'IOV?([CL,VC],EXP)',
        ),
        (
            IOV,
            [
                {'parameter': Ref('PK'), 'fp': 'EXP', 'optional': True},
            ],
            'IOV?(@PK,EXP)',
        ),
    ],
)
def test_repr_many(var_type, list_of_kwargs, expected):
    features = []
    for kwargs in list_of_kwargs:
        var = var_type.create(**kwargs)
        features.append(var)
    mfl1 = ModelFeatures.create(features)
    assert var_type.repr_many(mfl1) == expected
    mfl2 = ModelFeatures.create([features[0]])
    assert var_type.repr_many(mfl2) == repr(features[0])
