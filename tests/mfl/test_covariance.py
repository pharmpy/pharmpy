import pytest

from pharmpy.mfl.features import Covariance, Ref
from pharmpy.mfl.model_features import ModelFeatures


def test_init():
    c = Covariance(type='IIV', parameters=('CL', 'VC'), optional=True)
    assert c.args == ('IIV', ('CL', 'VC'), True)
    assert c.args == (c.type, c.parameters, c.optional)


def test_create():
    c1 = Covariance.create(type='iiv', parameters=['vc', 'cl'])
    assert c1.args == ('IIV', ('CL', 'VC'), False)
    assert c1.args == (c1.type, c1.parameters, c1.optional)

    c2 = Covariance.create(type='iov', parameters=['vc', 'cl'], optional=True)
    assert c2.args == ('IOV', ('CL', 'VC'), True)
    assert c2.args == (c2.type, c2.parameters, c2.optional)

    with pytest.raises(TypeError):
        Covariance.create(1, ['cl', 'vc'])

    with pytest.raises(ValueError):
        Covariance.create('x', ['cl', 'vc'])

    with pytest.raises(TypeError):
        Covariance.create('iiv', 1)

    with pytest.raises(ValueError):
        Covariance.create('iiv', ['cl'])

    with pytest.raises(TypeError):
        Covariance.create('iiv', [1, 2])

    with pytest.raises(TypeError):
        Covariance.create('iiv', ['cl', 'vc'], 1)


def test_replace():
    c1 = Covariance.create(type='iiv', parameters=['cl', 'vc'])
    assert c1.args == ('IIV', ('CL', 'VC'), False)
    c2 = c1.replace(type='iov')
    assert c2.args == ('IOV', ('CL', 'VC'), False)
    c3 = c2.replace(parameters=['vc', 'mat'])
    assert c3.args == ('IOV', ('MAT', 'VC'), False)

    with pytest.raises(TypeError):
        c1.replace(type=1)


def test_expand():
    expand_to = {Ref('PK'): [['CL', 'VC'], ['CL', 'MAT'], ['VC', 'MAT']]}

    c1 = Covariance.create(type='IIV', parameters=('CL', 'VC'))
    c1_expanded = c1.expand(expand_to)
    assert c1_expanded == (c1,)

    c2 = Covariance.create(type='IIV', parameters=Ref('PK'))
    c2_expanded = c2.expand(expand_to)
    assert len(c2_expanded) == 3
    assert c2_expanded[0].parameters == ('CL', 'MAT')

    assert c2.expand({Ref('PK'): tuple()}) == tuple()

    expand_to = {Ref('PK'): ('CL', 'VC', 'MAT')}

    c3 = Covariance.create(type='IIV', parameters=Ref('PK'))
    c3_expanded = c3.expand(expand_to)
    assert len(c3_expanded) == 3
    assert c3_expanded[0].parameters == ('CL', 'MAT')

    with pytest.raises(ValueError):
        c2.expand(dict())

    with pytest.raises(ValueError):
        c2.expand({'x': ('y', 'z')})


def test_repr():
    c1 = Covariance.create(type='iiv', parameters=['vc', 'cl'])
    assert repr(c1) == 'COVARIANCE(IIV,[CL,VC])'
    c2 = Covariance.create(type='iov', parameters=['vc', 'cl'], optional=True)
    assert repr(c2) == 'COVARIANCE?(IOV,[CL,VC])'
    c3 = Covariance.create(type='iov', parameters=Ref('PK'), optional=True)
    assert repr(c3) == 'COVARIANCE?(IOV,@PK)'


@pytest.mark.parametrize(
    'kwargs1, kwargs2',
    [
        (
            {'type': 'IIV', 'parameters': ('CL', 'VC'), 'optional': False},
            {'type': 'IOV', 'parameters': ('CL', 'VC'), 'optional': False},
        ),
        (
            {'type': 'IIV', 'parameters': ('CL', 'VC'), 'optional': False},
            {'type': 'IIV', 'parameters': ('CL', 'MAT'), 'optional': False},
        ),
        (
            {'type': 'IIV', 'parameters': ('CL', 'VC'), 'optional': False},
            {'type': 'IIV', 'parameters': ('CL', 'VC'), 'optional': True},
        ),
    ],
)
def test_eq(kwargs1, kwargs2):
    c1 = Covariance.create(**kwargs1)
    assert c1 == c1
    assert c1 == Covariance.create(**kwargs1)
    c2 = Covariance.create(**kwargs2)
    assert c1 != c2

    assert c1 != 1


@pytest.mark.parametrize(
    'kwargs1, kwargs2, expected',
    [
        (
            {'type': 'IIV', 'parameters': ('CL', 'VC'), 'optional': False},
            {'type': 'IOV', 'parameters': ('CL', 'VC'), 'optional': False},
            True,
        ),
        (
            {'type': 'IIV', 'parameters': ('CL', 'VC'), 'optional': False},
            {'type': 'IIV', 'parameters': ('CL', 'MAT'), 'optional': False},
            False,
        ),
        (
            {'type': 'IIV', 'parameters': ('CL', 'VC'), 'optional': False},
            {'type': 'IIV', 'parameters': Ref('PK'), 'optional': False},
            False,
        ),
        (
            {'type': 'IIV', 'parameters': Ref('PD'), 'optional': False},
            {'type': 'IIV', 'parameters': Ref('PK'), 'optional': False},
            True,
        ),
        (
            {'type': 'IIV', 'parameters': ('CL', 'VC'), 'optional': False},
            {'type': 'IOV', 'parameters': ('CL', 'VC'), 'optional': True},
            True,
        ),
    ],
)
def test_lt(kwargs1, kwargs2, expected):
    c1 = Covariance.create(**kwargs1)
    assert not c1 < c1
    c2 = Covariance.create(**kwargs2)
    assert (c1 < c2) == expected

    with pytest.raises(TypeError):
        c1 < 1


@pytest.mark.parametrize(
    'parameters, expected',
    [
        (
            (('CL', 'VC'),),
            (('CL', 'VC'),),
        ),
        (
            (('CL', 'VC'), ('CL', 'MAT'), ('MAT', 'VC')),
            (('CL', 'MAT', 'VC'),),
        ),
        (
            (('CL', 'VC'), ('CL', 'MAT'), ('MAT', 'VC'), ('QP1', 'VP1')),
            (('CL', 'MAT', 'VC'), ('QP1', 'VP1')),
        ),
        (
            (
                ('CL', 'VC'),
                ('CL', 'MAT'),
                ('MAT', 'VC'),
                ('QP1', 'VP1'),
                ('QP1', 'VP2'),
                ('VP1', 'VP2'),
            ),
            (('CL', 'MAT', 'VC'), ('QP1', 'VP1', 'VP2')),
        ),
        (
            (('CL', 'VC'), ('CL', 'MAT'), ('MAT', 'VC'), Ref('PD')),
            (
                (Ref('PD')),
                ('CL', 'MAT', 'VC'),
            ),
        ),
    ],
)
def test_get_covariance_blocks(parameters, expected):
    features = [Covariance.create('IIV', pair) for pair in parameters]
    mf = ModelFeatures.create(features)
    assert Covariance.get_covariance_blocks(mf) == expected

    with pytest.raises(ValueError):
        mf += Covariance.create('IOV', parameters[0])
        Covariance.get_covariance_blocks(mf)


@pytest.mark.parametrize(
    'list_of_kwargs, expected',
    [
        (
            [
                {'type': 'IIV', 'parameters': ('CL', 'VC'), 'optional': False},
                {'type': 'IOV', 'parameters': ('CL', 'VC'), 'optional': True},
            ],
            'COVARIANCE(IIV,[CL,VC]);COVARIANCE?(IOV,[CL,VC])',
        ),
        (
            [
                {'type': 'IIV', 'parameters': ('CL', 'VC'), 'optional': False},
                {'type': 'IIV', 'parameters': ('CL', 'MAT'), 'optional': False},
                {'type': 'IIV', 'parameters': ('MAT', 'VC'), 'optional': False},
            ],
            'COVARIANCE(IIV,[CL,MAT,VC])',
        ),
        (
            [
                {'type': 'IIV', 'parameters': ('CL', 'VC'), 'optional': False},
                {'type': 'IIV', 'parameters': ('CL', 'MAT'), 'optional': False},
                {'type': 'IIV', 'parameters': ('MAT', 'VC'), 'optional': False},
                {'type': 'IIV', 'parameters': ('MAT', 'MDT'), 'optional': False},
            ],
            'COVARIANCE(IIV,[CL,MAT]);COVARIANCE(IIV,[CL,VC]);COVARIANCE(IIV,[MAT,MDT]);COVARIANCE(IIV,[MAT,VC])',
        ),
    ],
)
def test_repr_many(list_of_kwargs, expected):
    features = []
    for kwargs in list_of_kwargs:
        c = Covariance.create(**kwargs)
        features.append(c)
    mfl1 = ModelFeatures.create(features)
    assert Covariance.repr_many(mfl1) == expected
    mfl2 = ModelFeatures.create([features[0]])
    assert Covariance.repr_many(mfl2) == repr(features[0])
