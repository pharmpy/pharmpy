import pickle

import numpy as np
import pytest
import sympy
from sympy import Symbol as symbol

from pharmpy.model import (
    JointNormalDistribution,
    NormalDistribution,
    RandomVariables,
    VariabilityHierarchy,
    VariabilityLevel,
)


def test_normal_rv():
    dist = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    assert dist.names == ('ETA(1)',)
    assert dist.level == 'IIV'
    dist = dist.replace(name='NEW')
    assert dist.names == ('NEW',)
    with pytest.raises(ValueError):
        NormalDistribution.create('X', 'iiv', 0, -1)


def test_joint_normal_rv():
    dist = JointNormalDistribution.create(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    assert dist.names == ('ETA(1)', 'ETA(2)')
    assert dist.level == 'IIV'
    dist = dist.replace(names=['NEW', 'ETA(2)'])
    assert dist.names == ('NEW', 'ETA(2)')
    with pytest.raises(ValueError):
        JointNormalDistribution.create(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[-1, 0.1], [0.1, 2]])


def test_eq_rv():
    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    assert dist1 == dist2
    dist3 = NormalDistribution.create('ETA(2)', 'iiv', 0, 1)
    assert dist1 != dist3
    dist4 = NormalDistribution.create('ETA(2)', 'iiv', 0, 0.1)
    assert dist3 != dist4


def test_empty_rvs():
    rvs = RandomVariables.create([])
    assert not rvs
    assert rvs.covariance_matrix == sympy.Matrix()


def test_repr_rv():
    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    assert repr(dist1) == 'ETA(1) ~ N(0, 1)'
    dist2 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]]
    )
    assert (
        repr(dist2)
        == """⎡ETA(1)⎤    ⎧⎡0⎤  ⎡ 1   0.1⎤⎫
⎢      ⎥ ~ N⎪⎢ ⎥, ⎢        ⎥⎪
⎣ETA(2)⎦    ⎩⎣0⎦  ⎣0.1   2 ⎦⎭"""
    )


def test_repr_latex_rv():
    dist1 = JointNormalDistribution.create(['x', 'y'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    assert (
        dist1._repr_latex_()
        == '$\\displaystyle \\left[\\begin{matrix}x\\\\y\\end{matrix}\\right]\\sim \\mathcal{N} \\left(\\displaystyle \\left[\\begin{matrix}0\\\\0\\end{matrix}\\right],\\displaystyle \\left[\\begin{matrix}1 & 0.1\\\\0.1 & 2\\end{matrix}\\right]\\right)$'  # noqa E501
    )

    dist2 = NormalDistribution.create('x', 'iiv', 0, 1)
    assert (
        dist2._repr_latex_()
        == '$\\displaystyle x\\sim  \\mathcal{N} \\left(\\displaystyle 0,\\displaystyle 1\\right)$'
    )


def test_parameters_rv():
    dist1 = NormalDistribution.create('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    assert dist1.parameter_names == ('OMEGA(2,2)',)


def test_init():
    rvs = RandomVariables.create([])
    assert len(rvs) == 0
    rvs = RandomVariables.create()
    assert len(rvs) == 0


def test_illegal_inits():
    with pytest.raises(TypeError):
        RandomVariables.create([8, 1])
    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    with pytest.raises(ValueError):
        RandomVariables.create([dist1, dist2])
    with pytest.raises(TypeError):
        RandomVariables.create([dist1], eta_levels=23)
    with pytest.raises(TypeError):
        RandomVariables.create([dist1], epsilon_levels=23)


def test_len():
    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA(2)', 'iiv', 0, 1)
    rvs = RandomVariables.create([dist1, dist2])
    assert len(rvs) == 2


def test_eq():
    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA(2)', 'iiv', 0, 1)
    dist3 = NormalDistribution.create('ETA(3)', 'iiv', 0, 1)
    rvs = RandomVariables.create([dist1, dist2])
    rvs2 = RandomVariables.create([dist1])
    assert rvs != rvs2
    rvs3 = RandomVariables.create([dist1, dist3])
    assert rvs != rvs3
    assert hash(rvs) != hash(rvs2)


def test_add():
    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA(2)', 'iiv', 0, 0.1)
    dist3 = NormalDistribution.create('ETA(3)', 'center', 0, 0.1)
    rvs1 = RandomVariables.create([dist1])
    rvs2 = RandomVariables.create([dist2])
    rvs3 = rvs1 + rvs2
    assert len(rvs3) == 2
    assert len(rvs1 + dist2) == 2
    assert len(dist1 + rvs2) == 2
    with pytest.raises(TypeError):
        rvs1 + None
    with pytest.raises(TypeError):
        None + rvs1
    assert len([dist2] + rvs1) == 2

    with pytest.raises(ValueError):
        dist3 + rvs1
    with pytest.raises(ValueError):
        rvs1 + dist3

    lev1 = VariabilityLevel('center', reference=True, group='CENTER')
    levs = VariabilityHierarchy([lev1])
    rvs4 = RandomVariables.create([dist3], eta_levels=levs)
    with pytest.raises(ValueError):
        rvs4 + rvs2


def test_getitem():
    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA(2)', 'iiv', 0, 0.1)
    rvs = RandomVariables.create([dist1, dist2])
    assert rvs[0] == dist1
    assert rvs[1] == dist2
    assert rvs['ETA(1)'] == dist1
    assert rvs['ETA(2)'] == dist2
    assert rvs[symbol('ETA(1)')] == dist1
    assert rvs[symbol('ETA(2)')] == dist2
    with pytest.raises(IndexError):
        rvs[23]
    with pytest.raises(KeyError):
        rvs['NOKEYOFTHIS']

    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA(2)', 'iiv', 0, 0.1)
    dist3 = NormalDistribution.create('ETA(3)', 'iiv', 0, 0.1)

    with pytest.raises(KeyError):
        dist1[(0, 1)]

    with pytest.raises(IndexError):
        dist1[1:23]

    with pytest.raises(KeyError):
        dist1['ETA(2)']

    with pytest.raises(IndexError):
        dist1[1]

    with pytest.raises(KeyError):
        dist1[None]

    assert len(dist1[0:1]) == 1
    assert len(dist1['ETA(1)']) == 1
    assert len(dist1[0]) == 1

    rvs = RandomVariables.create([dist1, dist2, dist3])
    selection = rvs[['ETA(1)', 'ETA(2)']]
    assert len(selection) == 2
    selection = rvs[1:]
    assert len(selection) == 2

    dist1 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')],
        ],
    )
    rvs = RandomVariables.create([dist1])
    selection = rvs[['ETA(1)']]
    assert isinstance(selection['ETA(1)'], NormalDistribution)

    with pytest.raises(KeyError):
        rvs[None]

    assert isinstance(dist1[['ETA(1)']], NormalDistribution)

    dist2 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)', 'ETA(3)'],
        'iiv',
        [0, 0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)'), symbol('OMEGA(3,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)'), symbol('OMEGA(2,3)')],
            [symbol('OMEGA(3,1)'), symbol('OMEGA(2,3)'), symbol('OMEGA(3,3)')],
        ],
    )

    assert isinstance(dist2[['ETA(1)', 'ETA(3)']], JointNormalDistribution)
    assert len(dist2[0:2]) == 2
    assert dist2['ETA(1)'].names == ('ETA(1)',)
    assert dist2[0].names == ('ETA(1)',)

    with pytest.raises(KeyError):
        dist2[None]

    with pytest.raises(KeyError):
        dist2[['ETA(1)', 'ETA(4)']]

    with pytest.raises(KeyError):
        dist2[['x', 'y', 'z', 'w']]

    with pytest.raises(KeyError):
        dist2[tuple()]

    with pytest.raises(IndexError):
        dist2[4]

    with pytest.raises(KeyError):
        dist2['ETA(23)']


def test_contains():
    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA(2)', 'iiv', 0, 0.1)
    dist3 = NormalDistribution.create('ETA(3)', 'iiv', 0, 0.1)
    rvs = RandomVariables.create([dist1, dist2, dist3])
    assert 'ETA(2)' in rvs
    assert 'ETA(4)' not in rvs


def test_names():
    dist1 = NormalDistribution.create('ETA1', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA2', 'iiv', 0, 0.1)
    rvs = RandomVariables.create([dist1, dist2])
    assert rvs.names == ['ETA1', 'ETA2']


def test_epsilons():
    dist1 = NormalDistribution.create('ETA', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA2', 'iiv', 0, 0.1)
    dist3 = NormalDistribution.create('EPS', 'ruv', 0, 0.1)
    rvs = RandomVariables.create([dist1, dist2, dist3])
    assert rvs.epsilons == RandomVariables.create([dist3])
    assert rvs.epsilons.names == ['EPS']


def test_etas():
    dist1 = NormalDistribution.create('ETA', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA2', 'iiv', 0, 0.1)
    dist3 = NormalDistribution.create('EPS', 'ruv', 0, 0.1)
    rvs = RandomVariables.create([dist1, dist2, dist3])
    assert rvs.etas == RandomVariables.create([dist1, dist2])
    assert rvs.etas.names == ['ETA', 'ETA2']


def test_iiv_iov():
    dist1 = NormalDistribution.create('ETA', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA2', 'iov', 0, 0.1)
    dist3 = NormalDistribution.create('EPS', 'ruv', 0, 0.1)
    rvs = RandomVariables.create([dist1, dist2, dist3])
    assert rvs.iiv == RandomVariables.create([dist1])
    assert rvs.iiv.names == ['ETA']
    assert rvs.iov == RandomVariables.create([dist2])
    assert rvs.iov.names == ['ETA2']


def test_subs():
    dist1 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')],
        ],
    )
    dist2 = NormalDistribution.create('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rvs = RandomVariables.create([dist1, dist2])
    rvs = rvs.subs(
        {
            symbol('ETA(2)'): symbol('w'),
            symbol('OMEGA(1,1)'): symbol('x'),
            symbol('OMEGA(3,3)'): symbol('y'),
        }
    )
    assert rvs['ETA(1)'].variance == sympy.ImmutableMatrix(
        [[symbol('x'), symbol('OMEGA(2,1)')], [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')]]
    )
    assert rvs['ETA(3)'].variance == symbol('y')
    assert rvs.names == ['ETA(1)', 'w', 'ETA(3)']
    dist3 = dist2.subs({'ETA(3)': 'X'})
    assert dist3.names == ('X',)


def test_free_symbols():
    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, 1)
    dist2 = NormalDistribution.create('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    assert dist2.free_symbols == {symbol('ETA(2)'), symbol('OMEGA(2,2)')}
    rvs = RandomVariables.create([dist1, dist2])
    assert rvs.free_symbols == {symbol('ETA(1)'), symbol('ETA(2)'), symbol('OMEGA(2,2)')}


def test_parameter_names():
    dist1 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')],
        ],
    )
    assert dist1.parameter_names == ('OMEGA(1,1)', 'OMEGA(2,1)', 'OMEGA(2,2)')
    dist2 = NormalDistribution.create('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    assert dist2.parameter_names == ('OMEGA(3,3)',)
    rvs = RandomVariables.create([dist1, dist2])
    assert rvs.parameter_names == ('OMEGA(1,1)', 'OMEGA(2,1)', 'OMEGA(2,2)', 'OMEGA(3,3)')


def test_subs_rv():
    dist = NormalDistribution.create('ETA(1)', 'iiv', 0, symbol('OMEGA(3,3)'))
    dist = dist.subs({symbol('OMEGA(3,3)'): symbol('VAR')})
    assert dist.variance == symbol('VAR')


def test_repr():
    dist1 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]]
    )
    dist2 = NormalDistribution.create('ETA(3)', 'iiv', 2, 1)
    rvs = RandomVariables.create([dist1, dist2])
    res = """⎡ETA(1)⎤    ⎧⎡0⎤  ⎡ 1   0.1⎤⎫
⎢      ⎥ ~ N⎪⎢ ⎥, ⎢        ⎥⎪
⎣ETA(2)⎦    ⎩⎣0⎦  ⎣0.1   2 ⎦⎭
ETA(3) ~ N(2, 1)"""
    assert repr(rvs) == res
    dist3 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'], 'iiv', [sympy.sqrt(sympy.Rational(2, 5)), 0], [[1, 0.1], [0.1, 2]]
    )
    assert (
        repr(dist3)
        == '''            ⎧⎡√10⎤            ⎫
⎡ETA(1)⎤    ⎪⎢───⎥  ⎡ 1   0.1⎤⎪
⎢      ⎥ ~ N⎪⎢ 5 ⎥, ⎢        ⎥⎪
⎣ETA(2)⎦    ⎪⎢   ⎥  ⎣0.1   2 ⎦⎪
            ⎩⎣ 0 ⎦            ⎭'''
    )


def test_repr_latex():
    dist1 = NormalDistribution.create('z', 'iiv', 0, 1)
    dist2 = JointNormalDistribution.create(['x', 'y'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    rvs = RandomVariables.create([dist1, dist2])
    assert (
        rvs._repr_latex_()
        == '\\begin{align*}\n\\displaystyle z & \\sim  \\mathcal{N} \\left(\\displaystyle 0,\\displaystyle 1\\right) \\\\ \\displaystyle \\left[\\begin{matrix}x\\\\y\\end{matrix}\\right] & \\sim \\mathcal{N} \\left(\\displaystyle \\left[\\begin{matrix}0\\\\0\\end{matrix}\\right],\\displaystyle \\left[\\begin{matrix}1 & 0.1\\\\0.1 & 2\\end{matrix}\\right]\\right)\\end{align*}'  # noqa E501
    )


def test_pickle():
    dist1 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]]
    )
    dist2 = NormalDistribution.create('ETA(3)', 'iiv', 2, 1)
    rvs = RandomVariables.create([dist1, dist2])
    pickled = pickle.dumps(rvs)
    obj = pickle.loads(pickled)
    assert obj == rvs


def test_hash():
    dist1 = NormalDistribution.create('ETA(3)', 'iiv', 2, 1)
    dist2 = NormalDistribution.create('ETA(2)', 'iiv', 2, 0)
    assert hash(dist1) != hash(dist2)
    dist3 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]]
    )
    dist4 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(3)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]]
    )
    assert hash(dist3) != hash(dist4)


def test_dict():
    dist1 = NormalDistribution.create('ETA_3', 'iiv', 2, 1)
    d = dist1.to_dict()
    assert d == {
        'class': 'NormalDistribution',
        'name': 'ETA_3',
        'level': 'IIV',
        'mean': 'Integer(2)',
        'variance': 'Integer(1)',
    }
    dist2 = NormalDistribution.from_dict(d)
    assert dist1 == dist2

    dist1 = NormalDistribution.create('ETA_1', 'iiv', 0, 1)
    dist2 = JointNormalDistribution.create(
        ['ETA_2', 'ETA_3'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA11'), symbol('OMEGA21')],
            [symbol('OMEGA21'), symbol('OMEGA22')],
        ],
    )
    rvs = RandomVariables.create([dist1, dist2])
    d = rvs.to_dict()
    assert d == {
        'dists': (
            {
                'class': 'NormalDistribution',
                'name': 'ETA_1',
                'level': 'IIV',
                'mean': 'Integer(0)',
                'variance': 'Integer(1)',
            },
            {
                'class': 'JointNormalDistribution',
                'names': ('ETA_2', 'ETA_3'),
                'level': 'IIV',
                'mean': 'ImmutableDenseMatrix([[Integer(0)], [Integer(0)]])',
                'variance': "ImmutableDenseMatrix([[Symbol('OMEGA11'), Symbol('OMEGA21')], [Symbol('OMEGA21'), Symbol('OMEGA22')]])",  # noqa: E501
            },
        ),
        'eta_levels': {
            'levels': (
                {'name': 'IIV', 'reference': True, 'group': 'ID'},
                {'name': 'IOV', 'reference': False, 'group': 'OCC'},
            )
        },
        'epsilon_levels': {'levels': ({'name': 'RUV', 'reference': True, 'group': None},)},
    }
    rvs2 = RandomVariables.from_dict(d)
    assert rvs == rvs2


def test_nearest_valid_parameters():
    values = {'x': 1, 'y': 0.1, 'z': 2}
    dist1 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [[symbol('x'), symbol('y')], [symbol('y'), symbol('z')]],
    )
    rvs = RandomVariables.create([dist1])
    new = rvs.nearest_valid_parameters(values)
    assert values == new
    values = {'x': 1, 'y': 1.1, 'z': 1}
    new = rvs.nearest_valid_parameters(values)
    assert new == {'x': 1.0500000000000005, 'y': 1.0500000000000003, 'z': 1.050000000000001}

    dist2 = NormalDistribution.create('ETA(3)', 'iiv', 2, 1)
    rvs = RandomVariables.create([dist2])
    values = {symbol('ETA(3)'): 5}
    new = rvs.nearest_valid_parameters(values)
    assert new == values


def test_validate_parameters():
    a, b, c, d = (symbol('a'), symbol('b'), symbol('c'), symbol('d'))
    dist1 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [[a, b], [c, d]],
    )
    dist2 = NormalDistribution.create('ETA(3)', 'iiv', 0.5, d)
    rvs = RandomVariables.create([dist1, dist2])
    params = {'a': 2, 'b': 0.1, 'c': 1, 'd': 23}
    assert rvs.validate_parameters(params)
    params2 = {'a': 2, 'b': 2, 'c': 23, 'd': 1}
    assert not rvs.validate_parameters(params2)
    with pytest.raises(TypeError):
        rvs.validate_parameters({})


def test_sample():
    dist = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [[symbol('a'), symbol('b')], [symbol('b'), symbol('c')]],
    )

    rv1, rv2 = list(map(symbol, dist.names))
    rvs = RandomVariables.create([dist])

    params = {'a': 1, 'b': 0.1, 'c': 2}
    samples = rvs.sample(rv1 + rv2, parameters=params, samples=2, rng=9532)
    assert list(samples) == pytest.approx([1.7033555824617346, -1.4031809274765599])

    with pytest.raises(ValueError):
        rvs.sample(rv1 + rv2, samples=1, rng=np.random.default_rng(9532))

    samples = rvs.sample(1, samples=2)
    assert list(samples) == [1.0, 1.0]

    with pytest.raises(ValueError):
        rvs.sample(symbol('ETA3'), parameters=params)


def test_variance_parameters():
    dist1 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')],
        ],
    )
    dist2 = NormalDistribution.create('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rvs = RandomVariables.create([dist1, dist2])
    assert rvs.variance_parameters == ['OMEGA(1,1)', 'OMEGA(2,2)', 'OMEGA(3,3)']

    dist1 = NormalDistribution.create('x', 'iiv', 0, symbol('omega'))
    dist2 = NormalDistribution.create('y', 'iiv', 0, symbol('omega'))
    rvs = RandomVariables.create([dist1, dist2])
    assert rvs.variance_parameters == ['omega']

    dist3 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(1,1)')],
        ],
    )
    rvs = RandomVariables.create([dist3])
    assert rvs.variance_parameters == ['OMEGA(1,1)']


def test_get_variance():
    dist1 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')],
        ],
    )
    dist2 = NormalDistribution.create('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rvs = RandomVariables.create([dist1, dist2])
    assert rvs['ETA(1)'].get_variance('ETA(1)') == symbol('OMEGA(1,1)')
    assert rvs['ETA(3)'].get_variance('ETA(3)') == symbol('OMEGA(3,3)')

    with pytest.raises(KeyError):
        dist2.get_variance('ETA(5)')


def test_get_covariance():
    dist1 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')],
        ],
    )
    dist2 = NormalDistribution.create('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rvs = RandomVariables.create([dist1, dist2])
    assert rvs.get_covariance('ETA(1)', 'ETA(2)') == symbol('OMEGA(2,1)')
    assert rvs.get_covariance('ETA(3)', 'ETA(2)') == 0

    assert dist2.get_covariance('ETA(3)', 'ETA(3)') == symbol('OMEGA(3,3)')

    with pytest.raises(KeyError):
        dist2.get_covariance('ETA(1)', 'ETA(1)')


def test_unjoin():
    dist1 = JointNormalDistribution.create(
        ['eta1', 'eta2', 'eta3'], 'iiv', [1, 2, 3], [[9, 2, 3], [4, 8, 6], [1, 2, 9]]
    )
    rvs = RandomVariables.create([dist1])
    new = rvs.unjoin('eta1')
    assert new.nrvs == 3
    assert isinstance(new['eta1'], NormalDistribution)
    assert isinstance(new['eta2'], JointNormalDistribution)


def test_join():
    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    dist2 = NormalDistribution.create('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    dist3 = NormalDistribution.create('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    dist4 = JointNormalDistribution.create(
        ['ETA(4)', 'ETA(5)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(4,4)'), symbol('OMEGA(5,4)')],
            [symbol('OMEGA(5,4)'), symbol('OMEGA(5,5)')],
        ],
    )
    dist5 = NormalDistribution.create('EPS(1)', 'ruv', 0, symbol('SIGMA(1,1)'))

    rvs = RandomVariables.create([dist1, dist2])
    joined_rvs, _ = rvs.join(['ETA(1)', 'ETA(2)'])
    assert len(joined_rvs) == 1
    assert joined_rvs[0].variance == sympy.Matrix(
        [[symbol('OMEGA(1,1)'), 0], [0, symbol('OMEGA(2,2)')]]
    )

    joined_rvs, _ = rvs.join(['ETA(1)', 'ETA(2)'], fill=1)
    assert joined_rvs[0].variance == sympy.Matrix(
        [[symbol('OMEGA(1,1)'), 1], [1, symbol('OMEGA(2,2)')]]
    )

    joined_rvs, _ = rvs.join(
        ['ETA(1)', 'ETA(2)'], name_template='IIV_{}_IIV_{}', param_names=['CL', 'V']
    )
    assert joined_rvs[0].variance == sympy.Matrix(
        [
            [symbol('OMEGA(1,1)'), symbol('IIV_CL_IIV_V')],
            [symbol('IIV_CL_IIV_V'), symbol('OMEGA(2,2)')],
        ]
    )

    rvs3 = RandomVariables.create([dist1, dist2, dist3])
    joined_rvs, _ = rvs3.join(['ETA(2)', 'ETA(3)'])
    assert joined_rvs[1].variance == sympy.Matrix(
        [[symbol('OMEGA(2,2)'), 0], [0, symbol('OMEGA(3,3)')]]
    )

    rvs5 = RandomVariables.create([dist1, dist2, dist3, dist4, dist5])
    with pytest.raises(KeyError):
        rvs5.join(['ETA(1)', 'ETA(23)'])


def test_parameters_sdcorr():
    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    dist2 = NormalDistribution.create('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rvs = RandomVariables.create([dist1, dist2])
    params = rvs.parameters_sdcorr({'OMEGA(1,1)': 4})
    assert params == {'OMEGA(1,1)': 2}
    params = rvs.parameters_sdcorr({'OMEGA(1,1)': 4, 'OMEGA(2,2)': 16})
    assert params == {'OMEGA(1,1)': 2, 'OMEGA(2,2)': 4}

    dist2 = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [[symbol('x'), symbol('y')], [symbol('y'), symbol('z')]],
    )
    rvs = RandomVariables.create([dist2])
    params = rvs.parameters_sdcorr({'x': 4, 'y': 0.5, 'z': 16, 'k': 23})
    assert params == {'x': 2.0, 'y': 0.0625, 'z': 4.0, 'k': 23}


def test_variability_level():
    level = VariabilityLevel('IIV', reference=True, group='ID')
    assert level.name == 'IIV'
    assert level.reference is True
    assert level.group == 'ID'

    new = level.replace(group='L1')
    assert new.name == 'IIV'
    assert new.reference is True
    assert new.group == 'L1'
    assert hash(level) != hash(new)

    new = level.replace(group='L1', reference=False)
    assert new.name == 'IIV'
    assert new.reference is False
    assert new.group == 'L1'

    assert repr(level) == "VariabilityLevel(IIV, reference=True, group=ID)"

    d = level.to_dict()
    assert d == {'name': 'IIV', 'reference': True, 'group': 'ID'}
    level2 = VariabilityLevel.from_dict(d)
    assert level == level2


def test_variability_hierarchy():
    lev1 = VariabilityLevel('IIV', reference=True, group='ID')
    levs = VariabilityHierarchy((lev1,))
    assert levs[0].name == 'IIV'
    with pytest.raises(IndexError):
        levs[1].name
    lev2 = VariabilityLevel('CENTER', reference=False, group='CENTER')
    levs2 = VariabilityHierarchy((lev2, lev1))
    assert hash(levs) != hash(levs2)
    assert len(levs2) == 2
    lev3 = VariabilityLevel('PLANET', reference=False, group='PLANET')
    levs3 = VariabilityHierarchy((lev3, lev2, lev1))
    assert len(levs3) == 3

    levs4 = levs + lev2
    assert len(levs4) == 2
    levs5 = lev2 + levs
    assert len(levs5) == 2

    d = levs2.to_dict()
    assert d == {
        'levels': (
            {'name': 'CENTER', 'group': 'CENTER', 'reference': False},
            {'name': 'IIV', 'group': 'ID', 'reference': True},
        )
    }
    levs6 = VariabilityHierarchy.from_dict(d)
    assert levs2 == levs6


def test_covariance_matrix():
    dist1 = NormalDistribution.create('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    dist2 = NormalDistribution.create('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rvs = RandomVariables.create([dist1, dist2])
    assert len(rvs.covariance_matrix) == 4

    dist3 = JointNormalDistribution.create(
        ['ETA(3)', 'ETA(4)'],
        'iiv',
        [0, 0],
        [[symbol('x'), symbol('y')], [symbol('y'), symbol('z')]],
    )
    rvs = RandomVariables.create([dist1, dist2, dist3])
    cov = rvs.covariance_matrix
    assert cov == sympy.Matrix(
        [
            [symbol('OMEGA(1,1)'), 0, 0, 0],
            [0, symbol('OMEGA(2,2)'), 0, 0],
            [0, 0, symbol('x'), symbol('y')],
            [0, 0, symbol('y'), symbol('z')],
        ]
    )

    rvs = RandomVariables.create([dist1, dist3, dist2])
    cov = rvs.covariance_matrix
    assert cov == sympy.Matrix(
        [
            [symbol('OMEGA(1,1)'), 0, 0, 0],
            [0, symbol('x'), symbol('y'), 0],
            [0, symbol('y'), symbol('z'), 0],
            [0, 0, 0, symbol('OMEGA(2,2)')],
        ]
    )


def test_get_rvs_with_same_dist():
    var1 = symbol('OMEGA(1,1)')

    dist1 = NormalDistribution.create('ETA1', 'iov', 0, var1)
    dist2 = NormalDistribution.create('ETA2', 'iov', 0, var1)
    dist3 = NormalDistribution.create('ETA3', 'iov', 0, 1)

    rvs = RandomVariables.create([dist1, dist2, dist3])
    same_dist = rvs.get_rvs_with_same_dist('ETA1')
    assert same_dist == RandomVariables.create([dist1, dist2])

    var2 = symbol('OMEGA(2,2)')
    cov = symbol('OMEGA(1,2)')
    cov_matrix = [[var1, cov], [cov, var2]]

    dist1 = JointNormalDistribution.create(['ETA1', 'ETA2'], 'iov', [0, 0], cov_matrix)
    dist2 = JointNormalDistribution.create(['ETA3', 'ETA4'], 'iov', [0, 0], cov_matrix)
    dist3 = NormalDistribution.create('ETA5', 'iov', 0, var1)

    rvs = RandomVariables.create([dist1, dist2, dist3])
    same_dist = rvs.get_rvs_with_same_dist('ETA1')
    assert same_dist == RandomVariables.create([dist1, dist2])


def test_evalf():
    var1 = symbol('OMEGA11')
    dist1 = NormalDistribution.create('ETA1', 'iiv', 0, var1)
    with pytest.raises(ValueError):
        dist1.evalf({})


def test_levels():
    dist1 = NormalDistribution.create('ETA1', 'iiv', 0, 'omega')
    rvs = RandomVariables.create([dist1])
    assert len(rvs.epsilon_levels) == 1
    assert len(rvs.eta_levels) == 2

    assert rvs.eta_levels.levels == {'IIV': 0, 'IOV': 1}
    assert rvs.epsilon_levels.levels == {'RUV': 0}

    with pytest.raises(ValueError):
        rvs.eta_levels + 23

    with pytest.raises(ValueError):
        23 + rvs.eta_levels

    assert len(rvs.eta_levels[['IIV', 'IOV']]) == 2
    assert len(rvs.eta_levels[rvs.eta_levels]) == 2
    with pytest.raises(ValueError):
        rvs.eta_levels[['IOV']]
    assert rvs.eta_levels['IIV'].group == 'ID'
    assert rvs.eta_levels[rvs.eta_levels['IIV']].group == 'ID'

    lev1 = VariabilityLevel('center', reference=True, group='CENTER')
    levs = VariabilityHierarchy([lev1])
    assert levs != rvs.epsilon_levels
    assert rvs.epsilon_levels != rvs.eta_levels
    assert rvs.epsilon_levels != 23

    levs2 = levs.replace(levels=[lev1])
    assert len(levs2) == 1

    lev2 = VariabilityLevel('center', reference=False, group='CENTER')
    with pytest.raises(ValueError):
        VariabilityHierarchy.create([lev2])
    lev3 = VariabilityLevel('other', reference=True, group='OTHER')
    with pytest.raises(ValueError):
        VariabilityHierarchy.create([lev1, lev3])
    with pytest.raises(ValueError):
        VariabilityHierarchy.create([23])
    with pytest.raises(KeyError):
        rvs.eta_levels[None]

    assert len(VariabilityHierarchy.create(levs)) == 1
    assert len(VariabilityHierarchy.create()) == 0

    assert len(lev2 + rvs.eta_levels) == 3

    x1 = VariabilityLevel.create('other', reference=False, group='OTHER')
    x2 = VariabilityLevel.create('iiv', reference=True, group='ID')
    x3 = VariabilityLevel.create('iov', reference=False, group='OCC')
    h = VariabilityHierarchy.create([x1, x2])
    assert h[x2].group == 'ID'
    with pytest.raises(KeyError):
        h[x3]
    with pytest.raises(KeyError):
        h['NOTHING']


def test_replace_with_sympy_rvs():
    var1 = symbol('OMEGA(1,1)')

    dist = NormalDistribution.create('ETA(1)', 'iiv', 0, var1)
    rvs = RandomVariables.create([dist])

    expr_symbs = symbol('ETA(1)') + symbol('x')
    expr_sympy = rvs.replace_with_sympy_rvs(expr_symbs)

    assert expr_symbs != expr_sympy
    assert not all(isinstance(arg, symbol) for arg in expr_sympy.args)

    var2, cov = symbol('OMEGA(2,2)'), symbol('OMEGA(2,1)')

    dist = JointNormalDistribution.create(
        ['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[var1, cov], [cov, var2]]
    )
    rvs = RandomVariables.create([dist])

    expr_symbs = symbol('ETA(1)') + symbol('ETA(2)') + symbol('x')
    expr_sympy = rvs.replace_with_sympy_rvs(expr_symbs)

    assert expr_symbs != expr_sympy
    assert not all(isinstance(arg, symbol) for arg in expr_sympy.args)
