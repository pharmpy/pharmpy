import pickle

import pytest
import sympy
import sympy.stats as stats

from pharmpy.random_variables import RandomVariable, RandomVariables, VariabilityHierarchy
from pharmpy.symbols import symbol


def test_general_rv():
    rv = RandomVariable('X', 'iiv', stats.Normal('X', 0, 1))
    assert rv.name == 'X'
    assert rv.symbol == symbol('X')
    assert rv.level == 'IIV'
    assert rv.sympy_rv.pspace.distribution.mean == 0
    with pytest.raises(ValueError):
        RandomVariable('X', 'iiv', stats.Normal('X', [0, 0], [[1, 0], [0, 2]]))
    rv = RandomVariable('X', 'iiv', stats.Exponential('X', 1))


def test_normal_rv():
    rv = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    assert rv.name == 'ETA(1)'
    assert rv.symbol == symbol('ETA(1)')
    assert rv.level == 'IIV'
    assert rv.joint_names == []
    rv.name = 'NEW'
    assert rv.name == 'NEW'
    with pytest.raises(ValueError):
        RandomVariable.normal('ETA(1)', 'uuu', 0, 1)
    with pytest.raises(ValueError):
        RandomVariable.normal('X', 'iiv', 0, -1)


def test_joint_normal_rv():
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]]
    )
    assert rv1.name == 'ETA(1)'
    assert rv2.name == 'ETA(2)'
    assert rv1.symbol == symbol('ETA(1)')
    assert rv2.symbol == symbol('ETA(2)')
    assert rv1.level == 'IIV'
    assert rv2.level == 'IIV'
    assert rv1.joint_names == ['ETA(1)', 'ETA(2)']
    assert rv2.joint_names == ['ETA(1)', 'ETA(2)']
    rv1.name = 'NEW'
    assert rv1.name == 'NEW'
    with pytest.raises(ValueError):
        RandomVariable.joint_normal(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[-1, 0.1], [0.1, 2]])


def test_eq_rv():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    assert rv1 == rv2
    rv3 = RandomVariable.normal('ETA(2)', 'iiv', 0, 1)
    assert rv1 != rv3
    rv4 = RandomVariable.normal('ETA(2)', 'iiv', 0, 0.1)
    assert rv3 != rv4


def test_sympy_rv():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    assert rv1.sympy_rv == sympy.stats.Normal('ETA(1)', 0, 1)
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 0)
    assert rv1.sympy_rv == sympy.Integer(0)


def test_repr_rv():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    assert repr(rv1) == 'ETA(1) ~ 𝒩 (0, 1)'
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]]
    )
    assert (
        repr(rv1)
        == """⎡ETA(1)⎤     ⎧⎡0⎤  ⎡ 1   0.1⎤⎫
⎢      ⎥ ~ 𝒩 ⎪⎢ ⎥, ⎢        ⎥⎪
⎣ETA(2)⎦     ⎩⎣0⎦  ⎣0.1   2 ⎦⎭"""
    )


def test_repr_latex_rv():
    rv1, rv2 = RandomVariable.joint_normal(['x', 'y'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    assert (
        rv1._repr_latex_()
        == '$\\displaystyle \\left[\\begin{matrix}x\\\\y\\end{matrix}\\right]\\sim \\mathcal{N} \\left(\\displaystyle \\left[\\begin{matrix}0\\\\0\\end{matrix}\\right],\\displaystyle \\left[\\begin{matrix}1 & 0.1\\\\0.1 & 2\\end{matrix}\\right]\\right)$'  # noqa E501
    )

    rv = RandomVariable.normal('x', 'iiv', 0, 1)
    assert (
        rv._repr_latex_()
        == '$\\displaystyle x\\sim  \\mathcal{N} \\left(\\displaystyle 0,\\displaystyle 1\\right)$'
    )


def test_copy_rv():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = rv1.copy()
    assert rv1 is not rv2
    assert rv1 == rv2


def test_parameters_rv():
    rv1 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    assert rv1.parameter_names == ['OMEGA(2,2)']


def test_init_from_rvs():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, 1)
    rvs = RandomVariables([rv1, rv2])
    rvs2 = RandomVariables(rvs)
    assert len(rvs2) == 2


def test_illegal_inits():
    with pytest.raises(ValueError):
        RandomVariables([8, 1])
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    with pytest.raises(ValueError):
        RandomVariables([rv1, rv2])


def test_len():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, 1)
    rvs = RandomVariables([rv1, rv2])
    assert len(rvs) == 2


def test_eq():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, 1)
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0, 1)
    rvs = RandomVariables([rv1, rv2])
    rvs2 = RandomVariables([rv1])
    assert rvs != rvs2
    rvs3 = RandomVariables([rv1, rv3])
    assert rvs != rvs3


def test_getitem():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2])
    assert rvs[0] == rv1
    assert rvs[1] == rv2
    assert rvs['ETA(1)'] == rv1
    assert rvs['ETA(2)'] == rv2
    assert rvs[symbol('ETA(1)')] == rv1
    assert rvs[symbol('ETA(2)')] == rv2
    with pytest.raises(IndexError):
        rvs[23]
    with pytest.raises(KeyError):
        rvs['NOKEYOFTHIS']

    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, 0.1)
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2, rv3])
    selection = rvs[[rv1, rv2]]
    assert len(selection) == 2


def test_contains():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, 0.1)
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2, rv3])
    assert 'ETA(2)' in rvs
    assert 'ETA(4)' not in rvs


def test_setitem():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, 0.1)
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0, 0.1)
    rvs = RandomVariables([rv1])
    rvs[rv1] = rv2
    assert len(rvs) == 1
    assert rvs[0].name == 'ETA(2)'
    with pytest.raises(KeyError):
        rvs[rv3] = rv2

    rv1, rv2 = RandomVariable.joint_normal(['x', 'y'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    rv3 = RandomVariable.normal('z', 'iiv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2])
    rvs['x'] = rv3

    rv1, rv2, rv3 = RandomVariable.joint_normal(
        ['x', 'y', 'z'], 'iiv', [0, 0, 1], [[1, 0.1, 0.1], [0.1, 4, 0.1], [0.1, 0.1, 9]]
    )
    rv4 = RandomVariable.normal('w', 'iiv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2, rv3])
    rvs['y'] = rv4
    cov = sympy.Matrix([[1, 0.1], [0.1, 9]])
    assert rv1.sympy_rv.pspace.distribution.sigma == cov
    assert rv3.sympy_rv.pspace.distribution.sigma == cov
    assert rvs.names == ['w', 'x', 'z']

    with pytest.raises(ValueError):
        rvs[0] = 0

    rv1, rv2, rv3 = RandomVariable.joint_normal(
        ['x', 'y', 'z'], 'iiv', [0, 0, 1], [[1, 0.1, 0.1], [0.1, 4, 0.1], [0.1, 0.1, 9]]
    )
    rv4 = RandomVariable.normal('w', 'iiv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2, rv3])
    rvs[0:1] = [rv4]
    assert len(rvs) == 3
    assert rvs.names == ['w', 'y', 'z']

    with pytest.raises(ValueError):
        rvs[0:2] = [rv4]

    rv1, rv2, rv3 = RandomVariable.joint_normal(
        ['x', 'y', 'z'], 'iiv', [0, 0, 1], [[1, 0.1, 0.1], [0.1, 4, 0.1], [0.1, 0.1, 9]]
    )
    rv4 = RandomVariable.normal('w', 'iiv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2, rv3])
    rvs[0:3:2] = [rv1, rv4]


def test_insert():
    rv1, rv2 = RandomVariable.joint_normal(['x', 'y'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    rvs = RandomVariables([rv1, rv2])
    with pytest.raises(ValueError):
        rvs.insert(0, 23)


def test_delitem():
    rv1 = RandomVariable.normal('ETA', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA2', 'iiv', 0, 0.1)
    rv3 = RandomVariable.normal('EPS', 'ruv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2, rv3])
    del rvs['ETA']
    assert len(rvs) == 2
    assert rvs.names == ['ETA2', 'EPS']

    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]]
    )
    rvs = RandomVariables([rv1, rv2])
    del rvs[0]
    assert len(rvs) == 1

    rv1, rv2, rv3 = RandomVariable.joint_normal(
        ['x', 'y', 'z'], 'iiv', [0, 0, 1], [[1, 0.1, 0.1], [0.1, 4, 0.1], [0.1, 0.1, 9]]
    )
    rvs = RandomVariables([rv1, rv2, rv3])
    del rvs[1]
    assert len(rvs) == 2


def test_names():
    rv1 = RandomVariable.normal('ETA1', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA2', 'iiv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2])
    assert rvs.names == ['ETA1', 'ETA2']


def test_epsilons():
    rv1 = RandomVariable.normal('ETA', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA2', 'iiv', 0, 0.1)
    rv3 = RandomVariable.normal('EPS', 'ruv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2, rv3])
    assert rvs.epsilons == RandomVariables([rv3])
    assert rvs.epsilons.names == ['EPS']


def test_etas():
    rv1 = RandomVariable.normal('ETA', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA2', 'iiv', 0, 0.1)
    rv3 = RandomVariable.normal('EPS', 'ruv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2, rv3])
    assert rvs.etas == RandomVariables([rv1, rv2])
    assert rvs.etas.names == ['ETA', 'ETA2']


def test_iiv_iov():
    rv1 = RandomVariable.normal('ETA', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA2', 'iov', 0, 0.1)
    rv3 = RandomVariable.normal('EPS', 'ruv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2, rv3])
    assert rvs.iiv == RandomVariables([rv1])
    assert rvs.iiv.names == ['ETA']
    assert rvs.iov == RandomVariables([rv2])
    assert rvs.iov.names == ['ETA2']


def test_subs():
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')],
        ],
    )
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rvs = RandomVariables([rv1, rv2, rv3])
    rvs.subs(
        {
            symbol('ETA(2)'): symbol('w'),
            symbol('OMEGA(1,1)'): symbol('x'),
            symbol('OMEGA(3,3)'): symbol('y'),
        }
    )
    assert rv1.sympy_rv.pspace.distribution.sigma == sympy.Matrix(
        [[symbol('x'), symbol('OMEGA(2,1)')], [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')]]
    )
    assert rv3.sympy_rv.pspace.distribution.std ** 2 == symbol('y')
    assert rvs.names == ['ETA(1)', 'w', 'ETA(3)']


def test_free_symbols():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    assert rv2.free_symbols == {symbol('ETA(2)'), symbol('OMEGA(2,2)')}
    rvs = RandomVariables([rv1, rv2])
    assert rvs.free_symbols == {symbol('ETA(1)'), symbol('ETA(2)'), symbol('OMEGA(2,2)')}
    rv_exp = RandomVariable('X', 'iiv', stats.Exponential('X', 1))
    assert rv_exp.free_symbols == {symbol('X')}


def test_parameter_names():
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')],
        ],
    )
    assert rv1.parameter_names == ['OMEGA(1,1)', 'OMEGA(2,1)', 'OMEGA(2,2)']
    assert rv2.parameter_names == ['OMEGA(1,1)', 'OMEGA(2,1)', 'OMEGA(2,2)']
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    assert rv3.parameter_names == ['OMEGA(3,3)']
    rvs = RandomVariables([rv1, rv2, rv3])
    assert rvs.parameter_names == ['OMEGA(1,1)', 'OMEGA(2,1)', 'OMEGA(2,2)', 'OMEGA(3,3)']
    rv_exp = RandomVariable('X', 'iiv', stats.Exponential('X', symbol('Z')))
    assert rv_exp.parameter_names == ['Z']


def test_subs_rv():
    rv = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rv.subs({symbol('OMEGA(3,3)'): symbol('VAR')})
    assert rv.sympy_rv.pspace.distribution.std == sympy.sqrt(symbol('VAR'))
    rv_exp = RandomVariable('X', 'iiv', stats.Exponential('X', symbol('Z')))
    rv_exp.subs({symbol('Z'): symbol('TV')})
    assert rv_exp.sympy_rv.pspace.distribution.rate == symbol('TV')


def test_distributions():
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]]
    )
    rvs = RandomVariables([rv1, rv2])
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0.5, 2)
    rvs.append(rv3)
    dists = rvs.distributions()
    symbols, dist = dists[0]
    assert symbols[0].name == 'ETA(1)'
    assert symbols[1].name == 'ETA(2)'
    assert len(symbols) == 2
    assert dist == rv1.sympy_rv.pspace.distribution
    symbols, dist = dists[1]
    assert symbols[0].name == 'ETA(3)'
    assert len(symbols) == 1
    assert dist == rv3.sympy_rv.pspace.distribution


def test_repr():
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]]
    )
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 2, 1)
    rvs = RandomVariables([rv1, rv2, rv3])
    res = """⎡ETA(1)⎤     ⎧⎡0⎤  ⎡ 1   0.1⎤⎫
⎢      ⎥ ~ 𝒩 ⎪⎢ ⎥, ⎢        ⎥⎪
⎣ETA(2)⎦     ⎩⎣0⎦  ⎣0.1   2 ⎦⎭
ETA(3) ~ 𝒩 (2, 1)"""
    assert str(rvs) == res
    rv_exp = RandomVariable('X', 'iiv', stats.Exponential('X', symbol('Z')))
    assert str(rv_exp) == 'X ~ Exp(Z)'
    rv_f = RandomVariable('X', 'iiv', stats.FDistribution('X', symbol('Z'), 2))
    assert str(rv_f) == 'X ~ UnknownDistribution'
    rv3, rv4 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'], 'iiv', [sympy.sqrt(sympy.Rational(2, 5)), 0], [[1, 0.1], [0.1, 2]]
    )
    assert (
        str(rv3)
        == '''             ⎧⎡√10⎤            ⎫
⎡ETA(1)⎤     ⎪⎢───⎥  ⎡ 1   0.1⎤⎪
⎢      ⎥ ~ 𝒩 ⎪⎢ 5 ⎥, ⎢        ⎥⎪
⎣ETA(2)⎦     ⎪⎢   ⎥  ⎣0.1   2 ⎦⎪
             ⎩⎣ 0 ⎦            ⎭'''
    )


def test_repr_latex():
    rv1 = RandomVariable.normal('z', 'iiv', 0, 1)
    rv2, rv3 = RandomVariable.joint_normal(['x', 'y'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    rvs = RandomVariables([rv1, rv2, rv3])
    assert (
        rvs._repr_latex_()
        == '\\begin{align*}\n\\displaystyle z & \\sim  \\mathcal{N} \\left(\\displaystyle 0,\\displaystyle 1\\right) \\\\ \\displaystyle \\left[\\begin{matrix}x\\\\y\\end{matrix}\\right] & \\sim \\mathcal{N} \\left(\\displaystyle \\left[\\begin{matrix}0\\\\0\\end{matrix}\\right],\\displaystyle \\left[\\begin{matrix}1 & 0.1\\\\0.1 & 2\\end{matrix}\\right]\\right)\\end{align*}'  # noqa E501
    )


def test_copy():
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]]
    )
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 2, 1)
    rvs = RandomVariables([rv1, rv2, rv3])
    rvs2 = rvs.copy()
    assert rvs == rvs2
    assert rvs is not rvs2
    rv4 = rv3.copy(deep=False)
    assert rv4.name == rv3.name


def test_pickle():
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]]
    )
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 2, 1)
    rvs = RandomVariables([rv1, rv2, rv3])
    pickled = pickle.dumps(rvs)
    obj = pickle.loads(pickled)
    assert obj == rvs


def test_hash():
    rv1 = RandomVariable.normal('ETA(3)', 'iiv', 2, 1)
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 2, 0)
    assert hash(rv1) != hash(rv2)


def test_nearest_valid_parameters():
    values = {'x': 1, 'y': 0.1, 'z': 2}
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [[symbol('x'), symbol('y')], [symbol('y'), symbol('z')]],
    )
    rvs = RandomVariables([rv1, rv2])
    new = rvs.nearest_valid_parameters(values)
    assert values == new
    values = {'x': 1, 'y': 1.1, 'z': 1}
    new = rvs.nearest_valid_parameters(values)
    assert new == {'x': 1.0500000000000005, 'y': 1.0500000000000003, 'z': 1.050000000000001}


def test_validate_parameters():
    a, b, c, d = (symbol('a'), symbol('b'), symbol('c'), symbol('d'))
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [[a, b], [c, d]],
    )
    rvs = RandomVariables([rv1, rv2])
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0.5, d)
    rvs.append(rv3)
    params = {'a': 2, 'b': 0.1, 'c': 1, 'd': 23}
    assert rvs.validate_parameters(params)
    params2 = {'a': 2, 'b': 2, 'c': 23, 'd': 1}
    assert not rvs.validate_parameters(params2)


def test_sample():
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [[symbol('a'), symbol('b')], [symbol('b'), symbol('c')]],
    )
    rvs = RandomVariables([rv1, rv2])
    params = {'a': 1, 'b': 0.1, 'c': 2}
    samples = rvs.sample(rv1.symbol + rv2.symbol, parameters=params, samples=2, seed=9532)
    assert list(samples) == pytest.approx([1.7033555824617346, -1.4031809274765599])


def test_variance_parameters():
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')],
        ],
    )
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rvs = RandomVariables([rv1, rv2, rv3])
    assert rvs.variance_parameters == ['OMEGA(1,1)', 'OMEGA(2,2)', 'OMEGA(3,3)']


def test_get_variance():
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')],
        ],
    )
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rvs = RandomVariables([rv1, rv2, rv3])
    assert rvs.get_variance('ETA(1)') == symbol('OMEGA(1,1)')
    assert rvs.get_variance(rv3) == symbol('OMEGA(3,3)')


def test_get_covariance():
    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')],
            [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')],
        ],
    )
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rvs = RandomVariables([rv1, rv2, rv3])
    assert rvs.get_covariance('ETA(1)', 'ETA(2)') == symbol('OMEGA(2,1)')
    assert rvs.get_covariance(rv3, rv2) == 0


def test_join():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rvs = RandomVariables([rv1, rv2])
    rvs.join(['ETA(1)', 'ETA(2)'])
    assert rv1.sympy_rv.pspace.distribution.sigma == sympy.Matrix(
        [[symbol('OMEGA(1,1)'), 0], [0, symbol('OMEGA(2,2)')]]
    )
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rvs = RandomVariables([rv1, rv2])
    rvs.join(['ETA(1)', 'ETA(2)'], fill=1)
    assert rv1.sympy_rv.pspace.distribution.sigma == sympy.Matrix(
        [[symbol('OMEGA(1,1)'), 1], [1, symbol('OMEGA(2,2)')]]
    )
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rvs = RandomVariables([rv1, rv2])
    rvs.join(['ETA(1)', 'ETA(2)'], name_template='IIV_{}_IIV_{}', param_names=['CL', 'V'])
    assert rv1.sympy_rv.pspace.distribution.sigma == sympy.Matrix(
        [
            [symbol('OMEGA(1,1)'), symbol('IIV_CL_IIV_V')],
            [symbol('IIV_CL_IIV_V'), symbol('OMEGA(2,2)')],
        ]
    )
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rvs = RandomVariables([rv1, rv2, rv3])
    rvs.join(['ETA(2)', 'ETA(3)'])
    assert rv2.sympy_rv.pspace.distribution.sigma == sympy.Matrix(
        [[symbol('OMEGA(2,2)'), 0], [0, symbol('OMEGA(3,3)')]]
    )

    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rv4, rv5 = RandomVariable.joint_normal(
        ['ETA(4)', 'ETA(5)'],
        'iiv',
        [0, 0],
        [
            [symbol('OMEGA(4,4)'), symbol('OMEGA(5,4)')],
            [symbol('OMEGA(5,4)'), symbol('OMEGA(5,5)')],
        ],
    )
    rv6 = RandomVariable.normal('EPS(1)', 'ruv', 0, symbol('SIGMA(1,1)'))
    rvs = RandomVariables([rv1, rv2, rv3, rv4, rv5, rv6])
    rvs_copy = rvs.copy()
    rvs.join(['ETA(1)', 'ETA(2)'])

    rvs = rvs_copy
    rvs.join(['ETA(1)', 'ETA(4)'])
    assert rvs['ETA(4)'].joint_names == ['ETA(1)', 'ETA(4)']
    assert rvs['ETA(5)'].joint_names == []


def test_sub():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rvs = RandomVariables([rv1, rv2])
    rvs2 = RandomVariables([rv1, rv3])
    rvs3 = rvs - rvs2
    assert rvs3.names == ['ETA(2)']


def test_parameters_sdcorr():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rvs = RandomVariables([rv1, rv2])
    params = rvs.parameters_sdcorr({'OMEGA(1,1)': 4})
    assert params == {'OMEGA(1,1)': 2}
    params = rvs.parameters_sdcorr({'OMEGA(1,1)': 4, 'OMEGA(2,2)': 16})
    assert params == {'OMEGA(1,1)': 2, 'OMEGA(2,2)': 4}

    rv1, rv2 = RandomVariable.joint_normal(
        ['ETA(1)', 'ETA(2)'],
        'iiv',
        [0, 0],
        [[symbol('x'), symbol('y')], [symbol('y'), symbol('z')]],
    )
    rvs = RandomVariables([rv1, rv2])
    params = rvs.parameters_sdcorr({'x': 4, 'y': 0.5, 'z': 16, 'k': 23})
    assert params == {'x': 2.0, 'y': 0.0625, 'z': 4.0, 'k': 23}


def test_variability_hierarchy():
    lev = VariabilityHierarchy()
    lev.add_variability_level('IIV', 0, 'ID')
    assert lev.get_name(0) == 'IIV'
    with pytest.raises(KeyError):
        lev.get_name(1)
    with pytest.raises(ValueError):
        lev.add_variability_level('IOV', 2, 'OCC')
    lev.add_variability_level('CENTER', -1, 'CENTER')
    assert len(lev) == 2
    lev.add_lower_level('PLANET', 'PLANET')
    assert len(lev) == 3
    lev.set_variability_level(-1, 'COHORT', 'X2')
    assert lev.get_name(-1) == 'COHORT'
    with pytest.raises(KeyError):
        lev.set_variability_level(2, 'X', 'Y')
    lev.remove_variability_level(-2)
    assert len(lev) == 2
    with pytest.raises(KeyError):
        lev.remove_variability_level(2)
    with pytest.raises(ValueError):
        lev.remove_variability_level('IIV')
    lev = VariabilityHierarchy()
    lev.add_variability_level('IIV', 0, 'ID')
    lev.add_higher_level('IOV', 'OCC')
    lev.add_higher_level('IOVIOV', 'SUBOCC')
    lev.add_lower_level('PLANET', 'PLANET')
    lev.add_lower_level('GALAXY', 'GALAXY')
    lev.remove_variability_level('IOV')
    assert len(lev) == 4
    lev.remove_variability_level('PLANET')
    assert len(lev) == 3
