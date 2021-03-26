import copy
import pickle

import pytest
import sympy
import numpy as np
import sympy.stats as stats

from pharmpy.random_variables import RandomVariable, RandomVariables
from pharmpy.symbols import symbol


def test_normal_rv():
    rv = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    assert rv.name == 'ETA(1)'
    assert rv.symbol == symbol('ETA(1)')
    assert rv.level == 'IIV'
    with pytest.raises(ValueError):
        rvbad = RandomVariable.normal('ETA(1)', 'uuu', 0, 1)


def test_joint_normal_rv():
    rv1, rv2 = RandomVariable.joint_normal(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    assert rv1.name == 'ETA(1)'
    assert rv2.name == 'ETA(2)'
    assert rv1.symbol == symbol('ETA(1)')
    assert rv2.symbol == symbol('ETA(2)')
    assert rv1.level == 'IIV'
    assert rv2.level == 'IIV'


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


def test_repr_rv():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    assert repr(rv1) == 'ETA(1) ~ 𝒩 (0, 1)\n'
    rv1, rv2 = RandomVariable.joint_normal(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    assert repr(rv1) == """⎡ETA(1)⎤     ⎧⎡0⎤  ⎡ 1   0.1⎤⎫
⎢      ⎥ ~ 𝒩 ⎪⎢ ⎥, ⎢        ⎥⎪
⎣ETA(2)⎦     ⎩⎣0⎦  ⎣0.1   2 ⎦⎭
"""

def test_repr_latex_rv():
    rv1, rv2 = RandomVariable.joint_normal(['x', 'y'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    assert rv1._repr_latex_() == '$\\displaystyle \\left[\\begin{matrix}x\\\\y\\end{matrix}\\right]\\sim \\mathcal{N} \\left(\\displaystyle \\left[\\begin{matrix}0\\\\0\\end{matrix}\\right],\\displaystyle \\left[\\begin{matrix}1 & 0.1\\\\0.1 & 2\\end{matrix}\\right]\\right)$'

    rv = RandomVariable.normal('x', 'iiv', 0, 1)
    assert rv._repr_latex_() == '$\\displaystyle x\\sim  \\mathcal{N} \\left(\\displaystyle 0,\\displaystyle 1\\right)$'


def test_copy_rv():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = rv1.copy()
    assert rv1 is not rv2
    assert rv1 == rv2


def test_parameters_rv():
    rv1 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    assert rv1.parameter_names == ['OMEGA(2,2)']


def test_len():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rvs = RandomVariables([rv1, rv2])
    assert len(rvs) == 2


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


def test_delitem():
    rv1 = RandomVariable.normal('ETA', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA2', 'iiv', 0, 0.1)
    rv3 = RandomVariable.normal('EPS', 'ruv', 0, 0.1)
    rvs = RandomVariables([rv1, rv2, rv3])
    del rvs['ETA']
    assert len(rvs) == 2
    assert rvs.names == ['ETA2', 'EPS']

    rv1, rv2 = RandomVariable.joint_normal(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    rvs = RandomVariables([rv1, rv2])
    del rvs[0]
    assert len(rvs) == 1


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


def test_free_symbols():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, 1)
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rvs = RandomVariables([rv1, rv2])
    assert rvs.free_symbols == {symbol('ETA(1)'), symbol('ETA(2)'), symbol('OMEGA(2,2)')}


def test_parameter_names():
    rv1, rv2 = RandomVariable.joint_normal(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')], [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')]])
    rv3 = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rvs = RandomVariables([rv1, rv2, rv3])
    assert rvs.parameter_names == ['OMEGA(1,1)', 'OMEGA(2,1)', 'OMEGA(2,2)', 'OMEGA(3,3)']


def test_distributions():
    rv1, rv2 = RandomVariable.joint_normal(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
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
    rv1, rv2 = RandomVariable.joint_normal(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 2, 1)
    rvs = RandomVariables([rv1, rv2, rv3])
    res = """⎡ETA(1)⎤     ⎧⎡0⎤  ⎡ 1   0.1⎤⎫
⎢      ⎥ ~ 𝒩 ⎪⎢ ⎥, ⎢        ⎥⎪
⎣ETA(2)⎦     ⎩⎣0⎦  ⎣0.1   2 ⎦⎭
ETA(3) ~ 𝒩 (2, 1)
"""
    assert str(rvs) == res


def test_repr_latex():
    rv1 = RandomVariable.normal('x', 'iiv', 0, 1)
    rv2, rv3 = RandomVariable.joint_normal(['x', 'y'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    rvs = RandomVariables([rv1, rv2, rv3])
    assert rvs._repr_latex_() == '\\begin{align*}\n\\displaystyle x & \\sim  \\mathcal{N} \\left(\\displaystyle 0,\\displaystyle 1\\right) \\\\ \\displaystyle \\left[\\begin{matrix}x\\\\y\\end{matrix}\\right] & \\sim \\mathcal{N} \\left(\\displaystyle \\left[\\begin{matrix}0\\\\0\\end{matrix}\\right],\\displaystyle \\left[\\begin{matrix}1 & 0.1\\\\0.1 & 2\\end{matrix}\\right]\\right)\\end{align*}'


def test_copy():
    rv1, rv2 = RandomVariable.joint_normal(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[1, 0.1], [0.1, 2]])
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 2, 1)
    rvs = RandomVariables([rv1, rv2, rv3])
    rvs2 = rvs.copy()
    assert rvs == rvs2
    assert rvs is not rvs2


def test_nearest_valid_parameters():
    values = {'x': 1, 'y': 0.1, 'z': 2}
    rv1, rv2 = RandomVariable.joint_normal(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[symbol('x'), symbol('y')], [symbol('y'), symbol('z')]])
    rvs = RandomVariables([rv1, rv2])
    new = rvs.nearest_valid_parameters(values)
    assert values == new
    values = {'x': 1, 'y': 1.1, 'z': 1}
    new = rvs.nearest_valid_parameters(values)
    assert new == {'x': 1.0500000000000005, 'y': 1.0500000000000003, 'z': 1.050000000000001}


def test_validate_parameters():
    a, b, c, d = (symbol('a'), symbol('b'), symbol('c'), symbol('d'))
    rv1, rv2 = RandomVariable.joint_normal(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[symbol('a'), symbol('b')], [symbol('c'), symbol('d')]])
    rvs = RandomVariables([rv1, rv2])
    rv3 = RandomVariable.normal('ETA(3)', 'iiv', 0.5, d)
    rvs.append(rv3)
    params = {'a': 2, 'b': 0.1, 'c': 1, 'd': 23}
    assert rvs.validate_parameters(params)
    params2 = {'a': 2, 'b': 2, 'c': 23, 'd': 1}
    assert not rvs.validate_parameters(params2)


def test_sample():
    rv1, rv2 = RandomVariable.joint_normal(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[symbol('a'), symbol('b')], [symbol('b'), symbol('c')]])
    rvs = RandomVariables([rv1, rv2])
    params = {'a': 1, 'b': 0.1, 'c': 2}
    np.random.seed(9532)
    samples = rvs.sample(rv1.symbol + rv2.symbol, parameters=params, samples=2)
    assert list(samples) == pytest.approx([-0.5628150524258084, 0.8789382930246744])


def test_variance_parameters():
    rv1, rv2 = RandomVariable.joint_normal(['ETA(1)', 'ETA(2)'], 'iiv', [0, 0], [[symbol('OMEGA(1,1)'), symbol('OMEGA(2,1)')], [symbol('OMEGA(2,1)'), symbol('OMEGA(2,2)')]])
    rv3 = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(3,3)'))
    rvs = RandomVariables([rv1, rv2, rv3])
    assert rvs.variance_parameters == ['OMEGA(1,1)', 'OMEGA(2,2)', 'OMEGA(3,3)']


def test_join():
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rvs = RandomVariables([rv1, rv2])
    rvs.join(['ETA(1)', 'ETA(2)'])
    assert rv1.sympy_rv.pspace.distribution.sigma == sympy.Matrix([[symbol('OMEGA(1,1)'), 0], [0, symbol('OMEGA(2,2)')]])
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rvs = RandomVariables([rv1, rv2])
    rvs.join(['ETA(1)', 'ETA(2)'], fill=1)
    assert rv1.sympy_rv.pspace.distribution.sigma == sympy.Matrix([[symbol('OMEGA(1,1)'), 1], [1, symbol('OMEGA(2,2)')]])
    rv1 = RandomVariable.normal('ETA(1)', 'iiv', 0, symbol('OMEGA(1,1)'))
    rv2 = RandomVariable.normal('ETA(2)', 'iiv', 0, symbol('OMEGA(2,2)'))
    rvs = RandomVariables([rv1, rv2])
    rvs.join(['ETA(1)', 'ETA(2)'], name_template='IIV_{}_IIV_{}', param_names=['CL', 'V'])
    assert rv1.sympy_rv.pspace.distribution.sigma == sympy.Matrix([[symbol('OMEGA(1,1)'), symbol('IIV_CL_IIV_V')], [symbol('IIV_CL_IIV_V'), symbol('OMEGA(2,2)')]])


"""


def test_all_parameters():
    omega1 = symbol('OMEGA(1,1)')
    eta1 = stats.Normal('ETA(1)', 0, sympy.sqrt(omega1))
    omega2 = symbol('OMEGA(2,2)')
    eta2 = stats.Normal('ETA(2)', 0, sympy.sqrt(omega2))
    sigma = symbol('SIGMA(1,1)')
    eps = stats.Normal('EPS(1)', 0, sympy.sqrt(sigma))

    rvs = RandomVariables([eta1, eta2, eps])

    assert len(rvs) == 3

    params = rvs.all_parameters()

    assert len(params) == 3
    assert params == ['OMEGA(1,1)', 'OMEGA(2,2)', 'SIGMA(1,1)']


@pytest.mark.parametrize(
    'model_file,expected_length',
    [
        ('nonmem/pheno_real.mod', 3),
        ('nonmem/frem/pheno/model_3.mod', 12),
        ('nonmem/frem/pheno/model_4.mod', 12),
    ],
)
def test_all_parameters_models(testdata, model_file, expected_length):
    model_path = testdata / model_file
    model = Model(model_path)

    assert len(model.random_variables.all_parameters()) == expected_length
    assert len(model.parameters) != len(model.random_variables.all_parameters())


def test_copy(testdata):
    model = Model(testdata / 'nonmem' / 'pheno.mod')
    rvs = model.random_variables.copy()
    for rv in rvs:
        assert bool(rv.variability_level)

    rvs = copy.deepcopy(model.random_variables)
    for rv in rvs:
        assert bool(rv.variability_level)

    ser = pickle.dumps(model.random_variables)
    rvs = pickle.loads(ser)
    for rv in rvs:
        assert bool(rv.variability_level)


def test_has_same_order():
    omega1 = symbol('OMEGA(1,1)')
    eta1 = stats.Normal('ETA(1)', 0, sympy.sqrt(omega1))
    omega2 = symbol('OMEGA(2,2)')
    eta2 = stats.Normal('ETA(2)', 0, sympy.sqrt(omega2))
    omega3 = symbol('OMEGA(1,1)')
    eta3 = stats.Normal('ETA(3)', 0, sympy.sqrt(omega3))

    rvs_full = RandomVariables([eta1, eta2, eta3])
    assert rvs_full.are_consecutive(rvs_full)

    rvs_sub = RandomVariables([eta1, eta2])
    assert rvs_full.are_consecutive(rvs_sub)

    rvs_rev = RandomVariables([eta3, eta2, eta1])
    assert not rvs_full.are_consecutive(rvs_rev)


def test_get_connected_iovs():
    omega = symbol('OMEGA(1,1)')
    eta1 = stats.Normal('ETA(1)', 0, sympy.sqrt(omega))
    eta2 = stats.Normal('ETA(2)', 0, sympy.sqrt(omega))
    eta3 = stats.Normal('ETA(3)', 0, sympy.sqrt(omega))

    omega = symbol('OMEGA(2,2)')
    eta4 = stats.Normal('ETA(4)', 0, sympy.sqrt(omega))
    eta5 = stats.Normal('ETA(5)', 0, sympy.sqrt(omega))

    rvs_iovs = RandomVariables([eta1, eta2, eta3, eta4, eta5])

    for rv in rvs_iovs:
        rv.variability_level = VariabilityLevel.IOV

    omega_iiv = symbol('OMEGA(3,3)')
    eta_iiv = stats.Normal('ETA(6)', 0, sympy.sqrt(omega_iiv))
    eta_iiv.variability_level = VariabilityLevel.IIV

    rvs = copy.deepcopy(rvs_iovs)
    rvs.add(eta_iiv)

    assert len(rvs.get_connected_iovs(eta1)) == 3
    assert len(rvs.get_connected_iovs(eta4)) == 2
    assert len(rvs.get_connected_iovs(eta_iiv)) == 1
"""
