import copy
import pickle

import pytest
import sympy
import sympy.stats as stats

from pharmpy.model_factory import Model
from pharmpy.random_variables import JointNormalSeparate, RandomVariables, VariabilityLevel
from pharmpy.symbols import symbol


def test_joint_normal_separate():
    rvs = JointNormalSeparate(['ETA(1)', 'ETA(2)'], [0, 0], [[1, 0], [0, 1]])
    assert rvs[0].name == 'ETA(1)'
    assert rvs[1].name == 'ETA(2)'
    assert rvs[0].pspace.distribution.mu == sympy.Matrix([[0], [0]])
    assert rvs[0].pspace.distribution.sigma == sympy.Matrix([[1, 0], [0, 1]])
    assert stats.random_symbols(rvs[0]) == [rvs[0]]
    assert stats.random_symbols(rvs[1]) == [rvs[1]]

    rvs2 = JointNormalSeparate(['ETA(3)', 'ETA(4)'], [1, 1], [[6, 3], [4, 5]])
    # Check that previously created rvs are still intact
    assert rvs[0].name == 'ETA(1)'
    assert rvs[1].name == 'ETA(2)'
    assert rvs[0].pspace.distribution.mu == sympy.Matrix([[0], [0]])
    assert rvs[0].pspace.distribution.sigma == sympy.Matrix([[1, 0], [0, 1]])
    assert stats.random_symbols(rvs[0]) == [rvs[0]]
    assert stats.random_symbols(rvs[1]) == [rvs[1]]
    assert len(rvs2) == 2


def test_rv():
    omega1 = symbol('OMEGA(1,1)')
    x = stats.Normal('ETA(1)', 0, sympy.sqrt(omega1))
    rvs = RandomVariables([x])
    assert len(rvs) == 1
    retrieved = rvs['ETA(1)']
    assert retrieved.name == 'ETA(1)'
    assert retrieved.pspace.distribution.mean == 0
    assert rvs.free_symbols == {symbol('ETA(1)'), omega1}


def test_distributions():
    rvs = JointNormalSeparate(['ETA(1)', 'ETA(2)'], [0, 0], [[3, 0.25], [0.25, 1]])
    rvs = RandomVariables(rvs)
    rvs.add(stats.Normal('ETA(3)', 0.5, 2))
    dists = rvs.distributions()
    symbols, dist = dists[0]
    assert symbols[0].name == 'ETA(1)'
    assert symbols[1].name == 'ETA(2)'
    assert len(symbols) == 2
    assert dist == rvs[0].pspace.distribution
    symbols, dist = dists[1]
    assert symbols[0].name == 'ETA(3)'
    assert len(symbols) == 1
    assert dist == rvs[2].pspace.distribution


def test_merge_normal_distributions():
    rvs = JointNormalSeparate(['ETA(1)', 'ETA(2)'], [0, 0], [[3, 0.25], [0.25, 1]])
    for rv in rvs:
        rv.variability_level = VariabilityLevel.IIV
    rvs = RandomVariables(rvs)
    eta3 = stats.Normal('ETA(3)', 0.5, 2)
    eta3.variability_level = VariabilityLevel.IIV
    rvs.add(eta3)
    rvs.merge_normal_distributions()
    assert len(rvs) == 3
    assert rvs['ETA(1)'].name == 'ETA(1)'
    assert rvs[1].name == 'ETA(2)'
    assert rvs[2].name == 'ETA(3)'
    assert rvs[0].pspace is rvs[1].pspace
    assert rvs[0].pspace is rvs[2].pspace
    dist = rvs[0].pspace.distribution
    assert dist.mu == sympy.Matrix([0, 0, 0.5])
    assert dist.sigma == sympy.Matrix([[3, 0.25, 0], [0.25, 1, 0], [0, 0, 4]])
    # rvs.merge_normal_distributions(fill=1)
    # dist = rvs[0].pspace.distribution
    # assert dist.sigma == sympy.Matrix([[3, 0.25, 1], [0.25, 1, 1], [1, 1, 4]])


def test_validate_parameters():
    a, b, c, d = (symbol('a'), symbol('b'), symbol('c'), symbol('d'))
    rvs = JointNormalSeparate(['ETA(1)', 'ETA(2)'], [0, 0], [[a, b], [b, c]])
    rvs = RandomVariables(rvs)
    rvs.add(stats.Normal('ETA(3)', 0.5, d))
    params = {'a': 2, 'b': 0.1, 'c': 1, 'd': 23}
    assert rvs.validate_parameters(params)
    params2 = {'a': 2, 'b': 2, 'c': 1, 'd': 23}
    assert not rvs.validate_parameters(params2)


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


def test_extract_from_block():
    etas = JointNormalSeparate(['ETA(1)', 'ETA(2)'], [0, 0], [[3, 0.25], [0.25, 1]])
    for rv in etas:
        rv.variability_level = VariabilityLevel.IIV
    rvs = RandomVariables(etas)
    eta3 = stats.Normal('ETA(3)', 0.5, 2)
    eta3.variability_level = VariabilityLevel.IIV
    rvs.add(eta3)

    dists = rvs.distributions()
    assert len(dists) == 2

    rvs.extract_from_block(etas[0])
    dists = rvs.distributions()
    assert len(dists) == 3
    assert rvs[0].name == 'ETA(1)'
    assert rvs[2].name == 'ETA(3)'


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
