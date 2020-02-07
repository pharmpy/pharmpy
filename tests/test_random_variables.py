import sympy
import sympy.stats as stats

from pharmpy.random_variables import JointNormalSeparate, RandomVariables


def test_joint_normal_separate():
    rvs = JointNormalSeparate(['ETA(1)', 'ETA(2)'], [0, 0], [[1, 0], [0, 1]])
    assert rvs[0].name == 'ETA(1)'
    assert rvs[1].name == 'ETA(2)'
    assert rvs[0].pspace.distribution.mu == sympy.Matrix([[0], [0]])
    assert rvs[0].pspace.distribution.sigma == sympy.Matrix([[1, 0], [0, 1]])
    assert stats.random_symbols(rvs[0]) == [rvs[0]]
    assert stats.random_symbols(rvs[1]) == [rvs[1]]


def test_rv():
    omega1 = sympy.Symbol('OMEGA(1,1)')
    x = stats.Normal('ETA(1)', 0, sympy.sqrt(omega1))
    rvs = RandomVariables([x])
    assert len(rvs) == 1
    retrieved = rvs['ETA(1)']
    assert retrieved.name == 'ETA(1)'
    assert retrieved.pspace.distribution.mean == 0
