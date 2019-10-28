import pytest
import sympy
import sympy.stats as stats
from pharmpy.random_variables import RandomVariables


def test_rv():
    omega1 = sympy.Symbol('OMEGA(1,1)')
    x = stats.Normal('ETA(1)', 0, sympy.sqrt(omega1))
    rvs = RandomVariables([x])
    assert len(rvs) == 1
    retrieved = rvs['ETA(1)']
    assert retrieved.name == 'ETA(1)'
    assert retrieved.pspace.distribution.mean == 0
