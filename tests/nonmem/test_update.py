import pytest

from pharmpy.plugins.nonmem.update import primary_pk_param_conversion_map
from pharmpy.symbols import symbol as s


@pytest.mark.parametrize(
    'ncomp,removed,result',
    [
        (
            3,
            1,
            {
                s('K23'): s('K12'),
                s('K32'): s('K21'),
                s('K2T3'): s('K1T2'),
                s('K3T2'): s('K2T1'),
                s('K20'): s('K10'),
                s('K2T0'): s('K1T0'),
            },
        ),
        (
            3,
            2,
            {s('K13'): s('K12'), s('K31'): s('K21'), s('K1T3'): s('K1T2'), s('K3T1'): s('K2T1')},
        ),
    ],
)
def test_primary_parameter_conversion(ncomp, removed, result):
    d = primary_pk_param_conversion_map(ncomp, removed)
    assert d == result
