import pytest
from sympy import Symbol

S = Symbol


@pytest.mark.usefixtures('parser')
@pytest.mark.parametrize('buf,symbol,expression', [
    ('$PRED\nY = THETA(1) + ETA(1) + EPS(1)', S('Y'), S('THETA(1)') + S('ETA(1)') + S('EPS(1)')),
    ('$PRED\nCL = 2', S('CL'), 2),
    ('$PRED\nCL = KA', S('CL'), S('KA')),
    ('$PRED\nG = BASE - LESS', S('G'), S('BASE') - S('LESS')),
])
def test_single_assignments(parser, buf, symbol, expression):
    rec = parser.parse(buf).records[0]
    rec.root.treeprint()
    assert len(rec.statements) == 1
    assert rec.statements[0].symbol == symbol
    assert rec.statements[0].expression == expression
