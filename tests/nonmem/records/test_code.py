from sympy import Symbol


def test_something(parser):
    rec = parser.parse('$PRED\nY = THETA(1) + ETA(1) + EPS(1)').records[0]
    rec.root.treeprint()
    assert len(rec.statements) == 1
    assert rec.statements[0].symbol == Symbol('Y')
    assert rec.statements[0].expression == Symbol('THETA(1)') + Symbol('ETA(1)') + Symbol('EPS(1)')
