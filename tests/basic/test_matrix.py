import pytest

from pharmpy.basic import Matrix


def test_init():
    m1 = Matrix(((1, 0), (0, 1)))
    assert len(m1) == 4
    assert m1.rows == 2
    assert m1.cols == 2

    m2 = Matrix(m1)
    assert m1 == m2


def test_repr():
    m = Matrix(((1, 0), (0, 1)))
    assert repr(m) == '⎡1  0⎤\n⎢    ⎥\n⎣0  1⎦'


def test_add():
    m1 = Matrix(((1, 0), (0, 1)))
    m2 = m1 + m1
    assert m2[(0, 0)] == 2
    m3 = ((1, 0), (0, 1))
    m4 = m1 + m3
    assert m4[(0, 0)] == 2
    m5 = m3 + m1

    assert m2 == m4 == m5

    with pytest.raises(TypeError):
        m1 + 1
    with pytest.raises(TypeError):
        1 + m1


def test_cholesky():
    # Example taken from: https://docs.sympy.org/latest/modules/matrices/matrices.html
    m = Matrix(((25, 15, -5), (15, 18, 0), (-5, 0, 11)))
    assert m.cholesky() == Matrix(((5, 0, 0), (3, 3, 0), (-1, 1, 3)))


def test_matmul():
    m1 = Matrix(((1, 0), (0, 1)))
    m2 = m1 @ m1
    assert m2[(0, 0)] == 1
    m3 = ((1, 0), (0, 1))
    m4 = m1 @ m3
    assert m4[(0, 0)] == 1
    m5 = m3 @ m1
    assert m5[(0, 0)] == 1

    assert m2 == m4 == m5

    with pytest.raises(TypeError):
        m1 @ 1
    with pytest.raises(TypeError):
        1 @ m1


def test_eigenvals():
    m1 = Matrix(((1, 0), (0, 1)))
    evs = m1.eigenvals()
    assert evs == {1: 2}
