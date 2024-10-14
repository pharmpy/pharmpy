from pharmpy.deps import sympy


def is_zero_matrix(A: sympy.Matrix) -> bool:
    for e in A:
        assert isinstance(e, sympy.Expr)
        if not e.is_zero:
            return False
    return True
