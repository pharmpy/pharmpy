from typing import TYPE_CHECKING, Union

from ..unicode import int_to_superscript
from .parse import parse as parse_expr
from .subs import subs

if TYPE_CHECKING:
    import sympy
    import sympy.printing as sympy_printing
else:
    from pharmpy.deps import sympy, sympy_printing


def parse(s: Union[str, int, float, sympy.Expr]) -> sympy.Expr:
    if not isinstance(s, sympy.Expr):
        return subs(parse_expr(s), _unit_subs(), simultaneous=True)
    else:
        return s
    # return (
    #    subs(parse_expr(s), _unit_subs(), simultaneous=True) if not isinstance(s, sympy.Expr) else s
    # )


_unit_subs_cache = None


def _unit_subs():
    global _unit_subs_cache
    if _unit_subs_cache is None:
        subs = {}
        import sympy.physics.units as units

        for k, v in units.__dict__.items():
            if isinstance(v, sympy.Expr) and v.has(units.Unit):
                subs[sympy.Symbol(k)] = v

        _unit_subs_cache = subs

    return _unit_subs_cache


# sympy_printing.str.StrPrinter._default_settings['abbrev'] = True


class UnitPrinter(sympy_printing.str.StrPrinter):
    """Print physical unit as unicode"""

    def _print_Mul(self, expr):
        pow_strings = [self._print(e) for e in expr.args if e.is_Pow]
        plain_strings = [self._print(e) for e in expr.args if not e.is_Pow]
        all_strings = sorted(plain_strings) + sorted(pow_strings)
        return '⋅'.join(all_strings)

    def _print_Pow(self, expr, rational=False):
        base = expr.args[0]
        exp = expr.args[1]
        if exp.is_Integer:
            exp_str = int_to_superscript(int(exp))
        else:
            exp_str = "^" + self._print(exp)
        return self._print(base) + exp_str

    def _print_Quantity(self, expr):
        # FIXME: sympy cannot handle the abbreviation of ml
        if str(expr) == "milliliter":
            return "ml"
        else:
            return str(expr.args[1])


def unit_string(expr: sympy.Expr) -> str:
    printer = UnitPrinter()
    return printer._print(expr)
