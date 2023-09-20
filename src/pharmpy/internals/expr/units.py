from typing import TYPE_CHECKING, Union

from .parse import parse as parse_expr
from .subs import subs

if TYPE_CHECKING:
    import sympy
else:
    from pharmpy.deps import sympy


def parse(s: Union[str, sympy.Expr, sympy.Basic]) -> Union[sympy.Expr, sympy.Basic]:
    return subs(parse_expr(s), _unit_subs(), simultaneous=True) if isinstance(s, str) else s


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
