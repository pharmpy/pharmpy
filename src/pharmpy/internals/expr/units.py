from pharmpy.deps import sympy

from .parse import parse as parse_expr
from .subs import subs


def parse(s):
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
