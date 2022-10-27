from __future__ import annotations

from pharmpy.deps import sympy

from .nmtran_parser import NMTranControlStream


def rv_translation(
    control_stream: NMTranControlStream, reverse=False, remove_idempotent=False, as_symbols=False
):
    d = {}
    for record in control_stream.get_records('OMEGA'):
        for key, value in record.eta_map.items():
            nonmem_name = f'ETA({value})'
            d[nonmem_name] = key
    for record in control_stream.get_records('SIGMA'):
        for key, value in record.eta_map.items():
            nonmem_name = f'EPS({value})'
            d[nonmem_name] = key
    if remove_idempotent:
        d = {key: val for key, val in d.items() if key != val}
    if reverse:
        d = {val: key for key, val in d.items()}
    if as_symbols:
        d = {sympy.Symbol(key): sympy.Symbol(val) for key, val in d.items()}
    return d
