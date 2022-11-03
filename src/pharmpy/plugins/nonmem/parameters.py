from __future__ import annotations

from pharmpy.deps import sympy

from .nmtran_parser import NMTranControlStream


def parameter_translation(
    control_stream: NMTranControlStream, reverse=False, remove_idempotent=False, as_symbols=False
):
    """Get a dict of NONMEM name to Pharmpy parameter name
    i.e. {'THETA(1)': 'TVCL', 'OMEGA(1,1)': 'IVCL'}
    """
    d = {}
    for theta_record in control_stream.get_records('THETA'):
        for key, value in theta_record.name_map.items():
            nonmem_name = f'THETA({value})'
            d[nonmem_name] = key
    for record in control_stream.get_records('OMEGA'):
        for key, value in record.name_map.items():
            nonmem_name = f'OMEGA({value[0]},{value[1]})'
            d[nonmem_name] = key
    for record in control_stream.get_records('SIGMA'):
        for key, value in record.name_map.items():
            nonmem_name = f'SIGMA({value[0]},{value[1]})'
            d[nonmem_name] = key
    if remove_idempotent:
        d = {key: val for key, val in d.items() if key != val}
    if reverse:
        d = {val: key for key, val in d.items()}
    if as_symbols:
        d = {sympy.Symbol(key): sympy.Symbol(val) for key, val in d.items()}
    return d
