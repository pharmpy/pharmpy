from __future__ import annotations

from pharmpy.deps import sympy

from .nmtran_parser import NMTranControlStream
from .records.omega_record import OmegaRecord
from .records.theta_record import ThetaRecord


def parameter_translation(
    control_stream: NMTranControlStream, reverse=False, remove_idempotent=False, as_symbols=False
):
    """Get a dict of NONMEM name to Pharmpy parameter name
    i.e. {'THETA(1)': 'TVCL', 'OMEGA(1,1)': 'IVCL'}
    """
    d = {}
    for theta_record in control_stream.get_records(ThetaRecord, 'THETA'):
        for key, value in theta_record.name_map.items():
            nonmem_name = f'THETA({value})'
            d[nonmem_name] = key
    eta_indexes = []
    for record in control_stream.get_records(OmegaRecord, 'OMEGA'):
        for key, value in record.name_map.items():
            nonmem_name = f'OMEGA({value[0]},{value[1]})'
            d[nonmem_name] = key
            if value[0] == value[1]:
                eta_indexes.append(value[0])
                if key.startswith('IIV_'):
                    nonmem_name = f'ETA({value[0]})'
                    d[nonmem_name] = f'ETA_{key[4:]}'
                elif key.startswith('OMEGA_IOV_'):
                    # NOTE This only takes care of the non-SAME IOV ETAs
                    nonmem_name = f'ETA({value[0]})'
                    i = int(key[10:])
                    d[nonmem_name] = f'ETA_IOV_{i}_1'

    # NOTE This takes care of the SAME IOV ETAs
    print('eta_indexes', eta_indexes)
    next_eta = min(eta_indexes, default=1)
    assert next_eta == 1
    prev_start = 1
    prev_cov = None
    same_index = 1
    for record in control_stream.get_records(OmegaRecord, 'OMEGA'):
        rvs, next_eta, prev_start, prev_cov, _ = record.random_variables(
            next_eta, prev_start, prev_cov
        )
        if bool(record.tree.find('same')):
            same_index += 1
            assert prev_cov is not None
            assert len(rvs) == 1
            etas = rvs[0].names
            first_eta = next_eta - len(etas)
            for j, eta in enumerate(etas):
                key = str(prev_cov if len(etas) == 1 else prev_cov[j, j])
                if key.startswith('OMEGA_IOV_'):
                    i = int(key[10:])
                    k = first_eta + j
                    assert eta == f'ETA_IOV_{i}_{same_index}'
                    nonmem_name = f'ETA({k})'
                    d[nonmem_name] = eta
        else:
            same_index = 1

    for record in control_stream.get_records(OmegaRecord, 'SIGMA'):
        for key, value in record.name_map.items():
            nonmem_name = f'SIGMA({value[0]},{value[1]})'
            d[nonmem_name] = key
    if remove_idempotent:
        d = {key: val for key, val in d.items() if key != val}
    if reverse:
        d = {val: key for key, val in d.items()}
    if as_symbols:
        d = {sympy.Symbol(key): sympy.Symbol(val) for key, val in d.items()}
    print('PARAMETER_TRANSLATION', d)
    return d
