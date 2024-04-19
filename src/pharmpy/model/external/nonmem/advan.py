from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Optional, Tuple

from pharmpy.basic import BooleanExpr, Expr
from pharmpy.deps import sympy
from pharmpy.model import (
    Assignment,
    Bolus,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    DataInfo,
    Infusion,
    ModelSyntaxError,
    output,
    to_compartmental_system,
)

if TYPE_CHECKING:
    from .model import Model

from .nmtran_parser import NMTranControlStream


def compartmental_model(model: Model, advan: str, trans, des=None):
    di = model.datainfo
    control_stream = model.internals.control_stream
    return _compartmental_model(di, model.dataset, control_stream, advan, trans, des)


def _compartmental_model(
    di: DataInfo,
    dataset,
    control_stream: NMTranControlStream,
    advan: str,
    trans,
    des=None,
):
    if advan == 'ADVAN1':
        # FIXME: Require multiple doses per comp before IV+ORAL
        doses = dosing(di, dataset, 1)
        cb = CompartmentalSystemBuilder()
        central = Compartment.create(
            'CENTRAL',
            doses=find_dose(doses, comp_number=1),
            lag_time=_get_alag(control_stream, 1),
            bioavailability=_get_bioavailability(control_stream, 1),
        )
        cb.add_compartment(central)
        cb.add_flow(central, output, _advan1and2_trans(trans))
        comp_map = {'CENTRAL': 1, 'OUTPUT': 2}
        ass = _f_link_assignment(control_stream, di, dataset, comp_map, central, 1)
    elif advan == 'ADVAN2':
        doses = dosing(di, dataset, 1)
        cb = CompartmentalSystemBuilder()
        depot = Compartment.create(
            'DEPOT',
            doses=find_dose(doses, comp_number=1),
            lag_time=_get_alag(control_stream, 1),
            bioavailability=_get_bioavailability(control_stream, 1),
        )
        central = Compartment.create(
            'CENTRAL',
            doses=find_dose(doses, comp_number=2),
            lag_time=_get_alag(control_stream, 2),
            bioavailability=_get_bioavailability(control_stream, 2),
        )
        cb.add_compartment(depot)
        cb.add_compartment(central)
        cb.add_flow(central, output, _advan1and2_trans(trans))
        cb.add_flow(depot, central, Expr.symbol('KA'))
        comp_map = {'DEPOT': 1, 'CENTRAL': 2, 'OUTPUT': 3}
        ass = _f_link_assignment(control_stream, di, dataset, comp_map, central, 2)
    elif advan == 'ADVAN3':
        # FIXME: Multiple doses per compartment IV+ORAL
        doses = dosing(di, dataset, 1)
        cb = CompartmentalSystemBuilder()
        central = Compartment.create(
            'CENTRAL',
            doses=find_dose(doses, comp_number=1),
            lag_time=_get_alag(control_stream, 1),
            bioavailability=_get_bioavailability(control_stream, 1),
        )
        peripheral = Compartment.create(
            'PERIPHERAL',
            doses=tuple(),
            lag_time=_get_alag(control_stream, 2),
            bioavailability=_get_bioavailability(control_stream, 2),
        )
        cb.add_compartment(central)
        cb.add_compartment(peripheral)
        k, k12, k21 = _advan3_trans(trans)
        cb.add_flow(central, output, k)
        cb.add_flow(central, peripheral, k12)
        cb.add_flow(peripheral, central, k21)
        comp_map = {'CENTRAL': 1, 'PERIPHERAL': 2, 'OUTPUT': 3}
        ass = _f_link_assignment(control_stream, di, dataset, comp_map, central, 1)
    elif advan == 'ADVAN4':
        doses = dosing(di, dataset, 1)
        cb = CompartmentalSystemBuilder()
        depot = Compartment.create(
            'DEPOT',
            doses=find_dose(doses, comp_number=1),
            lag_time=_get_alag(control_stream, 1),
            bioavailability=_get_bioavailability(control_stream, 1),
        )
        central = Compartment.create(
            'CENTRAL',
            doses=find_dose(doses, comp_number=2),
            lag_time=_get_alag(control_stream, 2),
            bioavailability=_get_bioavailability(control_stream, 2),
        )
        peripheral = Compartment.create(
            'PERIPHERAL',
            doses=tuple(),
            lag_time=_get_alag(control_stream, 3),
            bioavailability=_get_bioavailability(control_stream, 3),
        )
        cb.add_compartment(depot)
        cb.add_compartment(central)
        cb.add_compartment(peripheral)
        k, k23, k32, ka = _advan4_trans(trans)
        cb.add_flow(depot, central, ka)
        cb.add_flow(central, output, k)
        cb.add_flow(central, peripheral, k23)
        cb.add_flow(peripheral, central, k32)
        comp_map = {'DEPOT': 1, 'CENTRAL': 2, 'PERIPHERAL': 3, 'OUTPUT': 4}
        ass = _f_link_assignment(control_stream, di, dataset, comp_map, central, 2)
    elif advan == 'ADVAN5' or advan == 'ADVAN7':
        cb = CompartmentalSystemBuilder()

        defobs, defdose, comp_map = parse_model_record(control_stream)

        doses = dosing(di, dataset, defdose[1])
        obscomp = None
        compartments = []
        for i, name in enumerate(comp_map.keys(), start=1):
            curdose = find_dose(doses, i)
            comp = Compartment.create(
                name,
                doses=curdose,
                lag_time=_get_alag(control_stream, i),
                bioavailability=_get_bioavailability(control_stream, i),
            )
            cb.add_compartment(comp)
            compartments.append(comp)
            if name == defobs[0]:
                obscomp = comp
        compartments.append(output)

        assert obscomp is not None
        for from_n, to_n, rate in _find_rates(control_stream, len(compartments)):
            cb.add_flow(compartments[from_n - 1], compartments[to_n - 1], rate)
        ass = _f_link_assignment(control_stream, di, dataset, comp_map, obscomp, defobs[1])
    elif advan == 'ADVAN10':
        # FIXME: Multiple doses per compartment needed
        doses = dosing(di, dataset, 1)
        cb = CompartmentalSystemBuilder()
        central = Compartment.create(
            'CENTRAL',
            doses=find_dose(doses, comp_number=1),
            lag_time=_get_alag(control_stream, 1),
            bioavailability=_get_bioavailability(control_stream, 1),
        )
        cb.add_compartment(central)
        vm = Expr.symbol('VM')
        km = Expr.symbol('KM')
        cb.add_flow(central, output, vm / (km + Expr.function(central.amount.name, 't')))
        comp_map = {'CENTRAL': 1, 'OUTPUT': 2}
        ass = _f_link_assignment(control_stream, di, dataset, comp_map, central, 1)
    elif advan == 'ADVAN11':
        doses = dosing(di, dataset, 1)
        cb = CompartmentalSystemBuilder()
        central = Compartment.create(
            'CENTRAL',
            doses=find_dose(doses, comp_number=1),
            lag_time=_get_alag(control_stream, 1),
            bioavailability=_get_bioavailability(control_stream, 1),
        )
        per1 = Compartment.create(
            'PERIPHERAL1',
            doses=tuple(),
            lag_time=_get_alag(control_stream, 2),
            bioavailability=_get_bioavailability(control_stream, 2),
        )
        per2 = Compartment.create(
            'PERIPHERAL2',
            doses=tuple(),
            lag_time=_get_alag(control_stream, 3),
            bioavailability=_get_bioavailability(control_stream, 3),
        )
        cb.add_compartment(central)
        cb.add_compartment(per1)
        cb.add_compartment(per2)
        k, k12, k21, k13, k31 = _advan11_trans(trans)
        cb.add_flow(central, output, k)
        cb.add_flow(central, per1, k12)
        cb.add_flow(per1, central, k21)
        cb.add_flow(central, per2, k13)
        cb.add_flow(per2, central, k31)
        comp_map = {'CENTRAL': 1, 'PERIPHERAL1': 2, 'PERIPHERAL2': 3, 'OUTPUT': 4}
        ass = _f_link_assignment(control_stream, di, dataset, comp_map, central, 1)
    elif advan == 'ADVAN12':
        doses = dosing(di, dataset, 1)
        cb = CompartmentalSystemBuilder()
        depot = Compartment.create(
            'DEPOT',
            doses=find_dose(doses, comp_number=1),
            lag_time=_get_alag(control_stream, 1),
            bioavailability=_get_bioavailability(control_stream, 1),
        )
        central = Compartment.create(
            'CENTRAL',
            doses=find_dose(doses, comp_number=2),
            lag_time=_get_alag(control_stream, 2),
            bioavailability=_get_bioavailability(control_stream, 2),
        )
        per1 = Compartment.create(
            'PERIPHERAL1',
            doses=tuple(),
            lag_time=_get_alag(control_stream, 3),
            bioavailability=_get_bioavailability(control_stream, 3),
        )
        per2 = Compartment.create(
            'PERIPHERAL2',
            doses=tuple(),
            lag_time=_get_alag(control_stream, 4),
            bioavailability=_get_bioavailability(control_stream, 4),
        )
        cb.add_compartment(depot)
        cb.add_compartment(central)
        cb.add_compartment(per1)
        cb.add_compartment(per2)
        k, k23, k32, k24, k42, ka = _advan12_trans(trans)
        cb.add_flow(depot, central, ka)
        cb.add_flow(central, output, k)
        cb.add_flow(central, per1, k23)
        cb.add_flow(per1, central, k32)
        cb.add_flow(central, per2, k24)
        cb.add_flow(per2, central, k42)
        comp_map = {'DEPOT': 1, 'CENTRAL': 2, 'PERIPHERAL1': 3, 'PERIPHERAL2': 4, 'OUTPUT': 5}
        ass = _f_link_assignment(control_stream, di, dataset, comp_map, central, 2)
    elif des:
        # FIXME: Add dose based on presence of CMT column

        defobs, defdose, comp_map = parse_model_record(control_stream)

        subs_dict = {}
        func_to_name = {}

        t = Expr.symbol('t')
        subs_dict['T'] = t
        for name, i in comp_map.items():
            a = Expr.function(f'A_{name}', t)
            subs_dict[Expr.symbol(f'DADT({i})')] = Expr.derivative(a, t)
            subs_dict[Expr.symbol(f'DADT ({i})')] = Expr.derivative(a, t)
            subs_dict[Expr.symbol(f'A({i})')] = a
            func_to_name[a] = name

        sset = des.statements.subs(subs_dict)
        eqs = [sympy.Eq(s.symbol, s.expression) for s in sset if s.symbol.is_derivative()]

        cs = to_compartmental_system(func_to_name, eqs)
        cb = CompartmentalSystemBuilder(cs)
        doses = dosing(di, dataset, defdose[1])
        for name, i in comp_map.items():
            comp = cs.find_compartment(name)
            if comp is None:  # Compartments can be in $MODEL but not used in $DES
                continue
            cb.set_dose(comp, find_dose(doses, i))
            comp = cb.find_compartment(name)
            f = _get_bioavailability(control_stream, i)
            cb.set_bioavailability(comp, f)
            comp = cb.find_compartment(name)
            alag = _get_alag(control_stream, i)
            cb.set_lag_time(comp, alag)

        ass = _f_link_assignment(
            control_stream, di, dataset, comp_map, Expr.symbol(f'A_{defobs[0]}'), defobs[1]
        )
    else:
        return None
    return CompartmentalSystem(cb), ass, comp_map


def des_assign_statements(
    control_stream: NMTranControlStream,
    des=None,
):
    if des:
        rec_model = control_stream.get_records('MODEL')[0]

        subs_dict = {}
        comps = [c for c, _ in rec_model.compartments()]
        func_to_name = {}
        t = Expr.symbol('t')
        for i, c in enumerate(comps, 1):
            a = Expr.function(f'A_{c}', t)
            subs_dict[Expr.symbol(f'DADT({i})')] = Expr.derivative(a, t)
            subs_dict[Expr.symbol(f'DADT ({i})')] = Expr.derivative(a, t)
            subs_dict[Expr.symbol(f'A({i})')] = a
            func_to_name[a] = c

        sset = des.statements.subs(subs_dict)

        statements = [
            sympy.Eq(s.symbol, s.expression)
            for s in sset
            if s.symbol.is_symbol() and not s.symbol.is_derivative()
        ]
        if len(statements) == 0:
            statements = None
        return statements


def _f_link_assignment(
    control_stream: NMTranControlStream,
    di: DataInfo,
    dataset,
    comp_map,
    compartment: Compartment,
    compno: int,
):
    f = Expr.symbol('F')
    try:
        fexpr = compartment.amount
    except AttributeError:
        fexpr = compartment
    ffunc = Expr.function(fexpr.name, 't')
    pkrec = control_stream.get_records('PK')[0]
    scaling = f'S{compno}'
    if pkrec.statements.find_assignment(scaling):
        fexpr = ffunc / Expr.symbol(scaling)
    else:
        fexpr = ffunc

    if dataset is not None and 'CMT' in dataset and not di['CMT'].drop:
        obscol = _find_observation_column(di)
        df = dataset[dataset[obscol] == 0.0]
        cmtvals = set(df['CMT'].unique())
        if 100.0 in cmtvals or 1000.0 in cmtvals:
            # These are synonymous to output
            cmtvals.add(comp_map['OUTPUT'])
            cmtvals.remove(100.0)
            cmtvals.remove(1000.0)
        if float(compno) in cmtvals:
            # Set default output to be 0.0
            cmtvals.add(0.0)
            cmtvals.remove(float(compno))
        if cmtvals != {0.0}:
            inv_map = {v: k for k, v in comp_map.items()}
            pairs = []
            for val in sorted(cmtvals - {0.0}):
                cond = BooleanExpr.eq(Expr.symbol(obscol), 0.0) & BooleanExpr.eq(
                    Expr.symbol('CMT'), val
                )
                func = Expr.function(f'A_{inv_map[val]}', 't')
                s = f'S{int(val)}'
                if pkrec.statements.find_assignment(s):
                    expr = func / s
                else:
                    expr = func
                pair = (expr, cond)
                pairs.append(pair)
            if 0.0 in cmtvals:
                pair = (fexpr, True)
                pairs.append(pair)
            fexpr = Expr.piecewise(*pairs)
    ass = Assignment(f, fexpr)
    return ass


def _find_observation_column(di):
    colnames = di.names
    if 'EVID' in colnames and not di['EVID'].drop:
        return 'EVID'
    elif 'MDV' in colnames and not di['MDV'].drop:
        return 'MDV'
    else:
        return 'AMT'


def _find_rates(control_stream: NMTranControlStream, ncomps: int):
    pkrec = control_stream.get_records('PK')[0]
    for stat in pkrec.statements:
        if hasattr(stat, 'symbol'):
            name = stat.symbol.name
            m = re.match(r'^K(\d+)(T\d+)?$', name)
            if m:
                if m.group(2):
                    from_n = int(m.group(1))
                    to_n = int(m.group(2)[1:])
                else:
                    n = m.group(1)
                    if len(n) == 2:
                        from_n = int(n[0])
                        to_n = int(n[1])
                    elif len(n) == 3:
                        f1 = int(n[0])
                        t1 = int(n[1:])
                        f2 = int(n[:2])
                        t2 = int(n[2:])
                        q1 = f1 <= ncomps and t1 <= ncomps and t1 != 0
                        q2 = f2 <= ncomps and t2 <= ncomps
                        if q1 and q2:
                            raise ModelSyntaxError(
                                f'Rate parameter {n} is ambiguous. Use the KiTj notation.'
                            )
                        if q1:
                            from_n = f1
                            to_n = t1
                        elif q2:
                            from_n = f2
                            to_n = t2
                        else:
                            # Too large to or from compartment index. What would NONMEM do?
                            # Could also be too large later
                            continue
                    elif len(n) == 4:
                        from_n = int(n[:2])
                        to_n = int(n[2:])
                    else:
                        raise ValueError(f'Cannot handle {n}.')

                if to_n == 0:
                    to_n = ncomps

                yield from_n, to_n, Expr.symbol(name)


def _advan1and2_trans(trans: str):
    if trans == 'TRANS2':
        return Expr.symbol('CL') / Expr.symbol('V')
    else:  # TRANS1 which is also the default
        return Expr.symbol('K')


def _advan3_trans(trans: str):
    if trans == 'TRANS3':
        return (
            Expr.symbol('CL') / Expr.symbol('V'),
            Expr.symbol('Q') / Expr.symbol('V'),
            Expr.symbol('Q') / (Expr.symbol('VSS') - Expr.symbol('V')),
        )
    elif trans == 'TRANS4':
        return (
            Expr.symbol('CL') / Expr.symbol('V1'),
            Expr.symbol('Q') / Expr.symbol('V1'),
            Expr.symbol('Q') / Expr.symbol('V2'),
        )
    elif trans == 'TRANS5':
        return (
            Expr.symbol('ALPHA') * Expr.symbol('BETA') / Expr.symbol('K21'),
            Expr.symbol('ALPHA') + Expr.symbol('BETA') - Expr.symbol('K21') - Expr.symbol('K'),
            (Expr.symbol('AOB') * Expr.symbol('BETA') + Expr.symbol('ALPHA'))
            / (Expr.symbol('AOB') + 1),
        )
    elif trans == 'TRANS6':
        return (
            Expr.symbol('ALPHA') * Expr.symbol('BETA') / Expr.symbol('K21'),
            Expr.symbol('ALPHA') + Expr.symbol('BETA') - Expr.symbol('K21') - Expr.symbol('K'),
            Expr.symbol('K21'),
        )
    else:
        return (Expr.symbol('K'), Expr.symbol('K12'), Expr.symbol('K21'))


def _advan4_trans(trans: str):
    if trans == 'TRANS3':
        return (
            Expr.symbol('CL') / Expr.symbol('V'),
            Expr.symbol('Q') / Expr.symbol('V'),
            Expr.symbol('Q') / (Expr.symbol('VSS') - Expr.symbol('V')),
            Expr.symbol('KA'),
        )
    elif trans == 'TRANS4':
        return (
            Expr.symbol('CL') / Expr.symbol('V2'),
            Expr.symbol('Q') / Expr.symbol('V2'),
            Expr.symbol('Q') / Expr.symbol('V3'),
            Expr.symbol('KA'),
        )
    elif trans == 'TRANS5':
        return (
            Expr.symbol('ALPHA') * Expr.symbol('BETA') / Expr.symbol('K32'),
            Expr.symbol('ALPHA') + Expr.symbol('BETA') - Expr.symbol('K32') - Expr.symbol('K'),
            (Expr.symbol('AOB') * Expr.symbol('BETA') + Expr.symbol('ALPHA'))
            / (Expr.symbol('AOB') + 1),
            Expr.symbol('KA'),
        )
    elif trans == 'TRANS6':
        return (
            Expr.symbol('ALPHA') * Expr.symbol('BETA') / Expr.symbol('K32'),
            Expr.symbol('ALPHA') + Expr.symbol('BETA') - Expr.symbol('K32') - Expr.symbol('K'),
            Expr.symbol('K32'),
            Expr.symbol('KA'),
        )
    else:
        return (Expr.symbol('K'), Expr.symbol('K23'), Expr.symbol('K32'), Expr.symbol('KA'))


def _advan11_trans(trans: str):
    if trans == 'TRANS4':
        return (
            Expr.symbol('CL') / Expr.symbol('V1'),
            Expr.symbol('Q2') / Expr.symbol('V1'),
            Expr.symbol('Q2') / Expr.symbol('V2'),
            Expr.symbol('Q3') / Expr.symbol('V1'),
            Expr.symbol('Q3') / Expr.symbol('V3'),
        )
    elif trans == 'TRANS6':
        return (
            Expr.symbol('ALPHA')
            * Expr.symbol('BETA')
            * Expr.symbol('GAMMA')
            / (Expr.symbol('K21') * Expr.symbol('K31')),
            Expr.symbol('ALPHA')
            + Expr.symbol('BETA')
            + Expr.symbol('GAMMA')
            - Expr.symbol('K')
            - Expr.symbol('K13')
            - Expr.symbol('K21')
            - Expr.symbol('K31'),
            Expr.symbol('K21'),
            (
                Expr.symbol('ALPHA') * Expr.symbol('BETA')
                + Expr.symbol('ALPHA') * Expr.symbol('GAMMA')
                + Expr.symbol('BETA') * Expr.symbol('GAMMA')
                + Expr.symbol('K31') * Expr.symbol('K31')
                - Expr.symbol('K31')
                * (Expr.symbol('ALPHA') + Expr.symbol('BETA') + Expr.symbol('GAMMA'))
                - Expr.symbol('K') * Expr.symbol('K21')
            )
            / (Expr.symbol('K21') - Expr.symbol('K31')),
            Expr.symbol('K31'),
        )
    else:
        return (
            Expr.symbol('K'),
            Expr.symbol('K12'),
            Expr.symbol('K21'),
            Expr.symbol('K13'),
            Expr.symbol('K31'),
        )


def _advan12_trans(trans: str):
    if trans == 'TRANS4':
        return (
            Expr.symbol('CL') / Expr.symbol('V2'),
            Expr.symbol('Q3') / Expr.symbol('V2'),
            Expr.symbol('Q3') / Expr.symbol('V3'),
            Expr.symbol('Q4') / Expr.symbol('V2'),
            Expr.symbol('Q4') / Expr.symbol('V4'),
            Expr.symbol('KA'),
        )
    elif trans == 'TRANS6':
        return (
            Expr.symbol('ALPHA')
            * Expr.symbol('BETA')
            * Expr.symbol('GAMMA')
            / (Expr.symbol('K32') * Expr.symbol('K42')),
            Expr.symbol('ALPHA')
            + Expr.symbol('BETA')
            + Expr.symbol('GAMMA')
            - Expr.symbol('K')
            - Expr.symbol('K24')
            - Expr.symbol('K32')
            - Expr.symbol('K42'),
            Expr.symbol('K32'),
            (
                Expr.symbol('ALPHA') * Expr.symbol('BETA')
                + Expr.symbol('ALPHA') * Expr.symbol('GAMMA')
                + Expr.symbol('BETA') * Expr.symbol('GAMMA')
                + Expr.symbol('K42') * Expr.symbol('K42')
                - Expr.symbol('K42')
                * (Expr.symbol('ALPHA') + Expr.symbol('BETA') + Expr.symbol('GAMMA'))
                - Expr.symbol('K') * Expr.symbol('K32')
            )
            / (Expr.symbol('K32') - Expr.symbol('K42')),
            Expr.symbol('K42'),
            Expr.symbol('KA'),
        )
    else:
        return (
            Expr.symbol('K'),
            Expr.symbol('K23'),
            Expr.symbol('K32'),
            Expr.symbol('K24'),
            Expr.symbol('K42'),
            Expr.symbol('KA'),
        )


def dosing(di: DataInfo, dataset, dose_comp: int):
    # Only check doses
    if dataset is not None:
        dataset = dataset[dataset['AMT'] != 0]

    cmt_loop = False
    admid_name = None
    return_dose = False
    if dataset is None:
        return_dose = True
    elif 'admid' in di.types:
        admid_name = di.typeix["admid"][0].name
        if 'CMT' in di.names and not di['CMT'].drop:
            if len(dataset['CMT']) == 1:
                warnings.warn("CMT column present with only one value")
            cmt_loop = True
    elif 'CMT' in di.names and not di['CMT'].drop:
        if len(dataset['CMT']) == 1:
            warnings.warn("CMT column present with only one value")
            return_dose = True
        else:
            cmt_loop = True
    else:
        return_dose = True

    if return_dose:
        return (
            {
                'comp_number': dose_comp,
                'dose': _dosing(di, dataset, dose_comp),
                'admid': None,
            },
        )

    doses = tuple()
    if admid_name is not None and cmt_loop:
        for comp_number in dataset['CMT'].unique():
            cmt_dataset = dataset[dataset['CMT'] == comp_number]
            for admid in cmt_dataset[admid_name].unique():
                doses += (
                    {
                        'comp_number': comp_number,
                        'dose': _dosing(di, dataset.loc[dataset[admid_name] == admid], comp_number),
                        'admid': admid,
                    },
                )
    elif admid_name is not None:
        for admid in dataset[admid_name].unique():
            doses += (
                {
                    'comp_number': dose_comp,
                    'dose': _dosing(di, dataset.loc[dataset[admid_name] == admid], dose_comp),
                    'admid': admid,
                },
            )
    elif cmt_loop:
        for comp_number in dataset['CMT'].unique():
            cmt_dataset = dataset[dataset['CMT'] == comp_number]
            doses += (
                {
                    'comp_number': comp_number,
                    'dose': _dosing(di, cmt_dataset, comp_number),
                    'admid': None,
                },
            )

    return doses


def _dosing(di, dataset, dose_comp):
    amt = Expr.symbol('AMT')
    if 'RATE' not in di.names or di['RATE'].drop:
        return Bolus(amt)

    df = dataset

    if df is None:
        return Bolus(amt)
    elif (df['RATE'] == 0).all():
        return Bolus(amt)
    elif (df['RATE'] == -1).any():
        return Infusion(amt, rate=Expr.symbol(f'R{dose_comp}'))
    elif (df['RATE'] == -2).any():
        return Infusion(amt, duration=Expr.symbol(f'D{dose_comp}'))
    else:
        return Infusion(amt, rate=Expr.symbol('RATE'))


def find_dose(doses, comp_number, admid=1):
    comp_doses = tuple()
    for dose in doses:
        if dose['comp_number'] == comp_number:
            comp_dose = dose['dose']
            if dose['admid'] is not None:
                admid = dose['admid']
            comp_doses += (comp_dose.replace(admid=admid),)
    return comp_doses


def _get_alag(control_stream: NMTranControlStream, n: int):
    """Check if ALAGn is defined in model and return it else return 0"""
    alag = f'ALAG{n}'
    pkrec = control_stream.get_records('PK')[0]
    if pkrec.statements.find_assignment(alag):
        return Expr.symbol(alag)
    else:
        return Expr.integer(0)


def _get_bioavailability(control_stream: NMTranControlStream, n: int):
    """Check if Fn is defined in model and return it else return 0"""
    fn = f'F{n}'
    pkrec = control_stream.get_records('PK')[0]
    if pkrec.statements.find_assignment(fn):
        return Expr.symbol(fn)
    else:
        return Expr.integer(1)


def parse_model_record(control_stream):
    # Return DEFOBS, DEFDOSE and map from compartment name to number
    modrec = control_stream.get_records('MODEL')[0]
    defobs: Optional[Tuple[str, int]] = None
    defdose: Optional[Tuple[str, int]] = None
    defcentral: Optional[Tuple[str, int]] = None
    defdepot: Optional[Tuple[str, int]] = None
    deffirst_dose: Optional[Tuple[str, int]] = None
    comp_names = []
    for i, (name, opts) in enumerate(modrec.compartments(), 1):
        if 'DEFOBSERVATION' in opts:
            defobs = (name, i)
        if 'DEFDOSE' in opts:
            defdose = (name, i)
        if name == 'CENTRAL':
            defcentral = (name, i)
        elif name == 'DEPOT':
            defdepot = (name, i)
        if deffirst_dose is None and 'NODOSE' not in opts:
            deffirst_dose = (name, i)
        comp_names.append(name)

    comp_map = {name: i for i, name in enumerate(comp_names, 1)}

    if defobs is None:
        if defcentral is None:
            defobs = (comp_names[0], 1)
        else:
            defobs = defcentral

    if defdose is None:
        if defdepot is not None:
            defdose = defdepot
        elif deffirst_dose is not None:
            defdose = deffirst_dose
        else:
            raise ModelSyntaxError('Dosing compartment is unknown')

    return defobs, defdose, comp_map
