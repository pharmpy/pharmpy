import re

import sympy
from sympy import Derivative, Eq, Function, Piecewise
from sympy import Symbol as symbol

from pharmpy.model import ModelSyntaxError
from pharmpy.statements import (
    Assignment,
    Bolus,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    ExplicitODESystem,
    Infusion,
)


def compartmental_model(model, advan, trans, des=None):
    if advan == 'ADVAN1':
        dose = _dosing(model, 1)
        cb = CompartmentalSystemBuilder()
        central = Compartment('CENTRAL', dose, get_alag(model, 1), get_bioavailability(model, 1))
        output = Compartment('OUTPUT')
        cb.add_compartment(central)
        cb.add_compartment(output)
        cb.add_flow(central, output, _advan1and2_trans(trans))
        ass = _f_link_assignment(model, central)
        comp_map = {'CENTRAL': 1, 'OUTPUT': 2}
    elif advan == 'ADVAN2':
        dose = _dosing(model, 1)
        cb = CompartmentalSystemBuilder()
        depot = Compartment('DEPOT', dose, get_alag(model, 1), get_bioavailability(model, 1))
        central = Compartment('CENTRAL', None, get_alag(model, 2), get_bioavailability(model, 2))
        output = Compartment('OUTPUT')
        cb.add_compartment(depot)
        cb.add_compartment(central)
        cb.add_compartment(output)
        cb.add_flow(central, output, _advan1and2_trans(trans))
        cb.add_flow(depot, central, symbol('KA'))
        ass = _f_link_assignment(model, central)
        comp_map = {'DEPOT': 1, 'CENTRAL': 2, 'OUTPUT': 3}
    elif advan == 'ADVAN3':
        dose = _dosing(model, 1)
        cb = CompartmentalSystemBuilder()
        central = Compartment('CENTRAL', dose, get_alag(model, 1), get_bioavailability(model, 1))
        peripheral = Compartment(
            'PERIPHERAL', None, get_alag(model, 2), get_bioavailability(model, 2)
        )
        output = Compartment('OUTPUT')
        cb.add_compartment(central)
        cb.add_compartment(peripheral)
        cb.add_compartment(output)
        k, k12, k21 = _advan3_trans(trans)
        cb.add_flow(central, output, k)
        cb.add_flow(central, peripheral, k12)
        cb.add_flow(peripheral, central, k21)
        ass = _f_link_assignment(model, central)
        comp_map = {'CENTRAL': 1, 'PERIPHERAL': 2, 'OUTPUT': 3}
    elif advan == 'ADVAN4':
        dose = _dosing(model, 1)
        cb = CompartmentalSystemBuilder()
        depot = Compartment('DEPOT', dose, get_alag(model, 1), get_bioavailability(model, 1))
        central = Compartment('CENTRAL', None, get_alag(model, 2), get_bioavailability(model, 2))
        peripheral = Compartment(
            'PERIPHERAL', None, get_alag(model, 3), get_bioavailability(model, 3)
        )
        output = Compartment('OUTPUT')
        cb.add_compartment(depot)
        cb.add_compartment(central)
        cb.add_compartment(peripheral)
        cb.add_compartment(output)
        k, k23, k32, ka = _advan4_trans(trans)
        cb.add_flow(depot, central, ka)
        cb.add_flow(central, output, k)
        cb.add_flow(central, peripheral, k23)
        cb.add_flow(peripheral, central, k32)
        ass = _f_link_assignment(model, central)
        comp_map = {'DEPOT': 1, 'CENTRAL': 2, 'PERIPHERAL': 3, 'OUTPUT': 4}
    elif advan == 'ADVAN5' or advan == 'ADVAN7':
        cb = CompartmentalSystemBuilder()
        modrec = model.control_stream.get_records('MODEL')[0]
        defobs = None
        defdose = None
        central = None
        depot = None
        first_dose = None
        compartments = []
        comp_names = []
        for i, (name, opts) in enumerate(modrec.compartments()):
            if 'DEFOBSERVATION' in opts:
                defobs = name
            if 'DEFDOSE' in opts:
                defdose = name
                dose_no = i + 1
            if name == 'CENTRAL':
                central = name
            elif name == 'DEPOT':
                depot = name
                depot_no = i + 1
            if first_dose is None and 'NODOSE' not in opts:
                first_dose = name
                first_dose_no = i + 1
            comp_names.append(name)
        comp_names.append('OUTPUT')
        comp_map = {name: i + 1 for i, name in enumerate(comp_names)}
        ncomp = i + 2
        if not defobs:
            if central:
                defobs = central
            else:
                defobs = comp_names[0]
        if not defdose:
            if depot:
                defdose = depot
                dose_no = depot_no
            elif first_dose is not None:
                defdose = first_dose
                dose_no = first_dose_no
            else:
                raise ModelSyntaxError('Dosing compartment is unknown')
        dose = _dosing(model, dose_no)
        for i, name in enumerate(comp_names):
            if i == len(comp_names) - 1:
                output = Compartment(name)
                cb.add_compartment(output)
                compartments.append(output)
                break
            if name == defdose:
                curdose = dose
            else:
                curdose = None
            comp = Compartment(name, curdose, get_alag(model, i), get_bioavailability(model, i))
            cb.add_compartment(comp)
            compartments.append(comp)
            if name == defobs:
                defobs = comp
        for from_n, to_n, rate in _find_rates(model, ncomp):
            cb.add_flow(compartments[from_n - 1], compartments[to_n - 1], rate)
        ass = _f_link_assignment(model, defobs)
    elif advan == 'ADVAN10':
        dose = _dosing(model, 1)
        cb = CompartmentalSystemBuilder()
        central = Compartment('CENTRAL', dose, get_alag(model, 1), get_bioavailability(model, 1))
        output = Compartment('OUTPUT')
        cb.add_compartment(central)
        cb.add_compartment(output)
        vm = symbol('VM')
        km = symbol('KM')
        t = symbol('t')
        cb.add_flow(central, output, vm / (km + sympy.Function(central.amount.name)(t)))
        ass = _f_link_assignment(model, central)
        comp_map = {'CENTRAL': 1, 'OUTPUT': 2}
    elif advan == 'ADVAN11':
        dose = _dosing(model, 1)
        cb = CompartmentalSystemBuilder()
        central = Compartment('CENTRAL', dose, get_alag(model, 1), get_bioavailability(model, 1))
        per1 = Compartment('PERIPHERAL1', None, get_alag(model, 2), get_bioavailability(model, 2))
        per2 = Compartment('PERIPHERAL2', None, get_alag(model, 3), get_bioavailability(model, 3))
        output = Compartment('OUTPUT')
        cb.add_compartment(central)
        cb.add_compartment(per1)
        cb.add_compartment(per2)
        cb.add_compartment(output)
        k, k12, k21, k13, k31 = _advan11_trans(trans)
        cb.add_flow(central, output, k)
        cb.add_flow(central, per1, k12)
        cb.add_flow(per1, central, k21)
        cb.add_flow(central, per2, k13)
        cb.add_flow(per2, central, k31)
        ass = _f_link_assignment(model, central)
        comp_map = {'CENTRAL': 1, 'PERIPHERAL1': 2, 'PERIPHERAL2': 3, 'OUTPUT': 4}
    elif advan == 'ADVAN12':
        dose = _dosing(model, 1)
        cb = CompartmentalSystemBuilder()
        depot = Compartment('DEPOT', dose, get_alag(model, 1), get_bioavailability(model, 1))
        central = Compartment('CENTRAL', None, get_alag(model, 2), get_bioavailability(model, 2))
        per1 = Compartment('PERIPHERAL1', None, get_alag(model, 3), get_bioavailability(model, 3))
        per2 = Compartment('PERIPHERAL2', None, get_alag(model, 4), get_bioavailability(model, 4))
        output = Compartment('OUTPUT')
        cb.add_compartment(depot)
        cb.add_compartment(central)
        cb.add_compartment(per1)
        cb.add_compartment(per2)
        cb.add_compartment(output)
        k, k23, k32, k24, k42, ka = _advan12_trans(trans)
        cb.add_flow(depot, central, ka)
        cb.add_flow(central, output, k)
        cb.add_flow(central, per1, k23)
        cb.add_flow(per1, central, k32)
        cb.add_flow(central, per2, k24)
        cb.add_flow(per2, central, k42)
        ass = _f_link_assignment(model, central)
        comp_map = {'DEPOT': 1, 'CENTRAL': 2, 'PERIPHERAL1': 3, 'PERIPHERAL2': 4, 'OUTPUT': 5}
    elif des:
        rec_model = model.control_stream.get_records('MODEL')[0]

        subs_dict, comp_names = dict(), dict()
        comps = [c for c, _ in rec_model.compartments()]

        t = symbol('t')
        for i, c in enumerate(comps, 1):
            a = Function(f'A_{c}')
            subs_dict[f'DADT({i})'] = Derivative(a(t))
            subs_dict[f'A({i})'] = a(t)
            comp_names[f'A({i})'] = a

        sset = des.statements.subs(subs_dict)

        a_out = Function('A_OUTPUT')
        dose = _dosing(model, 1)

        ics = {v(0): sympy.Integer(0) for v in comp_names.values()}
        ics[a_out(0)] = sympy.Integer(0)
        ics[comp_names['A(1)'](0)] = dose.amount

        dadt_dose = sset.find_assignment(str(subs_dict['DADT(1)']))

        if len(comps) > 1:
            dadt_rest = [
                Eq(s.symbol, s.expression)
                for s in sset
                if s != dadt_dose and not s.symbol.is_Symbol
            ]
            lhs_sum = dadt_dose.expression
            for eq in dadt_rest:
                lhs_sum += eq.rhs
            dadt_out = Eq(Derivative(a_out(t)), -lhs_sum)
            dadt_rest.append(dadt_out)
        else:
            dadt_rest = [Eq(Derivative(a_out(t)), dadt_dose.expression * -1)]

        if isinstance(dose, Infusion):
            if dose.duration:
                rate = dose.amount / dose.duration
                duration = dose.duration
            else:
                rate = dose.rate
                duration = dose.amount / dose.rate

            dadt_dose.expression += Piecewise((rate, duration > t), (0, True))
            ics[comp_names['A(1)'](0)] = sympy.Integer(0)

        eqs = [Eq(dadt_dose.symbol, dadt_dose.expression)] + dadt_rest

        ode = ExplicitODESystem(eqs, ics)
        ass = _f_link_assignment(model, symbol('A_CENTRAL'))
        return ode, ass
    else:
        return None
    model._compartment_map = comp_map
    return CompartmentalSystem(cb), ass


def _f_link_assignment(model, compartment):
    f = symbol('F')
    try:
        fexpr = compartment.amount
    except AttributeError:
        fexpr = compartment
    pkrec = model.control_stream.get_records('PK')[0]
    if pkrec.statements.find_assignment('S1'):
        fexpr = fexpr / symbol('S1')
    ass = Assignment(f, fexpr)
    return ass


def _find_rates(model, ncomps):
    pkrec = model.control_stream.get_records('PK')[0]
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
                        f2 = int(n[0:2])
                        t2 = int(n[2:])
                        q1 = f1 <= ncomps and t1 <= ncomps and t1 != 0
                        q2 = f2 <= ncomps and t2 <= ncomps
                        if q1 and q2:
                            raise ModelSyntaxError(
                                f'Rate parameter {n} is ambiguous. ' f'Use the KiTj notation.'
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
                        from_n = int(n[0:2])
                        to_n = int(n[2:])
                if to_n == 0:
                    to_n = ncomps
                yield from_n, to_n, symbol(name)


def _advan1and2_trans(trans):
    if trans == 'TRANS2':
        return symbol('CL') / symbol('V')
    else:  # TRANS1 which is also the default
        return symbol('K')


def _advan3_trans(trans):
    if trans == 'TRANS3':
        return (
            symbol('CL') / symbol('V'),
            symbol('Q') / symbol('V'),
            symbol('Q') / (symbol('VSS') - symbol('V')),
        )
    elif trans == 'TRANS4':
        return (symbol('CL') / symbol('V1'), symbol('Q') / symbol('V1'), symbol('Q') / symbol('V2'))
    elif trans == 'TRANS5':
        return (
            symbol('ALPHA') * symbol('BETA') / symbol('K21'),
            symbol('ALPHA') + symbol('BETA') - symbol('K21') - symbol('K'),
            (symbol('AOB') * symbol('BETA') + symbol('ALPHA')) / (symbol('AOB') + 1),
        )
    elif trans == 'TRANS6':
        return (
            symbol('ALPHA') * symbol('BETA') / symbol('K21'),
            symbol('ALPHA') + symbol('BETA') - symbol('K21') - symbol('K'),
            symbol('K21'),
        )
    else:
        return (symbol('K'), symbol('K12'), symbol('K21'))


def _advan4_trans(trans):
    if trans == 'TRANS3':
        return (
            symbol('CL') / symbol('V'),
            symbol('Q') / symbol('V'),
            symbol('Q') / (symbol('VSS') - symbol('V')),
            symbol('KA'),
        )
    elif trans == 'TRANS4':
        return (
            symbol('CL') / symbol('V2'),
            symbol('Q') / symbol('V2'),
            symbol('Q') / symbol('V3'),
            symbol('KA'),
        )
    elif trans == 'TRANS5':
        return (
            symbol('ALPHA') * symbol('BETA') / symbol('K32'),
            symbol('ALPHA') + symbol('BETA') - symbol('K32') - symbol('K'),
            (symbol('AOB') * symbol('BETA') + symbol('ALPHA')) / (symbol('AOB') + 1),
            symbol('KA'),
        )
    elif trans == 'TRANS6':
        return (
            symbol('ALPHA') * symbol('BETA') / symbol('K32'),
            symbol('ALPHA') + symbol('BETA') - symbol('K32') - symbol('K'),
            symbol('K32'),
            symbol('KA'),
        )
    else:
        return (symbol('K'), symbol('K23'), symbol('K32'), symbol('KA'))


def _advan11_trans(trans):
    if trans == 'TRANS4':
        return (
            symbol('CL') / symbol('V1'),
            symbol('Q2') / symbol('V1'),
            symbol('Q2') / symbol('V2'),
            symbol('Q3') / symbol('V1'),
            symbol('Q3') / symbol('V3'),
        )
    elif trans == 'TRANS6':
        return (
            symbol('ALPHA') * symbol('BETA') * symbol('GAMMA') / (symbol('K21') * symbol('K31')),
            symbol('ALPHA')
            + symbol('BETA')
            + symbol('GAMMA')
            - symbol('K')
            - symbol('K13')
            - symbol('K21')
            - symbol('K31'),
            symbol('K21'),
            (
                symbol('ALPHA') * symbol('BETA')
                + symbol('ALPHA') * symbol('GAMMA')
                + symbol('BETA') * symbol('GAMMA')
                + symbol('K31') * symbol('K31')
                - symbol('K31') * (symbol('ALPHA') + symbol('BETA') + symbol('GAMMA'))
                - symbol('K') * symbol('K21')
            )
            / (symbol('K21') - symbol('K31')),
            symbol('K31'),
        )
    else:
        return (symbol('K'), symbol('K12'), symbol('K21'), symbol('K13'), symbol('K31'))


def _advan12_trans(trans):
    if trans == 'TRANS4':
        return (
            symbol('CL') / symbol('V2'),
            symbol('Q3') / symbol('V2'),
            symbol('Q3') / symbol('V3'),
            symbol('Q4') / symbol('V2'),
            symbol('Q4') / symbol('V4'),
            symbol('KA'),
        )
    elif trans == 'TRANS6':
        return (
            symbol('ALPHA') * symbol('BETA') * symbol('GAMMA') / (symbol('K32') * symbol('K42')),
            symbol('ALPHA')
            + symbol('BETA')
            + symbol('GAMMA')
            - symbol('K')
            - symbol('K24')
            - symbol('K32')
            - symbol('K42'),
            symbol('K32'),
            (
                symbol('ALPHA') * symbol('BETA')
                + symbol('ALPHA') * symbol('GAMMA')
                + symbol('BETA') * symbol('GAMMA')
                + symbol('K42') * symbol('K42')
                - symbol('K42') * (symbol('ALPHA') + symbol('BETA') + symbol('GAMMA'))
                - symbol('K') * symbol('K32')
            )
            / (symbol('K32') - symbol('K42')),
            symbol('K42'),
            symbol('KA'),
        )
    else:
        return (
            symbol('K'),
            symbol('K23'),
            symbol('K32'),
            symbol('K24'),
            symbol('K42'),
            symbol('KA'),
        )


def _dosing(model, dose_comp):
    return dosing(model.datainfo, lambda: model.dataset, dose_comp)


def dosing(di, dataset, dose_comp):
    colnames = di.names

    if 'RATE' in colnames and not di['RATE'].drop:
        df = dataset()
        if (df['RATE'] == 0).all():
            return Bolus(sympy.Symbol('AMT'))
        elif (df['RATE'] == -1).any():
            return Infusion(sympy.Symbol('AMT'), rate=sympy.Symbol(f'R{dose_comp}'))
        elif (df['RATE'] == -2).any():
            return Infusion(sympy.Symbol('AMT'), duration=sympy.Symbol(f'D{dose_comp}'))
        else:
            return Infusion(sympy.Symbol('AMT'), rate=sympy.Symbol('RATE'))
    else:
        return Bolus(sympy.Symbol('AMT'))


def get_alag(model, n):
    """Check if ALAGn is defined in model and return it else return 0"""
    alag = f'ALAG{n}'
    pkrec = model.control_stream.get_records('PK')[0]
    if pkrec.statements.find_assignment(alag):
        return sympy.Symbol(alag)
    else:
        return sympy.Integer(0)


def get_bioavailability(model, n):
    """Check if Fn is defined in model and return it else return 0"""
    fn = f'F{n}'
    pkrec = model.control_stream.get_records('PK')[0]
    if pkrec.statements.find_assignment(fn):
        return fn
    else:
        return sympy.Integer(1)
