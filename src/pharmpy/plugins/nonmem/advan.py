import re

import sympy

from pharmpy.model import ModelSyntaxError
from pharmpy.statements import Assignment, Bolus, CompartmentalSystem
from pharmpy.symbols import real


def compartmental_model(model, advan, trans):
    if advan == 'ADVAN1':
        cm = CompartmentalSystem()
        central = cm.add_compartment('CENTRAL')
        output = cm.add_compartment('OUTPUT')
        cm.add_flow(central, output, _advan1and2_trans(trans))
        dose = Bolus('AMT')
        central.dose = dose
        ass = _f_link_assignment(model, central)
    elif advan == 'ADVAN2':
        cm = CompartmentalSystem()
        depot = cm.add_compartment('DEPOT')
        central = cm.add_compartment('CENTRAL')
        output = cm.add_compartment('OUTPUT')
        cm.add_flow(central, output, _advan1and2_trans(trans))
        cm.add_flow(depot, central, real('KA'))
        dose = Bolus('AMT')
        depot.dose = dose
        ass = _f_link_assignment(model, central)
    elif advan == 'ADVAN3':
        cm = CompartmentalSystem()
        central = cm.add_compartment('CENTRAL')
        peripheral = cm.add_compartment('PERIPHERAL')
        output = cm.add_compartment('OUTPUT')
        k, k12, k21 = _advan3_trans(trans)
        cm.add_flow(central, output, k)
        cm.add_flow(central, peripheral, k12)
        cm.add_flow(peripheral, central, k21)
        dose = Bolus('AMT')
        central.dose = dose
        ass = _f_link_assignment(model, central)
    elif advan == 'ADVAN4':
        cm = CompartmentalSystem()
        depot = cm.add_compartment('DEPOT')
        central = cm.add_compartment('CENTRAL')
        peripheral = cm.add_compartment('PERIPHERAL')
        output = cm.add_compartment('OUTPUT')
        k, k23, k32, ka = _advan4_trans(trans)
        cm.add_flow(depot, central, ka)
        cm.add_flow(central, output, k)
        cm.add_flow(central, peripheral, k23)
        cm.add_flow(peripheral, central, k32)
        dose = Bolus('AMT')
        depot.dose = dose
        ass = _f_link_assignment(model, central)
    elif advan == 'ADVAN5' or advan == 'ADVAN7':
        cm = CompartmentalSystem()
        modrec = model.control_stream.get_records('MODEL')[0]
        defobs = None
        defdose = None
        central = None
        depot = None
        first_dose = None
        compartments = []
        for i, (name, opts) in enumerate(modrec.compartments()):
            comp = cm.add_compartment(name)
            if 'DEFOBSERVATION' in opts:
                defobs = comp
            if 'DEFDOSE' in opts:
                defdose = comp
            if name == 'CENTRAL':
                central = comp
            elif name == 'DEPOT':
                depot = comp
            if first_dose is None and 'NODOSE' not in opts:
                first_dose = comp
            compartments.append(comp)
        output = cm.add_compartment('OUTPUT')
        compartments.append(output)
        ncomp = i + 2
        if not defobs:
            if central:
                defobs = central
            else:
                defobs = compartments[0]
        if not defdose:
            if depot:
                defdose = depot
            elif first_dose is not None:
                defdose = first_dose
            else:
                raise ModelSyntaxError('Dosing compartment is unknown')
        for from_n, to_n, rate in _find_rates(model, ncomp):
            cm.add_flow(compartments[from_n - 1], compartments[to_n - 1], rate)
        dose = Bolus('AMT')
        defdose.dose = dose
        ass = _f_link_assignment(model, defobs)
    elif advan == 'ADVAN10':
        cm = CompartmentalSystem()
        central = cm.add_compartment('CENTRAL')
        output = cm.add_compartment('OUTPUT')
        vm = real('VM')
        km = real('KM')
        dose = Bolus('AMT')
        central.dose = dose
        t = real('t')
        cm.add_flow(central, output, vm / (km + sympy.Function(central.amount.name)(t)))
        ass = _f_link_assignment(model, central)
    elif advan == 'ADVAN11':
        cm = CompartmentalSystem()
        central = cm.add_compartment('CENTRAL')
        per1 = cm.add_compartment('PERIPHERAL1')
        per2 = cm.add_compartment('PERIPHERAL2')
        output = cm.add_compartment('OUTPUT')
        k, k12, k21, k13, k31 = _advan11_trans(trans)
        cm.add_flow(central, output, k)
        cm.add_flow(central, per1, k12)
        cm.add_flow(per1, central, k21)
        cm.add_flow(central, per2, k13)
        cm.add_flow(per2, central, k31)
        dose = Bolus('AMT')
        central.dose = dose
        ass = _f_link_assignment(model, central)
    elif advan == 'ADVAN12':
        cm = CompartmentalSystem()
        depot = cm.add_compartment('DEPOT')
        central = cm.add_compartment('CENTRAL')
        per1 = cm.add_compartment('PERIPHERAL1')
        per2 = cm.add_compartment('PERIPHERAL2')
        output = cm.add_compartment('OUTPUT')
        k, k23, k32, k24, k42, ka = _advan12_trans(trans)
        cm.add_flow(depot, central, ka)
        cm.add_flow(central, output, k)
        cm.add_flow(central, per1, k23)
        cm.add_flow(per1, central, k32)
        cm.add_flow(central, per2, k24)
        cm.add_flow(per2, central, k42)
        dose = Bolus('AMT')
        depot.dose = dose
        ass = _f_link_assignment(model, central)
    else:
        return None
    return cm, ass


def _f_link_assignment(model, compartment):
    f = real('F')
    fexpr = compartment.amount
    pkrec = model.control_stream.get_records('PK')[0]
    if pkrec.statements.find_assignment('S1'):
        fexpr = fexpr / real('S1')
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
                        q1 = f1 <= ncomps and t1 <= ncomps
                        q2 = f2 <= ncomps and t2 <= ncomps
                        if q1 and q2:
                            raise ModelSyntaxError(f'Rate parameter {n} is ambiguous. '
                                                   f'Use the KiTj notation.')
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
                yield from_n, to_n, real(name)


def _advan1and2_trans(trans):
    if trans == 'TRANS2':
        return real('CL') / real('V')
    else:       # TRANS1 which is also the default
        return real('K')


def _advan3_trans(trans):
    if trans == 'TRANS3':
        return (real('CL') / real('V'),
                real('Q') / real('V'),
                real('Q') / (real('VSS') - real('V')))
    elif trans == 'TRANS4':
        return (real('CL') / real('V1'),
                real('Q') / real('V1'),
                real('Q') / real('V2'))
    elif trans == 'TRANS5':
        return (real('ALPHA') * real('BETA') / real('K21'),
                real('ALPHA') + real('BETA') - real('K21') - real('K'),
                (real('AOB') * real('BETA') + real('ALPHA')) / (real('AOB') + 1))
    elif trans == 'TRANS6':
        return (real('ALPHA') * real('BETA') / real('K21'),
                real('ALPHA') + real('BETA') - real('K21') - real('K'),
                real('K21'))
    else:
        return (real('K'),
                real('K12'),
                real('K21'))


def _advan4_trans(trans):
    if trans == 'TRANS3':
        return (real('CL') / real('V'),
                real('Q') / real('V'),
                real('Q') / (real('VSS') - real('V')),
                real('KA'))
    elif trans == 'TRANS4':
        return (real('CL') / real('V2'),
                real('Q') / real('V2'),
                real('Q') / real('V3'),
                real('KA'))
    elif trans == 'TRANS5':
        return (real('ALPHA') * real('BETA') / real('K32'),
                real('ALPHA') + real('BETA') - real('K32') - real('K'),
                (real('AOB') * real('BETA') + real('ALPHA')) / (real('AOB') + 1),
                real('KA'))
    elif trans == 'TRANS6':
        return (real('ALPHA') * real('BETA') / real('K32'),
                real('ALPHA') + real('BETA') - real('K32') - real('K'),
                real('K32'),
                real('KA'))
    else:
        return (real('K'),
                real('K23'),
                real('K32'),
                real('KA'))


def _advan11_trans(trans):
    if trans == 'TRANS4':
        return (real('CL') / real('V1'),
                real('Q2') / real('V1'),
                real('Q2') / real('V2'),
                real('Q3') / real('V1'),
                real('Q3') / real('V3'))
    elif trans == 'TRANS6':
        return (real('ALPHA') * real('BETA') * real('GAMMA') / (real('K21') * real('K31')),
                real('ALPHA') + real('BETA') + real('GAMMA') - real('K') - real('K13') -
                real('K21') - real('K31'),
                real('K21'),
                (real('ALPHA') * real('BETA') + real('ALPHA') * real('GAMMA') +
                 real('BETA') * real('GAMMA') + real('K31') * real('K31') -
                 real('K31') * (real('ALPHA') + real('BETA') + real('GAMMA')) -
                 real('K') * real('K21')) / (real('K21') - real('K31')),
                real('K31'))
    else:
        return (real('K'),
                real('K12'),
                real('K21'),
                real('K13'),
                real('K31'))


def _advan12_trans(trans):
    if trans == 'TRANS4':
        return (real('CL') / real('V2'),
                real('Q3') / real('V2'),
                real('Q3') / real('V3'),
                real('Q4') / real('V2'),
                real('Q4') / real('V4'),
                real('KA'))
    elif trans == 'TRANS6':
        return (real('ALPHA') * real('BETA') * real('GAMMA') / (real('K32') * real('K42')),
                real('ALPHA') + real('BETA') + real('GAMMA') - real('K') - real('K24') -
                real('K32') - real('K42'),
                real('K32'),
                (real('ALPHA') * real('BETA') + real('ALPHA') * real('GAMMA') +
                 real('BETA') * real('GAMMA') + real('K42') * real('K42') -
                 real('K42') * (real('ALPHA') + real('BETA') + real('GAMMA')) -
                 real('K') * real('K32')) / (real('K32') - real('K42')),
                real('K42'),
                real('KA'))
    else:
        return (real('K'),
                real('K23'),
                real('K32'),
                real('K24'),
                real('K42'),
                real('KA'))
