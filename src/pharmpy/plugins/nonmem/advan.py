import sympy

from pharmpy.statements import Assignment, Bolus, CompartmentalSystem


def real(name):
    return sympy.Symbol(name, real=True)


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
        cm.add_flow(depot, central, sympy.Symbol('KA', real=True))
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
    f = sympy.Symbol('F', real=True)
    fexpr = compartment.amount
    pkrec = model.control_stream.get_records('PK')[0]
    if pkrec.statements.find_assignment('S1'):
        fexpr = fexpr / sympy.Symbol('S1', real=True)
    ass = Assignment(f, fexpr)
    return ass


def _advan1and2_trans(trans):
    if trans == 'TRANS2':
        return sympy.Symbol('CL', real=True) / sympy.Symbol('V', real=True)
    else:       # TRANS1 which is also the default
        return sympy.Symbol('K', real=True)


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
