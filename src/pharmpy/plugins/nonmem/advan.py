import sympy

from pharmpy.statements import Assignment, CompartmentalSystem, IVBolus


def real(name):
    return sympy.Symbol(name, real=True)


def compartmental_model(model, advan, trans):
    if advan == 'ADVAN1':
        cm = CompartmentalSystem()
        central = cm.add_compartment('CENTRAL')
        output = cm.add_compartment('OUTPUT')
        cm.add_flow(central, output, _advan12_trans(trans))
        dose = IVBolus('AMT')
        central.dose = dose
        ass = _f_link_assignment(model, central)
    elif advan == 'ADVAN2':
        cm = CompartmentalSystem()
        depot = cm.add_compartment('DEPOT')
        central = cm.add_compartment('CENTRAL')
        output = cm.add_compartment('OUTPUT')
        cm.add_flow(central, output, _advan12_trans(trans))
        cm.add_flow(depot, central, sympy.Symbol('KA', real=True))
        dose = IVBolus('AMT')
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
        dose = IVBolus('AMT')
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
        dose = IVBolus('AMT')
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


def _advan12_trans(trans):
    if trans == 'TRANS2':
        return [sympy.Symbol('CL', real=True) / sympy.Symbol('V', real=True)]
    else:       # TRANS1 which is also the default
        return [sympy.Symbol('K', real=True)]


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
