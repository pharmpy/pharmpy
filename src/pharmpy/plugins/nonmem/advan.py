import sympy

from pharmpy.statements import Assignment, CompartmentalSystem, IVBolus


def compartmental_model(model, advan, trans):
    cm = None
    if advan == 'ADVAN1':
        cm = CompartmentalSystem()
        central = cm.add_compartment('CENTRAL')
        _first_order_elimination(cm, central, trans)
        dose = IVBolus('AMT')
        central.dose = dose
        ass = _f_link_assignment(model, central)
    elif advan == 'ADVAN2':
        cm = CompartmentalSystem()
        depot = cm.add_compartment('DEPOT')
        central = cm.add_compartment('CENTRAL')
        _first_order_elimination(cm, central, trans)
        cm.add_flow(depot, central, sympy.Symbol('KA', real=True))
        dose = IVBolus('AMT')
        depot.dose = dose
        ass = _f_link_assignment(model, central)
    return cm, ass


def _f_link_assignment(model, compartment):
    f = sympy.Symbol('F', real=True)
    fexpr = compartment.amount
    pkrec = model.control_stream.get_records('PK')[0]
    if pkrec.statements.find_assignment('S1'):
        fexpr = fexpr / sympy.Symbol('S1', real=True)
    ass = Assignment(f, fexpr)
    return ass


def _first_order_elimination(cm, compartment, trans):
    if trans == 'TRANS2':
        out = sympy.Symbol('CL', real=True) / sympy.Symbol('V', real=True)
    else:
        out = sympy.Symbol('K', real=True)
    cm.add_flow(compartment, None, out)
