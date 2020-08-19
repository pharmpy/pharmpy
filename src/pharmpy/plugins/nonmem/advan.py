import sympy

from pharmpy.statements import Assignment, CompartmentalSystem, IVBolus


def compartmental_model(model):
    cm = None
    sub = model.control_stream.get_records('SUBROUTINE')[0]
    if sub.has_option('ADVAN1'):
        cm = CompartmentalSystem()
        central = cm.add_compartment('CENTRAL')
        if sub.has_option('TRANS2'):
            out = sympy.Symbol('CL', real=True) / sympy.Symbol('V', real=True)
        else:
            out = sympy.Symbols('K', real=True)
        cm.add_flow(central, None, out)
        dose = IVBolus('AMT')
        central.dose = dose
        f = sympy.Symbol('F', real=True)
        fexpr = central.amount
        if model.statements.find_assignment('S1'):
            fexpr = fexpr / sympy.Symbol('S1', real=True)
        ass = Assignment(f, fexpr)
    return cm, ass
