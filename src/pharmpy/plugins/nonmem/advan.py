import sympy

from pharmpy.statements import ODE, Assignment


def advan_equations(model):
    sub = model.control_stream.get_records('SUBROUTINE')[0]
    t = sympy.Symbol('t', real=True)
    statements = model.statements
    if sub.has_option('ADVAN1'):
        A_1 = sympy.Function('A_1')
        dAdt = sympy.Derivative(A_1(t), t)
        f = sympy.Symbol('F', real=True)
        if sub.has_option('TRANS2'):
            CL = sympy.Symbol('CL', real=True)
            V = sympy.Symbol('V', real=True)
            eq = sympy.Eq(dAdt, -(CL / V) * A_1(t))
        elif sub.has_option('TRANS1'):
            K = sympy.Symbol('K', real=True)
            eq = sympy.Eq(dAdt, -K * A_1(t))
        ode = ODE()
        ode.equation = eq
        ics = {A_1(0): sympy.Symbol('AMT', real=True)}     # FIXME: Only handles bolus dose via AMT
        ode.ics = ics
        fexpr = sympy.Symbol(A_1.name, real=True)
        if statements.find_assignment('S1'):
            fexpr = fexpr / sympy.Symbol('S1', real=True)
        ass = Assignment(f, fexpr)
    return ode, ass
