from pharmpy.statements import CompartmentalSystem, ExplicitODESystem


def to_explicit_odes(model):
    """Convert model from compartmental system to explicit ODE system
       or do nothing if it already has an explicit ODE system
    """
    statements = model.statements
    odes = statements.ode_system()
    if isinstance(odes, CompartmentalSystem):
        eqs, ics = odes.to_explicit_odes()
        new = ExplicitODESystem(eqs, ics)
        model.statements[model.statements.index(odes)] = new
    return model
