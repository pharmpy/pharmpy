from pharmpy.statements import CompartmentalSystem, ExplicitODESystem


def explicit_odes(model):
    """Convert model from compartmental system to explicit ODE system
       or do nothing if it already has an explicit ODE system
    """
    statements = model.statements
    odes = statements.ode_system
    if isinstance(odes, CompartmentalSystem):
        eqs, ics = odes.to_explicit_odes()
        new = ExplicitODESystem(eqs, ics)
        statements[model.statements.index(odes)] = new
        model.statements = statements
    return model


def absorption(model, order, rate=None):
    """Set or change the absorption for a model

        order - 0 or 1
    """
    statements = model.statements
    odes = statements.ode_system
    if not isinstance(odes, CompartmentalSystem):
        raise ValueError("Setting absorption is not supported for ExplicitODESystem")

    depot = odes.find_depot()
    if order == 0:
        if depot:
            ka = odes.get_flow(depot, odes.find_central())
            odes.remove_compartment(depot)
            symbols = ka.free_symbols
            for s in symbols:
                statements.remove_symbol_definition(s, odes)
            model.statements = statements
            model.remove_unused_parameters_and_rvs()

    return model
