def get_lag_times(model):
    """Get lag times for all compartments

    Parameters
    ----------
    model : Pharmpy model

    Result
    ------
    dict
        Dictionary from compartment name to lag time expression
    """
    odes = model.statements.ode_system
    names = odes.compartment_names
    d = dict()
    for name in names:
        cmt = odes.find_compartment(name)
        if cmt.lag_time:
            d[name] = cmt.lag_time
    return d


def get_bioavailability(model):
    """Get bioavailability of doses for all compartments

    Parameters
    ----------
    model : Pharmpy model

    Result
    ------
    dict
        Dictionary from compartment name to bioavailability expression
    """
    odes = model.statements.ode_system
    names = odes.compartment_names
    d = dict()
    for name in names:
        cmt = odes.find_compartment(name)
        if cmt.bioavailability:
            d[name] = cmt.bioavailability
    return d
