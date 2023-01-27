from pharmpy.model import Model, ODESystem


def get_lag_times(model: Model):
    """Get lag times for all compartments

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    dict
        Dictionary from compartment name to lag time expression
    """
    odes = model.statements.ode_system
    assert isinstance(odes, ODESystem)
    names = odes.compartment_names
    d = {}
    for name in names:
        cmt = odes.find_compartment(name)
        if cmt.lag_time != 0:
            d[name] = cmt.lag_time
    return d


def get_bioavailability(model: Model):
    """Get bioavailability of doses for all compartments

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    dict
        Dictionary from compartment name to bioavailability expression
    """
    odes = model.statements.ode_system
    assert isinstance(odes, ODESystem)
    names = odes.compartment_names
    d = {}
    for name in names:
        cmt = odes.find_compartment(name)
        if cmt.bioavailability != 1:
            d[name] = cmt.bioavailability
    return d
