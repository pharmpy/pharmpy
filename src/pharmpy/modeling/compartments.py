from __future__ import annotations

from pharmpy.model import Model


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
    d = {}
    odes = model.statements.ode_system
    if odes is None:
        return d
    names = odes.compartment_names
    for name in names:
        cmt = odes.find_compartment_or_raise(name)
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
    d = {}
    odes = model.statements.ode_system
    if odes is None:
        return d
    names = odes.compartment_names
    for name in names:
        cmt = odes.find_compartment_or_raise(name)
        if cmt.bioavailability != 1:
            d[name] = cmt.bioavailability
    return d
