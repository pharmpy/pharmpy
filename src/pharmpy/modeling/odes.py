import sympy

import pharmpy.symbols
from pharmpy.parameter import Parameter
from pharmpy.statements import Assignment, Bolus, CompartmentalSystem, ExplicitODESystem, Infusion


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


def set_transit_compartments(model, n):
    """Set the number of transit compartments of model"""
    statements = model.statements
    odes = statements.ode_system
    transits = odes.find_transit_compartments(statements)
    if len(transits) == n:
        pass
    elif len(transits) == 0:
        # FIXME
        pass
    elif len(transits) > n:
        nremove = len(transits) - n
        comp = odes.find_dosing()
        dose = comp.dose
        removed_symbols = set()
        while nremove > 0:
            to_comp, to_flow = odes.get_compartment_outflows(comp)[0]
            odes.remove_compartment(comp)
            removed_symbols |= to_flow.free_symbols
            comp = to_comp
            nremove -= 1
        comp.dose = dose
        statements.remove_symbol_definitions(removed_symbols, odes)
        model.remove_unused_parameters_and_rvs()
    else:
        nadd = n - len(transits)
        comp = odes.find_dosing()
        dose = comp.dose
        _, rate = odes.get_compartment_outflows(comp)[0]
        comp.dose = None
        while nadd > 0:
            new_comp = odes.add_compartment(f'TRANSIT{len(transits) + nadd}')
            nadd -= 1
            odes.add_flow(new_comp, comp, rate)
            comp = new_comp
        comp.dose = dose
    return model


def add_lag_time(model):
    """Add lag time to the dose compartment of model"""
    mdt_symb = model.create_symbol('MDT')
    odes = model.statements.ode_system
    dosing_comp = odes.find_dosing()
    old_lag_time = dosing_comp.lag_time
    dosing_comp.lag_time = mdt_symb
    if old_lag_time:
        model.statements.remove_symbol_definitions(old_lag_time.free_symbols, odes)
        model.remove_unused_parameters_and_rvs()
    tvmdt_symb = model.create_symbol('TVMDT')
    mdt_param = Parameter(tvmdt_symb.name, init=0.1, lower=0)
    model.parameters.add(mdt_param)
    imdt = Assignment(mdt_symb, mdt_param.symbol)
    model.statements.insert(0, imdt)
    return model


def remove_lag_time(model):
    """Remove lag time from the dose compartment of model"""
    odes = model.statements.ode_system
    dosing_comp = odes.find_dosing()
    lag_time = dosing_comp.lag_time
    if lag_time:
        symbols = lag_time.free_symbols
        dosing_comp.lag_time = 0
        model.statements.remove_symbol_definitions(symbols, odes)
        model.remove_unused_parameters_and_rvs()
    return model


def absorption_rate(model, rate):
    """Set or change the absorption rate of a model

    Parameters
    ----------
    model
        Model to set or change absorption for
    rate
        'bolus', 'ZO', 'FO' or 'seq-ZO-FO'
    """
    statements = model.statements
    odes = statements.ode_system
    if not isinstance(odes, CompartmentalSystem):
        raise ValueError("Setting absorption is not supported for ExplicitODESystem")

    depot = odes.find_depot()
    if rate == 'bolus':
        if depot:
            to_comp, _ = odes.get_compartment_outflows(depot)[0]
            to_comp.dose = depot.dose
            ka = odes.get_flow(depot, odes.find_central())
            odes.remove_compartment(depot)
            symbols = ka.free_symbols
            statements.remove_symbol_definitions(symbols, odes)
            model.remove_unused_parameters_and_rvs()
        if have_zero_order_absorption(model):
            dose_comp = odes.find_dosing()
            old_symbols = dose_comp.free_symbols
            dose_comp.dose = Bolus(dose_comp.dose.amount)
            unneeded_symbols = old_symbols - dose_comp.dose.free_symbols
            statements.remove_symbol_definitions(unneeded_symbols, odes)
            model.remove_unused_parameters_and_rvs()
    elif rate == 'ZO':
        dose_comp = odes.find_dosing()
        symbols = dose_comp.free_symbols
        dose = dose_comp.dose
        if depot:
            to_comp, _ = odes.get_compartment_outflows(depot)[0]
            ka = odes.get_flow(depot, odes.find_central())
            odes.remove_compartment(depot)
            symbols |= ka.free_symbols
            to_comp.dose = dose
        else:
            to_comp = dose_comp
        statements.remove_symbol_definitions(symbols, odes)
        model.remove_unused_parameters_and_rvs()
        if not have_zero_order_absorption(model):
            add_zero_order_absorption(model, dose.amount, to_comp, 'MAT')
    elif rate == 'FO':
        dose_comp = odes.find_dosing()
        amount = dose_comp.dose.amount
        symbols = dose_comp.free_symbols
        if depot:
            dose_comp.dose = Bolus(depot.dose.amount)
        else:
            dose_comp.dose = None
        statements.remove_symbol_definitions(symbols, odes)
        model.remove_unused_parameters_and_rvs()
        if not depot:
            add_first_order_absorption(model, Bolus(amount), dose_comp)
    elif rate == 'seq-ZO-FO':
        dose_comp = odes.find_dosing()
        have_ZO = have_zero_order_absorption(model)
        if depot and not have_ZO:
            add_zero_order_absorption(model, dose_comp.amount, depot, 'MDT')
        elif not depot and have_ZO:
            add_first_order_absorption(model, dose_comp.dose, dose_comp)
            dose_comp.dose = None
        elif not depot and not have_ZO:
            amount = dose_comp.dose.amount
            dose_comp.dose = None
            depot = add_first_order_absorption(model, amount, dose_comp)
            add_zero_order_absorption(model, amount, depot, 'MDT')
    else:
        raise ValueError(
            f'Requested rate {rate} but only rates  ' f'bolus, FO, ZO and seq-ZO-FO are supported'
        )
    return model


def have_zero_order_absorption(model):
    """Check if ode system describes a zero order absorption

    currently defined as having Infusion dose with rate not in dataset
    """
    odes = model.statements.ode_system
    dosing = odes.find_dosing()
    dose = dosing.dose
    if isinstance(dose, Infusion):
        if dose.rate is None:
            value = dose.duration
        else:
            value = dose.rate
        if isinstance(value, sympy.Symbol) or isinstance(value, str):
            name = str(value)
            if name not in model.dataset.columns:
                return True
    return False


def add_zero_order_absorption(model, amount, to_comp, parameter_name):
    """Add zero order absorption to a compartment.
    Disregards what is currently in the model.
    """
    tvmat_symb = model.create_symbol(f'TV{parameter_name}')
    mat_param = Parameter(tvmat_symb.name, init=0.1, lower=0)
    model.parameters.add(mat_param)
    mat_symb = model.create_symbol(parameter_name)
    imat = Assignment(mat_symb, mat_param.symbol)
    new_dose = Infusion(amount, duration=mat_symb * 2)
    to_comp.dose = new_dose
    model.statements.insert(0, imat)


def add_first_order_absorption(model, dose, to_comp):
    """Add first order absorption
    Disregards what is currently in the model.
    """
    odes = model.statements.ode_system
    depot = odes.add_compartment('DEPOT')
    depot.dose = dose
    tvmat_symb = model.create_symbol('TVMAT')
    mat_param = Parameter(tvmat_symb.name, init=0.1, lower=0)
    model.parameters.add(mat_param)
    mat_symb = model.create_symbol('MAT')
    imat = Assignment(mat_symb, mat_param.symbol)
    model.statements.insert(0, imat)
    odes.add_flow(depot, to_comp, 1 / pharmpy.symbols.symbol('MAT'))
    return depot
