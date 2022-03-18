"""
:meta private:
"""

import sympy

import pharmpy.symbols
from pharmpy.model import ModelError
from pharmpy.modeling.help_functions import _as_integer
from pharmpy.parameter import Parameter
from pharmpy.statements import Assignment, Bolus, Infusion

from .common import remove_unused_parameters_and_rvs
from .data import get_observations
from .expressions import create_symbol


def _extract_params_from_symb(statements, symbol_name, pset):
    terms = {
        symb.name
        for symb in statements.before_odes.full_expression(sympy.Symbol(symbol_name)).free_symbols
    }
    theta_name = terms.intersection(pset.names).pop()
    return pset[theta_name]


def add_individual_parameter(model, name):
    """Add an individual or pk parameter to a model

    Parameters
    ----------
    model : Model
        Pharmpy model
    name : str
        Name of individual/pk parameter

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> add_individual_parameter(model, "KA")   # doctest: +ELLIPSIS
    <...>
    >>> model.statements.find_assignment("KA")
    KA := POP_KA

    """
    _add_parameter(model, name)
    return model


def _add_parameter(model, name, init=0.1):
    pops = create_symbol(model, f'POP_{name}')
    pop_param = Parameter(pops.name, init=init, lower=0)
    model.parameters.append(pop_param)
    symb = create_symbol(model, name)
    ass = Assignment(symb, pop_param.symbol)
    model.statements.insert(0, ass)
    return symb


def set_first_order_elimination(model):
    """Sets elimination to first order

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_first_order_elimination(model)     # doctest: +ELLIPSIS
    <...>
    >>> model.statements.ode_system
    Bolus(AMT)
    ┌───────┐       ┌──────┐
    │CENTRAL│──CL/V→│OUTPUT│
    └───────┘       └──────┘

    See also
    --------
    set_zero_order_elimination
    set_michaelis_menten_elimination

    """
    if has_first_order_elimination(model):
        pass
    elif has_zero_order_elimination(model) or has_michaelis_menten_elimination(model):
        model.parameters['POP_CLMM'].name = 'POP_CL'
        ass = model.statements.find_assignment('CLMM')
        ass.symbol = 'CL'
        ass.subs({'POP_CLMM': 'POP_CL'})
        odes = model.statements.ode_system
        central = odes.central_compartment
        output = odes.output_compartment
        v = sympy.Symbol('V')
        rate = odes.get_flow(central, output)
        if v not in rate.free_symbols:
            v = sympy.Symbol('VC')
        odes.remove_flow(central, output)
        odes.add_flow(central, output, sympy.Symbol('CL') / v)
        model.statements.remove_symbol_definitions({sympy.Symbol('KM')}, odes)
        remove_unused_parameters_and_rvs(model)
    elif has_mixed_mm_fo_elimination(model):
        odes = model.statements.ode_system
        central = odes.central_compartment
        output = odes.output_compartment
        v = sympy.Symbol('V')
        rate = odes.get_flow(central, output)
        if v not in rate.free_symbols:
            v = sympy.Symbol('VC')
        odes.remove_flow(central, output)
        odes.add_flow(central, output, sympy.Symbol('CL') / v)
        model.statements.remove_symbol_definitions({sympy.Symbol('KM'), sympy.Symbol('CLMM')}, odes)
        remove_unused_parameters_and_rvs(model)
    return model


def set_zero_order_elimination(model):
    """Sets elimination to zero order.

    Initial estimate for KM is set to 1% of smallest observation.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_zero_order_elimination(model)     # doctest: +ELLIPSIS
    <...>
    >>> model.statements.ode_system
    Bolus(AMT)
    ┌───────┐                                    ┌──────┐
    │CENTRAL│──CLMM*KM/(V*(KM + A_CENTRAL(t)/V))→│OUTPUT│
    └───────┘                                    └──────┘

    See also
    --------
    set_first_order_elimination
    set_michaelis_menten_elimination

    """
    if has_zero_order_elimination(model):
        pass
    elif has_michaelis_menten_elimination(model):
        model.parameters['POP_KM'].fix = True
    elif has_mixed_mm_fo_elimination(model):
        model.parameters['POP_KM'].fix = True
        odes = model.statements.ode_system
        central = odes.central_compartment
        output = odes.output_compartment
        rate = odes.get_flow(central, output)
        rate = rate.subs('CL', 0)
        odes.remove_flow(central, output)
        odes.add_flow(central, output, rate)
        model.statements.remove_symbol_definitions({sympy.Symbol('CL')}, odes)
        remove_unused_parameters_and_rvs(model)
    else:
        _do_michaelis_menten_elimination(model)
        obs = get_observations(model)
        init = obs.min() / 100  # 1% of smallest observation
        model.parameters['POP_KM'].init = init
        model.parameters['POP_KM'].fix = True
    return model


def has_michaelis_menten_elimination(model):
    """Check if the model describes Michaelis-Menten elimination

    This function relies on heuristics and will not be able to detect all
    possible ways of coding the Michalis-Menten elimination.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    bool
        True if model has describes Michaelis-Menten elimination

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> has_michaelis_menten_elimination(model)
    False
    >>> set_michaelis_menten_elimination(model)   # doctest: +ELLIPSIS
    <...>
    >>> has_michaelis_menten_elimination(model)
    True
    """
    model.statements.to_compartmental_system()
    odes = model.statements.ode_system
    output = odes.output_compartment
    central = odes.central_compartment
    rate = odes.get_flow(central, output)
    is_nonlinear = odes.t in rate.free_symbols
    is_zero_order = 'POP_KM' in model.parameters and model.parameters['POP_KM'].fix
    could_be_mixed = sympy.Symbol('CL') in rate.free_symbols
    return is_nonlinear and not is_zero_order and not could_be_mixed


def has_zero_order_elimination(model):
    """Check if the model describes zero-order elimination

    This function relies on heuristics and will not be able to detect all
    possible ways of coding the zero-order elimination.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    bool
        True if model has describes zero order elimination

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> has_zero_order_elimination(model)
    False
    >>> set_zero_order_elimination(model)   # doctest: +ELLIPSIS
    <...>
    >>> has_zero_order_elimination(model)
    True
    """
    model.statements.to_compartmental_system()
    odes = model.statements.ode_system
    output = odes.output_compartment
    central = odes.central_compartment
    rate = odes.get_flow(central, output)
    is_nonlinear = odes.t in rate.free_symbols
    is_zero_order = 'POP_KM' in model.parameters and model.parameters['POP_KM'].fix
    could_be_mixed = sympy.Symbol('CL') in rate.free_symbols
    return is_nonlinear and is_zero_order and not could_be_mixed


def has_mixed_mm_fo_elimination(model):
    """Check if the model describes mixed Michaelis-Menten and first order elimination

    This function relies on heuristics and will not be able to detect all
    possible ways of coding the mixed Michalis-Menten and first order elimination.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    bool
        True if model has describes Michaelis-Menten elimination

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> has_mixed_mm_fo_elimination(model)
    False
    >>> set_mixed_mm_fo_elimination(model)   # doctest: +ELLIPSIS
    <...>
    >>> has_mixed_mm_fo_elimination(model)
    True
    """
    model.statements.to_compartmental_system()
    odes = model.statements.ode_system
    output = odes.output_compartment
    central = odes.central_compartment
    rate = odes.get_flow(central, output)
    is_nonlinear = odes.t in rate.free_symbols
    is_zero_order = 'POP_KM' in model.parameters and model.parameters['POP_KM'].fix
    could_be_mixed = sympy.Symbol('CL') in rate.free_symbols
    return is_nonlinear and not is_zero_order and could_be_mixed


def has_first_order_elimination(model):
    """Check if the model describes first order elimination

    This function relies on heuristics and will not be able to detect all
    possible ways of coding the first order elimination.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    bool
        True if model has describes first order elimination

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> has_first_order_elimination(model)
    True

    """
    model.statements.to_compartmental_system()
    odes = model.statements.ode_system
    output = odes.output_compartment
    central = odes.central_compartment
    rate = odes.get_flow(central, output)
    is_nonlinear = odes.t in rate.free_symbols
    return not is_nonlinear


def set_michaelis_menten_elimination(model):
    """Sets elimination to Michaelis-Menten.

    Initial estimate for CLMM is set to CL and KM is set to :math:`2*max(DV)`.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_michaelis_menten_elimination(model)     # doctest: +ELLIPSIS
    <...>
    >>> model.statements.ode_system
    Bolus(AMT)
    ┌───────┐                                    ┌──────┐
    │CENTRAL│──CLMM*KM/(V*(KM + A_CENTRAL(t)/V))→│OUTPUT│
    └───────┘                                    └──────┘

    See also
    --------
    set_first_order_elimination
    set_zero_order_elimination

    """
    if has_michaelis_menten_elimination(model):
        pass
    elif has_zero_order_elimination(model):
        model.parameters['POP_KM'].fix = False
    elif has_mixed_mm_fo_elimination(model):
        odes = model.statements.ode_system
        central = odes.central_compartment
        output = odes.output_compartment
        rate = odes.get_flow(central, output)
        rate = rate.subs('CL', 0)
        odes.remove_flow(central, output)
        odes.add_flow(central, output, rate)
        model.statements.remove_symbol_definitions({sympy.Symbol('CL')}, odes)
        remove_unused_parameters_and_rvs(model)
    else:
        _do_michaelis_menten_elimination(model)
    return model


def set_mixed_mm_fo_elimination(model):
    """Sets elimination to mixed Michaelis-Menten and first order.

    Initial estimate for CLMM is set to CL/2 and KM is set to :math:`2*max(DV)`.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_mixed_mm_fo_elimination(model)     # doctest: +ELLIPSIS
    <...>
    >>> model.statements.ode_system
    Bolus(AMT)
    ┌───────┐                                         ┌──────┐
    │CENTRAL│──(CL + CLMM*KM/(KM + A_CENTRAL(t)/V))/V→│OUTPUT│
    └───────┘                                         └──────┘

    See also
    --------
    set_first_order_elimination
    set_zero_order_elimination
    set_michaelis_menten_elimination

    """
    if has_mixed_mm_fo_elimination(model):
        pass
    elif has_michaelis_menten_elimination(model) or has_zero_order_elimination(model):
        model.parameters['POP_KM'].fix = False
        odes = model.statements.ode_system
        central = odes.central_compartment
        output = odes.output_compartment
        add_individual_parameter(model, 'CL')
        rate = odes.get_flow(central, output)
        odes.remove_flow(central, output)
        v = sympy.Symbol('V')
        if v not in rate.free_symbols:
            v = sympy.Symbol('VC')
        odes.add_flow(central, output, sympy.Symbol('CL') / v + rate)
    else:
        _do_michaelis_menten_elimination(model, combined=True)
    return model


def _do_michaelis_menten_elimination(model, combined=False):
    model.statements.to_compartmental_system()
    sset = model.statements
    odes = sset.ode_system
    central = odes.central_compartment
    output = odes.output_compartment
    old_rate = odes.get_flow(central, output)
    numer, denom = old_rate.as_numer_denom()

    km_init, clmm_init = _get_mm_inits(model, numer, combined)

    km = _add_parameter(model, 'KM', init=km_init)
    model.parameters['POP_KM'].upper = 20 * get_observations(model).max()

    if denom != 1:
        if combined:
            cl = numer
            clmm = _add_parameter(model, 'CLMM', init=clmm_init)
        else:
            _rename_parameter(model, 'CL', 'CLMM')
            clmm = sympy.Symbol('CLMM')
            cl = 0
        vc = denom
    else:
        if combined:
            if sset.find_assignment('CL'):
                cl = sset.find_assignment('CL').symbol
            else:
                cl = _add_parameter(model, 'CL', clmm_init)
        else:
            cl = 0
        if sset.find_assignment('VC'):
            vc = sset.find_assignment('VC').symbol
        else:
            vc = _add_parameter(model, 'VC')  # FIXME: decide better initial estimate
        if not combined and sset.find_assignment('CL'):
            _rename_parameter(model, 'CL', 'CLMM')
            clmm = sset.find_assignment('CLMM').symbol
        else:
            clmm = _add_parameter(model, 'CLMM', init=clmm_init)

    amount = sympy.Function(central.amount.name)(pharmpy.symbols.symbol('t'))
    rate = (clmm * km / (km + amount / vc) + cl) / vc
    odes.add_flow(central, output, rate)
    model.statements.remove_symbol_definitions(numer.free_symbols, odes)
    remove_unused_parameters_and_rvs(model)
    return model


def _rename_parameter(model, old_name, new_name):
    a = model.statements.find_assignment(old_name)
    for s in a.rhs_symbols:
        if s in model.parameters:
            old_par = s
            model.parameters[s].name = f'POP_{new_name}'
            new_par = sympy.Symbol(f'POP_{new_name}')
            model.statements.subs({old_par: new_par})
            break
    for s in a.rhs_symbols:
        if s in model.random_variables.iiv:
            rv = model.random_variables[s]
            cov = model.random_variables.iiv.covariance_matrix
            ind = model.random_variables.iiv.index(rv)
            pars = [e for e in cov[ind, :] if e.is_Symbol]
            diag = cov[ind, ind]
            d = {diag: f'IIV_{new_name}'}
            for p in pars:
                if p != diag:
                    if p.name.startswith('IIV'):
                        d[p] = p.name.replace(f'IIV_{old_name}', f'IIV_{new_name}')
            model.random_variables.subs(d)
            for key, val in d.items():
                model.parameters[key].name = val
            break
    model.statements.subs({old_name: new_name})


def _get_mm_inits(model, rate_numer, combined):
    pset, sset = model.parameters, model.statements
    clmm_init = _extract_params_from_symb(sset, rate_numer.name, pset).init

    if combined:
        clmm_init /= 2

    dv_max = get_observations(model).max()
    km_init = dv_max * 2

    return km_init, clmm_init


def set_transit_compartments(model, n, keep_depot=True):
    """Set the number of transit compartments of model.

    Initial estimate for absorption rate is
    set the previous rate if available, otherwise it is set to the time of first observation/2.

    Parameters
    ----------
    model : Model
        Pharmpy model
    n : int
        Number of transit compartments
    keep_depot : bool
        False to convert depot compartment into a transit compartment

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_transit_compartments(model, 3)     # doctest: +ELLIPSIS
    <...>
    >>> model.statements.ode_system
    Bolus(AMT)
    ┌────────┐        ┌────────┐        ┌────────┐        ┌───────┐       ┌──────┐
    │TRANSIT1│──3/MDT→│TRANSIT2│──3/MDT→│TRANSIT3│──3/MDT→│CENTRAL│──CL/V→│OUTPUT│
    └────────┘        └────────┘        └────────┘        └───────┘       └──────┘

    See also
    --------
    add_lag_time

    """
    statements = model.statements
    statements.to_compartmental_system()
    odes = statements.ode_system
    transits = odes.find_transit_compartments(statements)
    try:
        n = _as_integer(n)
    except ValueError:
        raise ValueError(f'Number of compartments must be integer: {n}')

    # Handle keep_depot option
    depot = odes.find_depot(statements)
    mdt_init = None
    if not keep_depot and depot:
        central = odes.central_compartment
        rate = odes.get_flow(depot, central)
        if not rate.is_Symbol:
            num, den = rate.as_numer_denom()
            if num == 1 and den.is_Symbol:
                symbol = den
            else:
                symbol = None
        else:
            symbol = rate
        if symbol:
            mdt_init = _extract_params_from_symb(statements, symbol.name, model.parameters).init
        inflows = odes.get_compartment_inflows(depot)
        if len(inflows) == 1:
            innode, inflow = inflows[0]
            odes.add_flow(innode, central, inflow)
        else:
            central.dose = depot.dose
        odes.remove_compartment(depot)
        statements.remove_symbol_definitions(rate.free_symbols, odes)
        remove_unused_parameters_and_rvs(model)

    if len(transits) == n:
        pass
    elif len(transits) == 0:
        if mdt_init is not None:
            init = mdt_init
        else:
            init = _get_absorption_init(model, 'MDT')
        mdt_symb = _add_parameter(model, 'MDT', init=init)
        rate = n / mdt_symb
        comp = odes.dosing_compartment
        dose = comp.dose
        comp.dose = None
        while n > 0:
            new_comp = odes.add_compartment(f'TRANSIT{n}')
            n -= 1
            odes.add_flow(new_comp, comp, rate)
            comp = new_comp
        comp.dose = dose
    elif len(transits) > n:
        nremove = len(transits) - n
        removed_symbols = set()
        trans, destination, flow = _find_last_transit(odes, transits)
        if n == 0:  # The dosing compartment will be removed
            dosing = odes.dosing_compartment
            dose = dosing.dose
        remaining = set(transits)
        while nremove > 0:
            from_comp, from_flow = odes.get_compartment_inflows(trans)[0]
            odes.add_flow(from_comp, destination, from_flow)
            odes.remove_compartment(trans)
            remaining.remove(trans)
            removed_symbols |= flow.free_symbols
            trans = from_comp
            flow = from_flow
            nremove -= 1
        if n == 0:
            destination.dose = dose
        _update_numerators(model)
        statements.remove_symbol_definitions(removed_symbols, odes)
        remove_unused_parameters_and_rvs(model)
    else:
        nadd = n - len(transits)
        last, destination, rate = _find_last_transit(odes, transits)
        odes.remove_flow(last, destination)
        while nadd > 0:
            new_comp = odes.add_compartment(f'TRANSIT{n - nadd + 1}')
            odes.add_flow(last, new_comp, rate)
            if rate.is_Symbol:
                ass = statements.find_assignment(rate.name)
                if ass is not None:
                    rate = ass.expression
            last = new_comp
            nadd -= 1
        odes.add_flow(last, destination, rate)
        _update_numerators(model)
    return model


def _find_last_transit(odes, transits):
    for trans in transits:
        destination, flow = odes.get_compartment_outflows(trans)[0]
        if destination not in transits:
            return trans, destination, flow


def _update_numerators(model):
    # update numerators for transit compartment rates
    statements = model.statements
    odes = statements.ode_system
    transits = odes.find_transit_compartments(statements)
    new_numerator = sympy.Integer(len(transits))
    for comp in transits:
        to_comp, rate = odes.get_compartment_outflows(comp)[0]
        numer, denom = rate.as_numer_denom()
        if numer.is_Integer and numer != new_numerator:
            new_rate = new_numerator / denom
            odes.add_flow(comp, to_comp, new_rate)
        elif numer.is_Symbol:
            ass = statements.find_assignment(numer.name)
            if ass is not None:
                ass_numer, ass_denom = ass.expression.as_numer_denom()
                if ass_numer.is_Integer and ass_numer != new_numerator:
                    new_rate = new_numerator / ass_denom
                    statements.reassign(numer, new_rate)


def add_lag_time(model):
    """Add lag time to the dose compartment of model.

    Initial estimate for lag time is set the
    previous lag time if available, otherwise it is set to the time of first observation/2.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> add_lag_time(model)     # doctest: +ELLIPSIS
    <...>

    See also
    --------
    set_transit_compartments
    remove_lag_time

    """
    model.statements.to_compartmental_system()
    odes = model.statements.ode_system
    dosing_comp = odes.dosing_compartment
    old_lag_time = dosing_comp.lag_time
    mdt_symb = _add_parameter(model, 'MDT', init=_get_absorption_init(model, 'MDT'))
    dosing_comp.lag_time = mdt_symb
    if old_lag_time:
        model.statements.remove_symbol_definitions(old_lag_time.free_symbols, odes)
        remove_unused_parameters_and_rvs(model)
    return model


def remove_lag_time(model):
    """Remove lag time from the dose compartment of model.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> remove_lag_time(model)     # doctest: +ELLIPSIS
    <...>

    See also
    --------
    set_transit_compartments
    add_lag_time


    """
    model.statements.to_compartmental_system()
    odes = model.statements.ode_system
    dosing_comp = odes.dosing_compartment
    lag_time = dosing_comp.lag_time
    if lag_time:
        symbols = lag_time.free_symbols
        dosing_comp.lag_time = 0
        model.statements.remove_symbol_definitions(symbols, odes)
        remove_unused_parameters_and_rvs(model)
    return model


def set_zero_order_absorption(model):
    """Set or change to zero order absorption rate.

    Initial estimate for absorption rate is set
    the previous rate if available, otherwise it is set to the time of first observation/2.

    Parameters
    ----------
    model : Model
        Model to set or change to first order absorption rate

    Return
    ------
    Model
        Reference to the same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_zero_order_absorption(model)     # doctest: +ELLIPSIS
    <...>
    >>> model.statements.ode_system
    Infusion(AMT, duration=2*MAT)
    ┌───────┐       ┌──────┐
    │CENTRAL│──CL/V→│OUTPUT│
    └───────┘       └──────┘

    See also
    --------
    set_bolus_order_absorption
    set_first_order_absorption

    """
    statements = model.statements
    statements.to_compartmental_system()
    odes = statements.ode_system
    _disallow_infusion(model)
    depot = odes.find_depot(statements)

    dose_comp = odes.dosing_compartment
    symbols = dose_comp.free_symbols
    dose = dose_comp.dose
    lag_time = dose_comp.lag_time
    if depot:
        to_comp, _ = odes.get_compartment_outflows(depot)[0]
        ka = odes.get_flow(depot, odes.central_compartment)
        odes.remove_compartment(depot)
        symbols = ka.free_symbols
        to_comp.dose = dose
    else:
        to_comp = dose_comp
    mat_assign = statements.find_assignment('MAT')
    if mat_assign:
        mat_idx = statements.index(mat_assign)
    statements.remove_symbol_definitions(symbols, odes)
    if mat_assign:
        statements.insert(mat_idx, mat_assign)
    remove_unused_parameters_and_rvs(model)
    if not has_zero_order_absorption(model):
        _add_zero_order_absorption(model, dose.amount, to_comp, 'MAT', lag_time)
    return model


def set_first_order_absorption(model):
    """Set or change to first order absorption rate.

    Initial estimate for absorption rate is set to
    the previous rate if available, otherwise it is set to the time of first observation/2.

    Parameters
    ----------
    model : Model
        Model to set or change to use first order absorption rate

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_first_order_absorption(model)     # doctest: +ELLIPSIS
    <...>
    >>> model.statements.ode_system
    Bolus(AMT)
    ┌─────┐        ┌───────┐       ┌──────┐
    │DEPOT│──1/MAT→│CENTRAL│──CL/V→│OUTPUT│
    └─────┘        └───────┘       └──────┘

    See also
    --------
    set_bolus_order_absorption
    set_zero_order_absorption

    """
    statements = model.statements
    statements.to_compartmental_system()
    odes = statements.ode_system
    depot = odes.find_depot(statements)

    dose_comp = odes.dosing_compartment
    amount = dose_comp.dose.amount
    symbols = dose_comp.free_symbols
    lag_time = dose_comp.lag_time
    if depot and depot == dose_comp:
        dose_comp.dose = Bolus(dose_comp.dose.amount)
    elif not depot:
        dose_comp.dose = None
    mat_assign = statements.find_assignment('MAT')
    if mat_assign:
        mat_idx = statements.index(mat_assign)
    statements.remove_symbol_definitions(symbols, odes)
    if mat_assign:
        statements.insert(mat_idx, mat_assign)
    remove_unused_parameters_and_rvs(model)
    if not depot:
        _add_first_order_absorption(model, Bolus(amount), dose_comp, lag_time)
    return model


def set_bolus_absorption(model):
    """Set or change to bolus absorption rate.

    Currently lagtime together with bolus absorption is not supported.

    Parameters
    ----------
    model : Model
        Model to set or change absorption rate

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_bolus_absorption(model)     # doctest: +ELLIPSIS
    <...>
    >>> model.statements.ode_system
    Bolus(AMT)
    ┌───────┐       ┌──────┐
    │CENTRAL│──CL/V→│OUTPUT│
    └───────┘       └──────┘

    See also
    --------
    set_zero_order_absorption
    set_first_order_absorption

    """
    statements = model.statements
    statements.to_compartmental_system()
    odes = statements.ode_system
    depot = odes.find_depot(statements)
    if depot:
        to_comp, _ = odes.get_compartment_outflows(depot)[0]
        to_comp.dose = depot.dose
        ka = odes.get_flow(depot, odes.central_compartment)
        odes.remove_compartment(depot)
        symbols = ka.free_symbols
        statements.remove_symbol_definitions(symbols, odes)
        remove_unused_parameters_and_rvs(model)
    if has_zero_order_absorption(model):
        dose_comp = odes.dosing_compartment
        old_symbols = dose_comp.free_symbols
        dose_comp.dose = Bolus(dose_comp.dose.amount)
        unneeded_symbols = old_symbols - dose_comp.dose.free_symbols
        statements.remove_symbol_definitions(unneeded_symbols, odes)
        remove_unused_parameters_and_rvs(model)
    return model


def set_seq_zo_fo_absorption(model):
    """Set or change to sequential zero order first order absorption rate.

    Initial estimate for
    absorption rate is set the previous rate if available, otherwise it is set to the time of
    first observation/2.

    Currently lagtime together with sequential zero order first order absorption is not
    supported.

    Parameters
    ----------
    model : Model
        Model to set or change absorption rate

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_seq_zo_fo_absorption(model)     # doctest: +ELLIPSIS
    <...>
    >>> model.statements.ode_system
    Infusion(AMT, duration=2*MDT)
    ┌─────┐        ┌───────┐       ┌──────┐
    │DEPOT│──1/MAT→│CENTRAL│──CL/V→│OUTPUT│
    └─────┘        └───────┘       └──────┘

    See also
    --------
    set_bolus_order_absorption
    set_zero_order_absorption
    set_first_order_absorption

    """
    statements = model.statements
    statements.to_compartmental_system()
    odes = statements.ode_system
    _disallow_infusion(model)
    depot = odes.find_depot(statements)

    dose_comp = odes.dosing_compartment
    have_ZO = has_zero_order_absorption(model)
    if depot and not have_ZO:
        _add_zero_order_absorption(model, dose_comp.amount, depot, 'MDT')
    elif not depot and have_ZO:
        _add_first_order_absorption(model, dose_comp.dose, dose_comp)
        dose_comp.dose = None
    elif not depot and not have_ZO:
        amount = dose_comp.dose.amount
        dose_comp.dose = None
        depot = _add_first_order_absorption(model, Bolus(amount), dose_comp)
        _add_zero_order_absorption(model, amount, depot, 'MDT')
    return model


def _disallow_infusion(model):
    odes = model.statements.ode_system
    dose_comp = odes.dosing_compartment
    if isinstance(dose_comp.dose, Infusion):
        if dose_comp.dose.rate is not None:
            ex = dose_comp.dose.rate
        else:
            ex = dose_comp.dose.duration
        for s in ex.free_symbols:
            if s.name in model.datainfo.names:
                raise ModelError("Model already has an infusion given in the dataset")


def has_zero_order_absorption(model):
    """Check if ode system describes a zero order absorption

    currently defined as having Infusion dose with rate not in dataset

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> has_zero_order_absorption(model)
    False

    """
    odes = model.statements.ode_system
    dosing = odes.dosing_compartment
    dose = dosing.dose
    if isinstance(dose, Infusion):
        if dose.rate is None:
            value = dose.duration
        else:
            value = dose.rate
        if isinstance(value, sympy.Symbol) or isinstance(value, str):
            name = str(value)
            if name not in model.datainfo.names:
                return True
        elif isinstance(value, sympy.Expr):
            names = {symb.name for symb in value.free_symbols}
            if not all(name in model.datainfo.names for name in names):
                return True
    return False


def _add_zero_order_absorption(model, amount, to_comp, parameter_name, lag_time=None):
    """Add zero order absorption to a compartment. Initial estimate for absorption rate is set
    the previous rate if available, otherwise it is set to the time of first observation/2 is used.
    Disregards what is currently in the model.
    """
    mat_assign = model.statements.find_assignment(parameter_name)
    if mat_assign:
        mat_symb = mat_assign.symbol
    else:
        mat_symb = _add_parameter(
            model, parameter_name, init=_get_absorption_init(model, parameter_name)
        )
    new_dose = Infusion(amount, duration=mat_symb * 2)
    to_comp.dose = new_dose
    if lag_time and lag_time != 0:
        to_comp.lag_time = lag_time


def _add_first_order_absorption(model, dose, to_comp, lag_time=None):
    """Add first order absorption
    Disregards what is currently in the model.
    """
    odes = model.statements.ode_system
    depot = odes.add_compartment('DEPOT')
    depot.dose = dose
    mat_assign = model.statements.find_assignment('MAT')
    if mat_assign:
        mat_symb = mat_assign.symbol
    else:
        mat_symb = _add_parameter(model, 'MAT', _get_absorption_init(model, 'MAT'))
    odes.add_flow(depot, to_comp, 1 / mat_symb)
    if lag_time and lag_time != 0:
        depot.lag_time = lag_time
    return depot


def _get_absorption_init(model, param_name):
    try:
        if param_name == 'MDT':
            param_prev = model.statements.lag_time
        else:
            param_prev = _extract_params_from_symb(model.statements, param_name, model.parameters)
        return param_prev.init
    except (AttributeError, KeyError):
        pass

    time_label = model.datainfo.idv_column.name
    obs = get_observations(model)
    time_min = obs.index.get_level_values(level=time_label).min()

    if param_name == 'MDT':
        return float(time_min) / 2
    elif param_name == 'MAT':
        return float(time_min) * 2


def set_peripheral_compartments(model, n):
    """Sets the number of peripheral compartments to a specified number.

    Parameters
    ----------
    model : Model
        Pharmpy model
    n : int
        Number of transit compartments

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_peripheral_compartments(model, 2)     # doctest: +ELLIPSIS
    <...>
    >>> model.statements.ode_system
    Bolus(AMT)
    ┌───────────┐
    │PERIPHERAL1│
    └───────────┘
      ↑      │
    QP1/V QP1/VP1
      │      ↓
    ┌───────────┐       ┌──────┐
    │  CENTRAL  │──CL/V→│OUTPUT│
    └───────────┘       └──────┘
       ↑      │
    QP2/VP2 QP2/V
       │      ↓
    ┌───────────┐
    │PERIPHERAL2│
    └───────────┘

    See also
    --------
    add_peripheral_compartment
    remove_peripheral_compartment

    """
    model.statements.to_compartmental_system()
    try:
        n = _as_integer(n)
    except TypeError:
        raise TypeError(f'Number of compartments must be integer: {n}')

    per = len(model.statements.ode_system.peripheral_compartments)
    if per < n:
        for _ in range(n - per):
            add_peripheral_compartment(model)
    elif per > n:
        for _ in range(per - n):
            remove_peripheral_compartment(model)
    return model


def add_peripheral_compartment(model):
    r"""Add a peripheral distribution compartment to model

    The rate of flow from the central to the peripheral compartment
    will be parameterized as QPn / VC where VC is the volume of the central compartment.
    The rate of flow from the peripheral to the central compartment
    will be parameterized as QPn / VPn where VPn is the volumne of the added peripheral
    compartment.

    Initial estimates:

    ==  ===================================================
    n
    ==  ===================================================
    1   :math:`\mathsf{CL} = \mathsf{CL'}`, :math:`\mathsf{VC} = \mathsf{VC'}`,
        :math:`\mathsf{QP1} = \mathsf{CL'}` and :math:`\mathsf{VP1} = \mathsf{VC'} \cdot 0.05`
    2   :math:`\mathsf{QP1} = \mathsf{QP1' / 2}`, :math:`\mathsf{VP1} = \mathsf{VP1'}`,
        :math:`\mathsf{QP2} = \mathsf{QP1' / 2}` and :math:`\mathsf{VP2} = \mathsf{VP1'}`
    ==  ===================================================

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> add_peripheral_compartment(model)     # doctest: +ELLIPSIS
    <...>
    >>> model.statements.ode_system
    Bolus(AMT)
    ┌───────────┐
    │PERIPHERAL1│
    └───────────┘
      ↑      │
    QP1/V QP1/VP1
      │      ↓
    ┌───────────┐       ┌──────┐
    │  CENTRAL  │──CL/V→│OUTPUT│
    └───────────┘       └──────┘

    See also
    --------
    set_peripheral_compartment
    remove_peripheral_compartment

    """
    statements = model.statements
    statements.to_compartmental_system()
    odes = statements.ode_system
    per = odes.peripheral_compartments
    n = len(per) + 1

    central = odes.central_compartment
    output = odes.output_compartment
    elimination_rate = odes.get_flow(central, output)
    cl, vc = elimination_rate.as_numer_denom()
    if cl.is_Symbol and vc == 1:
        # If K = CL / V
        s = statements.find_assignment(cl.name)
        cl, vc = s.expression.as_numer_denom()

    if n == 1:
        if vc == 1:
            kpc = _add_parameter(model, f'KPC{n}', init=0.1)
            kcp = _add_parameter(model, f'KCP{n}', init=0.1)
            peripheral = odes.add_compartment(f'PERIPHERAL{n}')
            odes.add_flow(central, peripheral, kcp)
            odes.add_flow(peripheral, central, kpc)
        else:
            full_cl = statements.before_odes.full_expression(cl)
            full_vc = statements.before_odes.full_expression(vc)
            pop_cl_candidates = full_cl.free_symbols & set(model.parameters.symbols)
            pop_cl = pop_cl_candidates.pop()
            pop_vc_candidates = full_vc.free_symbols & set(model.parameters.symbols)
            pop_vc = pop_vc_candidates.pop()
            pop_cl_init = model.parameters[pop_cl].init
            pop_vc_init = model.parameters[pop_vc].init
            qp_init = pop_cl_init
            vp_init = pop_vc_init * 0.05
    elif n == 2:
        per1 = per[0]
        from_rate = odes.get_flow(per1, central)
        qp1, vp1 = from_rate.as_numer_denom()
        full_qp1 = statements.before_odes.full_expression(qp1)
        full_vp1 = statements.before_odes.full_expression(vp1)
        if full_vp1 == 1:
            full_qp1, full_vp1 = full_qp1.as_numer_denom()
        pop_qp1_candidates = full_qp1.free_symbols & set(model.parameters.symbols)
        pop_qp1 = pop_qp1_candidates.pop()
        pop_vp1_candidates = full_vp1.free_symbols & set(model.parameters.symbols)
        pop_vp1 = pop_vp1_candidates.pop()
        pop_qp1_init = model.parameters[pop_qp1].init
        pop_vp1_init = model.parameters[pop_vp1].init
        model.parameters[pop_qp1].init = pop_qp1_init * 0.49
        qp_init = pop_qp1_init * 0.51
        vp_init = pop_vp1_init
    else:
        qp_init = 0.1
        vp_init = 0.1

    if vc != 1:
        qp = _add_parameter(model, f'QP{n}', init=qp_init)
        vp = _add_parameter(model, f'VP{n}', init=vp_init)
        peripheral = odes.add_compartment(f'PERIPHERAL{n}')
        odes.add_flow(central, peripheral, qp / vc)
        odes.add_flow(peripheral, central, qp / vp)

    return model


def remove_peripheral_compartment(model):
    r"""Remove a peripheral distribution compartment from model

    Initial estimates:

    ==  ===================================================
    n
    ==  ===================================================
    2   :math:`\mathsf{CL} = \mathsf{CL'}`,
        :math:`\mathsf{QP1} = \mathsf{CL'}` and :math:`\mathsf{VP1} = \mathsf{VC'} \cdot 0.05`
    3   :math:`\mathsf{QP1} = (\mathsf{QP1'} + \mathsf{QP2'}) / 2`,
        :math:`\mathsf{VP1} = \mathsf{VP1'} + \mathsf{VP2'}`
    ==  ===================================================

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_peripheral_compartments(model, 2)     # doctest: +ELLIPSIS
    <...>
    >>> remove_peripheral_compartment(model)      # doctest: +ELLIPSIS
    <...>
    >>> model.statements.ode_system
    Bolus(AMT)
    ┌───────────┐
    │PERIPHERAL1│
    └───────────┘
      ↑      │
    QP1/V QP1/VP1
      │      ↓
    ┌───────────┐       ┌──────┐
    │  CENTRAL  │──CL/V→│OUTPUT│
    └───────────┘       └──────┘

    See also
    --------
    set_peripheral_compartment
    add_peripheral_compartment

    """
    statements = model.statements
    statements.to_compartmental_system()
    odes = statements.ode_system
    peripherals = odes.peripheral_compartments
    if peripherals:
        last_peripheral = peripherals[-1]
        central = odes.central_compartment
        if len(peripherals) == 1:
            output = odes.output_compartment
            elimination_rate = odes.get_flow(central, output)
            cl, vc = elimination_rate.as_numer_denom()
            from_rate = odes.get_flow(last_peripheral, central)
            qp1, vp1 = from_rate.as_numer_denom()
            full_cl = statements.before_odes.full_expression(cl)
            full_vc = statements.before_odes.full_expression(vc)
            full_qp1 = statements.before_odes.full_expression(qp1)
            full_vp1 = statements.before_odes.full_expression(vp1)
            pop_cl_candidates = full_cl.free_symbols & set(model.parameters.symbols)
            pop_cl = pop_cl_candidates.pop()
            pop_vc_candidates = full_vc.free_symbols & set(model.parameters.symbols)
            pop_vc = pop_vc_candidates.pop()
            pop_qp1_candidates = full_qp1.free_symbols & set(model.parameters.symbols)
            pop_qp1 = pop_qp1_candidates.pop()
            pop_vp1_candidates = full_vp1.free_symbols & set(model.parameters.symbols)
            pop_vp1 = pop_vp1_candidates.pop()
            pop_vc_init = model.parameters[pop_vc].init
            pop_cl_init = model.parameters[pop_cl].init
            pop_qp1_init = model.parameters[pop_qp1].init
            pop_vp1_init = model.parameters[pop_vp1].init
            new_vc_init = pop_vc_init + pop_qp1_init / pop_cl_init * pop_vp1_init
            model.parameters[pop_vc].init = new_vc_init
        elif len(peripherals) == 2:
            first_peripheral = peripherals[0]
            from1_rate = odes.get_flow(first_peripheral, central)
            qp1, vp1 = from1_rate.as_numer_denom()
            from2_rate = odes.get_flow(last_peripheral, central)
            qp2, vp2 = from2_rate.as_numer_denom()
            full_qp2 = statements.before_odes.full_expression(qp2)
            full_vp2 = statements.before_odes.full_expression(vp2)
            full_qp1 = statements.before_odes.full_expression(qp1)
            full_vp1 = statements.before_odes.full_expression(vp1)
            pop_qp2_candidates = full_qp2.free_symbols & set(model.parameters.symbols)
            pop_qp2 = pop_qp2_candidates.pop()
            pop_vp2_candidates = full_vp2.free_symbols & set(model.parameters.symbols)
            pop_vp2 = pop_vp2_candidates.pop()
            pop_qp1_candidates = full_qp1.free_symbols & set(model.parameters.symbols)
            pop_qp1 = pop_qp1_candidates.pop()
            pop_vp1_candidates = full_vp1.free_symbols & set(model.parameters.symbols)
            pop_vp1 = pop_vp1_candidates.pop()
            pop_qp2_init = model.parameters[pop_qp2].init
            pop_vp2_init = model.parameters[pop_vp2].init
            pop_qp1_init = model.parameters[pop_qp1].init
            pop_vp1_init = model.parameters[pop_vp1].init
            new_qp1_init = (pop_qp1_init + pop_qp2_init) / 2
            new_vp1_init = pop_vp1_init + pop_vp2_init
            model.parameters[pop_qp1].init = new_qp1_init
            model.parameters[pop_vp1].init = new_vp1_init

        symbols = odes.get_flow(central, last_peripheral).free_symbols
        symbols |= odes.get_flow(last_peripheral, central).free_symbols
        odes.remove_compartment(last_peripheral)
        model.statements.remove_symbol_definitions(symbols, odes)
        remove_unused_parameters_and_rvs(model)
    return model


def set_ode_solver(model, solver):
    """Sets ODE solver to use for model

    Recognized solvers and their corresponding NONMEM advans:

    +----------------------------+------------------+
    | Solver                     | NONMEM ADVAN     |
    +============================+==================+
    | CVODES                     | ADVAN14          |
    +----------------------------+------------------+
    | DGEAR                      | ADVAN8           |
    +----------------------------+------------------+
    | DVERK                      | ADVAN6           |
    +----------------------------+------------------+
    | IDA                        | ADVAN15          |
    +----------------------------+------------------+
    | GL (general linear)        | ADVAN5           |
    +----------------------------+------------------+
    | GL_REAL (real eigenvalues) | ADVAN7           |
    +----------------------------+------------------+
    | LSODA                      | ADVAN13          |
    +----------------------------+------------------+
    | LSODI                      | ADVAN9           |
    +----------------------------+------------------+

    Parameters
    ----------
    model : Model
        Pharmpy model
    solver : str
        Solver to use or None for no preference

    Return
    ------
    Model
        Reference to same model

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> set_ode_solver(model, 'LSODA')    # doctest: +ELLIPSIS
    <...>

    """
    odes = model.statements.ode_system
    odes.solver = solver.upper()
    return model
