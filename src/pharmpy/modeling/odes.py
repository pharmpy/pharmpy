"""
:meta private:
"""
from typing import Set

from pharmpy.deps import sympy
from pharmpy.internals.expr.subs import subs
from pharmpy.model import (
    Assignment,
    Bolus,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    EstimationSteps,
    Infusion,
    Model,
    ModelError,
    Parameter,
    Parameters,
    Statements,
)
from pharmpy.modeling.help_functions import _as_integer

from .common import remove_unused_parameters_and_rvs, rename_symbols
from .data import get_observations
from .expressions import create_symbol
from .parameters import (
    add_population_parameter,
    fix_parameters,
    fix_parameters_to,
    set_initial_estimates,
    set_upper_bounds,
    unfix_parameters,
)


def _extract_params_from_symb(
    statements: Statements, symbol_name: str, pset: Parameters
) -> Parameter:
    terms = {
        symb.name
        for symb in statements.before_odes.full_expression(sympy.Symbol(symbol_name)).free_symbols
    }
    theta_name = terms.intersection(pset.names).pop()
    return pset[theta_name]


def add_individual_parameter(model: Model, name: str):
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
    KA = POP_KA

    """
    _add_parameter(model, name)
    return model


def _add_parameter(model: Model, name: str, init: float = 0.1):
    pops = create_symbol(model, f'POP_{name}')
    add_population_parameter(model, pops.name, init, lower=0)
    symb = create_symbol(model, name)
    ass = Assignment(symb, pops)
    model.statements = ass + model.statements
    return symb


def set_first_order_elimination(model: Model):
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
        rename_symbols(model, {'POP_CLMM': 'POP_CL'})
        ind = model.statements.find_assignment_index('CLMM')
        assert ind is not None
        clmm_assignment = model.statements[ind]
        assert isinstance(clmm_assignment, Assignment)
        cl_ass = Assignment(sympy.Symbol('CL'), clmm_assignment.expression)
        statements = model.statements[0:ind] + cl_ass + model.statements[ind + 1 :]
        odes = statements.ode_system
        assert odes is not None
        odes = odes.to_compartmental_system()
        central = odes.central_compartment
        output = odes.output_compartment
        v = sympy.Symbol('V')
        rate = odes.get_flow(central, output)
        assert rate is not None
        if v not in rate.free_symbols:
            v = sympy.Symbol('VC')
        cb = CompartmentalSystemBuilder(odes)
        cb.remove_flow(central, output)
        cb.add_flow(central, output, sympy.Symbol('CL') / v)
        statements = statements.before_odes + CompartmentalSystem(cb) + statements.after_odes
        model.statements = statements.remove_symbol_definitions(
            {sympy.Symbol('KM')}, statements.ode_system
        )
        remove_unused_parameters_and_rvs(model)
    elif has_mixed_mm_fo_elimination(model):
        odes = model.statements.ode_system
        assert odes is not None
        odes = odes.to_compartmental_system()
        central = odes.central_compartment
        output = odes.output_compartment
        v = sympy.Symbol('V')
        rate = odes.get_flow(central, output)
        assert rate is not None
        if v not in rate.free_symbols:
            v = sympy.Symbol('VC')
        cb = CompartmentalSystemBuilder(odes)
        cb.remove_flow(central, output)
        cb.add_flow(central, output, sympy.Symbol('CL') / v)
        model.statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
        model.statements = model.statements.remove_symbol_definitions(
            {sympy.Symbol('KM'), sympy.Symbol('CLMM')}, model.statements.ode_system
        )
        remove_unused_parameters_and_rvs(model)
    return model


def set_zero_order_elimination(model: Model):
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
        fix_parameters(model, 'POP_KM')
    elif has_mixed_mm_fo_elimination(model):
        fix_parameters(model, 'POP_KM')
        odes = model.statements.ode_system
        assert odes is not None
        odes = odes.to_compartmental_system()
        central = odes.central_compartment
        output = odes.output_compartment
        rate = odes.get_flow(central, output)
        rate = subs(rate, {'CL': 0})
        cb = CompartmentalSystemBuilder(odes)
        cb.remove_flow(central, output)
        cb.add_flow(central, output, rate)
        model.statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
        model.statements = model.statements.remove_symbol_definitions(
            {sympy.Symbol('CL')}, model.statements.ode_system
        )
        remove_unused_parameters_and_rvs(model)
    else:
        _do_michaelis_menten_elimination(model)
        obs = get_observations(model)
        init = obs.min() / 100  # 1% of smallest observation
        fix_parameters_to(model, {'POP_KM': init})
    return model


def has_michaelis_menten_elimination(model: Model):
    """Check if the model describes Michaelis-Menten elimination

    This function relies on heuristics and will not be able to detect all
    possible ways of coding the Michaelis-Menten elimination.

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
    odes = model.statements.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    odes = odes.to_compartmental_system()
    output = odes.output_compartment
    central = odes.central_compartment
    rate = odes.get_flow(central, output)
    assert rate is not None
    is_nonlinear = odes.t in rate.free_symbols
    is_zero_order = 'POP_KM' in model.parameters and model.parameters['POP_KM'].fix
    could_be_mixed = sympy.Symbol('CL') in rate.free_symbols
    return is_nonlinear and not is_zero_order and not could_be_mixed


def has_zero_order_elimination(model: Model):
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
    odes = model.statements.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    odes = odes.to_compartmental_system()
    output = odes.output_compartment
    central = odes.central_compartment
    rate = odes.get_flow(central, output)
    assert rate is not None
    is_nonlinear = odes.t in rate.free_symbols
    is_zero_order = 'POP_KM' in model.parameters and model.parameters['POP_KM'].fix
    could_be_mixed = sympy.Symbol('CL') in rate.free_symbols
    return is_nonlinear and is_zero_order and not could_be_mixed


def has_mixed_mm_fo_elimination(model: Model):
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
    odes = model.statements.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    odes = odes.to_compartmental_system()
    output = odes.output_compartment
    central = odes.central_compartment
    rate = odes.get_flow(central, output)
    assert rate is not None
    is_nonlinear = odes.t in rate.free_symbols
    is_zero_order = 'POP_KM' in model.parameters and model.parameters['POP_KM'].fix
    could_be_mixed = sympy.Symbol('CL') in rate.free_symbols
    return is_nonlinear and not is_zero_order and could_be_mixed


def has_first_order_elimination(model: Model):
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
    odes = model.statements.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    odes = odes.to_compartmental_system()
    output = odes.output_compartment
    central = odes.central_compartment
    rate = odes.get_flow(central, output)
    assert rate is not None
    is_nonlinear = odes.t in rate.free_symbols
    return not is_nonlinear


def set_michaelis_menten_elimination(model: Model):
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
        unfix_parameters(model, 'POP_KM')  # model.parameters['POP_KM'].fix = False
    elif has_mixed_mm_fo_elimination(model):
        odes = model.statements.ode_system
        assert odes is not None
        odes = odes.to_compartmental_system()
        central = odes.central_compartment
        output = odes.output_compartment
        rate = odes.get_flow(central, output)
        rate = subs(rate, {'CL': 0})
        cb = CompartmentalSystemBuilder(odes)
        cb.remove_flow(central, output)
        cb.add_flow(central, output, rate)
        statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
        model.statements = statements.remove_symbol_definitions(
            {sympy.Symbol('CL')}, statements.ode_system
        )
        remove_unused_parameters_and_rvs(model)
    else:
        _do_michaelis_menten_elimination(model)
    return model


def set_mixed_mm_fo_elimination(model: Model):
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
        unfix_parameters(model, 'POP_KM')
        odes = model.statements.ode_system
        assert odes is not None
        odes = odes.to_compartmental_system()
        central = odes.central_compartment
        output = odes.output_compartment
        add_individual_parameter(model, 'CL')
        rate = odes.get_flow(central, output)
        assert rate is not None
        cb = CompartmentalSystemBuilder(odes)
        cb.remove_flow(central, output)
        v = sympy.Symbol('V')
        if v not in rate.free_symbols:
            v = sympy.Symbol('VC')
        cb.add_flow(central, output, sympy.Symbol('CL') / v + rate)
        model.statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
    else:
        _do_michaelis_menten_elimination(model, combined=True)
    return model


def _do_michaelis_menten_elimination(model: Model, combined: bool = False):
    model.statements = model.statements.to_compartmental_system()
    sset = model.statements
    odes = sset.ode_system
    assert isinstance(odes, CompartmentalSystem)
    central = odes.central_compartment
    output = odes.output_compartment
    old_rate = odes.get_flow(central, output)
    assert old_rate is not None
    numer, denom = old_rate.as_numer_denom()

    km_init, clmm_init = _get_mm_inits(model, numer, combined)

    km = _add_parameter(model, 'KM', init=km_init)
    set_upper_bounds(model, {'POP_KM': 20 * get_observations(model).max()})

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
            if model.statements.find_assignment('CL'):
                assignment = model.statements.find_assignment('CL')
                assert assignment is not None
                cl = assignment.symbol
            else:
                cl = _add_parameter(model, 'CL', clmm_init)
        else:
            cl = 0
        if model.statements.find_assignment('VC'):
            assignment = sset.find_assignment('VC')
            assert assignment is not None
            vc = assignment.symbol
        else:
            vc = _add_parameter(model, 'VC')  # FIXME: decide better initial estimate
        if not combined and model.statements.find_assignment('CL'):
            _rename_parameter(model, 'CL', 'CLMM')
            assignment = model.statements.find_assignment('CLMM')
            assert assignment is not None
            clmm = assignment.symbol
        else:
            clmm = _add_parameter(model, 'CLMM', init=clmm_init)

    amount = sympy.Function(central.amount.name)(sympy.Symbol('t'))
    rate = (clmm * km / (km + amount / vc) + cl) / vc
    cb = CompartmentalSystemBuilder(odes)
    cb.add_flow(central, output, rate)
    model.statements = (
        model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
    )
    model.statements = model.statements.remove_symbol_definitions(
        numer.free_symbols, model.statements.ode_system
    )
    remove_unused_parameters_and_rvs(model)
    return model


def _rename_parameter(model: Model, old_name, new_name):
    a = model.statements.find_assignment(old_name)
    assert a is not None
    d = {}
    for s in a.rhs_symbols:
        if s in model.parameters:
            old_par = s
            d[model.parameters[s].symbol] = f'POP_{new_name}'
            new_par = sympy.Symbol(f'POP_{new_name}')
            model.statements = model.statements.subs({old_par: new_par})
            break
    for s in a.rhs_symbols:
        iivs = model.random_variables.iiv
        if s.name in iivs.names:
            cov = iivs.covariance_matrix
            ind = iivs.names.index(s.name)
            pars = [e for e in cov[ind, :] if e.is_Symbol]
            diag = cov[ind, ind]
            d[diag] = f'IIV_{new_name}'
            for p in pars:
                if p != diag:
                    if p.name.startswith('IIV'):
                        # FIXME: in some cases, parameters are read as symbols, this is
                        #  a workaround
                        try:
                            symb = p.symbol
                        except AttributeError:
                            symb = p
                        d[symb] = p.name.replace(f'IIV_{old_name}', f'IIV_{new_name}')
            model.random_variables = model.random_variables.subs(d)
            break
    new = []
    for p in model.parameters:
        if p.symbol in d:
            newparam = Parameter(
                name=d[p.symbol], init=p.init, upper=p.upper, lower=p.lower, fix=p.fix
            )
        else:
            newparam = p
        new.append(newparam)
    model.parameters = Parameters(new)
    model.statements = model.statements.subs({old_name: new_name})


def _get_mm_inits(model: Model, rate_numer, combined):
    pset, sset = model.parameters, model.statements
    parameter = _extract_params_from_symb(sset, rate_numer.name, pset)
    assert parameter is not None
    clmm_init = parameter.init

    if combined:
        clmm_init /= 2

    dv_max = get_observations(model).max()
    km_init = dv_max * 2

    return km_init, clmm_init


def set_transit_compartments(model: Model, n: int, keep_depot: bool = True):
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
    statements = model.statements.to_compartmental_system()
    odes = statements.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    cs = odes.to_compartmental_system()
    transits = cs.find_transit_compartments(statements)
    try:
        n = _as_integer(n)
    except ValueError:
        raise ValueError(f'Number of compartments must be integer: {n}')

    # Handle keep_depot option
    depot = cs.find_depot(statements)
    mdt_init = None
    mdt_assign = None
    if not keep_depot and depot:
        central = cs.central_compartment
        rate = cs.get_flow(depot, central)
        assert rate is not None
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
        inflows = cs.get_compartment_inflows(depot)
        cb = CompartmentalSystemBuilder(cs)
        if len(inflows) == 1:
            innode, inflow = inflows[0]
            cb.add_flow(innode, central, inflow)
        else:
            cb.set_dose(central, depot.dose)
        if statements.find_assignment('MAT'):
            _rename_parameter(model, 'MAT', 'MDT')
            statements = model.statements
            mdt_assign = statements.find_assignment('MDT')
        cb.remove_compartment(depot)
        statements = statements.before_odes + CompartmentalSystem(cb) + statements.after_odes
        statements = statements.remove_symbol_definitions(rate.free_symbols, statements.ode_system)
        if mdt_assign:
            statements = mdt_assign + statements
        model.statements = statements
        remove_unused_parameters_and_rvs(model)
        odes = statements.ode_system
        assert odes is not None
        cs = odes.to_compartmental_system()

    if len(transits) == n:
        return model
    elif len(transits) == 0:
        if mdt_assign:
            mdt_symb = mdt_assign.symbol
        else:
            if mdt_init is not None:
                init = mdt_init
            else:
                init = _get_absorption_init(model, 'MDT')
            mdt_symb = _add_parameter(model, 'MDT', init=init)
        rate = n / mdt_symb
        dosing_comp = cs.dosing_compartment
        comp = dosing_comp
        cb = CompartmentalSystemBuilder(cs)
        while n > 0:
            new_comp = Compartment(f'TRANSIT{n}')
            cb.add_compartment(new_comp)
            n -= 1
            cb.add_flow(new_comp, comp, rate)
            comp = new_comp
        cb.move_dose(dosing_comp, comp)
        model.statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
    elif len(transits) > n:
        nremove = len(transits) - n
        removed_symbols = set()
        remaining = set(transits)
        trans, destination, flow = _find_last_transit(cs, remaining)

        cb = CompartmentalSystemBuilder(cs)

        while nremove > 0:
            from_comp, from_flow = cs.get_compartment_inflows(trans)[0]
            cb.add_flow(from_comp, destination, from_flow)
            cb.remove_compartment(trans)
            remaining.remove(trans)
            removed_symbols |= flow.free_symbols
            trans = from_comp
            flow = from_flow
            nremove -= 1

        if n == 0:
            dose = cs.dosing_compartment.dose
            cb.set_dose(destination, dose)

        model.statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
        _update_numerators(model)
        model.statements = model.statements.remove_symbol_definitions(
            removed_symbols, model.statements.ode_system
        )
        remove_unused_parameters_and_rvs(model)
    else:
        nadd = n - len(transits)
        last, destination, rate = _find_last_transit(cs, set(transits))
        cb = CompartmentalSystemBuilder(cs)
        cb.remove_flow(last, destination)
        while nadd > 0:
            new_comp = Compartment(f'TRANSIT{n - nadd + 1}')
            cb.add_compartment(new_comp)
            cb.add_flow(last, new_comp, rate)
            if rate.is_Symbol:
                ass = statements.find_assignment(rate.name)
                if ass is not None:
                    rate = ass.expression
            last = new_comp
            nadd -= 1
        cb.add_flow(last, destination, rate)
        model.statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
        _update_numerators(model)
    return model


def _find_last_transit(odes: CompartmentalSystem, transits: Set[Compartment]):
    for trans in transits:
        destination, flow = odes.get_compartment_outflows(trans)[0]
        if destination not in transits:
            return trans, destination, flow

    raise ValueError('Could not find last transit')


def _update_numerators(model: Model):
    # update numerators for transit compartment rates
    statements = model.statements
    odes = statements.ode_system
    assert odes is not None
    odes = odes.to_compartmental_system()
    transits = odes.find_transit_compartments(statements)
    new_numerator = sympy.Integer(len(transits))
    cb = CompartmentalSystemBuilder(odes)
    for comp in transits:
        to_comp, rate = odes.get_compartment_outflows(comp)[0]
        numer, denom = rate.as_numer_denom()
        if numer.is_Integer and numer != new_numerator:
            new_rate = new_numerator / denom
            cb.add_flow(comp, to_comp, new_rate)
        elif numer.is_Symbol:
            ass = statements.find_assignment(numer.name)
            if ass is not None:
                ass_numer, ass_denom = ass.expression.as_numer_denom()
                if ass_numer.is_Integer and ass_numer != new_numerator:
                    new_rate = new_numerator / ass_denom
                    statements = statements.reassign(numer, new_rate)
    model.statements = statements.before_odes + CompartmentalSystem(cb) + statements.after_odes


def add_lag_time(model: Model):
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
    odes = model.statements.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    odes = odes.to_compartmental_system()
    dosing_comp = odes.dosing_compartment
    old_lag_time = dosing_comp.lag_time
    mdt_symb = _add_parameter(model, 'MDT', init=_get_absorption_init(model, 'MDT'))
    cb = CompartmentalSystemBuilder(odes)
    cb.set_lag_time(dosing_comp, mdt_symb)
    model.statements = (
        model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
    )
    if old_lag_time:
        model.statements = model.statements.remove_symbol_definitions(
            old_lag_time.free_symbols, odes
        )
        remove_unused_parameters_and_rvs(model)
    return model


def remove_lag_time(model: Model):
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
    odes = model.statements.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    odes = odes.to_compartmental_system()
    dosing_comp = odes.dosing_compartment
    lag_time = dosing_comp.lag_time
    if lag_time:
        symbols = lag_time.free_symbols
        cb = CompartmentalSystemBuilder(odes)
        cb.set_lag_time(dosing_comp, sympy.Integer(0))
        model.statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
        model.statements = model.statements.remove_symbol_definitions(
            symbols, model.statements.ode_system
        )
        remove_unused_parameters_and_rvs(model)
    return model


def set_zero_order_absorption(model: Model):
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
    odes = statements.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    odes = odes.to_compartmental_system()
    _disallow_infusion(model, odes)
    depot = odes.find_depot(statements)

    dose_comp = odes.dosing_compartment
    symbols = dose_comp.free_symbols
    dose = dose_comp.dose
    lag_time = dose_comp.lag_time
    if depot:
        to_comp, _ = odes.get_compartment_outflows(depot)[0]
        ka = odes.get_flow(depot, odes.central_compartment)
        assert ka is not None
        cb = CompartmentalSystemBuilder(odes)
        cb.remove_compartment(depot)
        cb.set_dose(to_comp, dose)
        statements = statements.before_odes + CompartmentalSystem(cb) + statements.after_odes
        symbols = ka.free_symbols
    else:
        to_comp = dose_comp

    new_statements = statements.remove_symbol_definitions(symbols, statements.ode_system)
    mat_idx = statements.find_assignment_index('MAT')
    if mat_idx is not None:
        mat_assign = statements[mat_idx]
        new_statements = new_statements[0:mat_idx] + mat_assign + new_statements[mat_idx:]

    model.statements = new_statements

    remove_unused_parameters_and_rvs(model)
    if not has_zero_order_absorption(model):
        odes = model.statements.ode_system
        assert odes is not None
        cs = odes.to_compartmental_system()
        _add_zero_order_absorption(model, dose.amount, cs.dosing_compartment, 'MAT', lag_time)
    return model


def set_first_order_absorption(model: Model):
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
    odes = statements.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    cs = odes.to_compartmental_system()
    depot = cs.find_depot(statements)

    dose_comp = cs.dosing_compartment
    amount = dose_comp.dose.amount
    symbols = dose_comp.free_symbols
    lag_time = dose_comp.lag_time
    cb = CompartmentalSystemBuilder(cs)
    if depot and depot == dose_comp:
        dose_comp = cb.set_dose(dose_comp, Bolus(dose_comp.dose.amount))
        dose_comp = cb.set_lag_time(dose_comp, sympy.Integer(0))
    if not depot:
        dose_comp = cb.set_dose(dose_comp, Bolus(amount))
    statements = statements.before_odes + CompartmentalSystem(cb) + statements.after_odes

    new_statements = statements.remove_symbol_definitions(symbols, statements.ode_system)
    mat_idx = statements.find_assignment_index('MAT')
    if mat_idx is not None:
        mat_assign = statements[mat_idx]
        new_statements = new_statements[0:mat_idx] + mat_assign + new_statements[mat_idx:]

    model.statements = new_statements

    remove_unused_parameters_and_rvs(model)
    if not depot:
        _add_first_order_absorption(model, Bolus(amount), dose_comp, lag_time)
    return model


def set_bolus_absorption(model: Model):
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
    statements = model.statements.to_compartmental_system()
    cs = statements.ode_system
    if cs is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    depot = cs.find_depot(statements)
    if depot:
        to_comp, _ = cs.get_compartment_outflows(depot)[0]
        cb = CompartmentalSystemBuilder(cs)
        cb.set_dose(to_comp, depot.dose)
        ka = cs.get_flow(depot, cs.central_compartment)
        cb.remove_compartment(depot)
        symbols = ka.free_symbols
        statements = statements.before_odes + CompartmentalSystem(cb) + statements.after_odes
        model.statements = statements.remove_symbol_definitions(symbols, statements.ode_system)
        remove_unused_parameters_and_rvs(model)
    if has_zero_order_absorption(model):
        dose_comp = cs.dosing_compartment
        old_symbols = dose_comp.free_symbols
        cb = CompartmentalSystemBuilder(cs)
        new_dose = Bolus(dose_comp.dose.amount)
        cb.set_dose(dose_comp, new_dose)
        unneeded_symbols = old_symbols - new_dose.free_symbols
        statements = statements.before_odes + CompartmentalSystem(cb) + statements.after_odes
        model.statements = statements.remove_symbol_definitions(
            unneeded_symbols, statements.ode_system
        )
        remove_unused_parameters_and_rvs(model)
    return model


def set_seq_zo_fo_absorption(model: Model):
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
    odes = statements.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    cs = odes.to_compartmental_system()
    model.statements = statements.before_odes + cs + statements.after_odes
    _disallow_infusion(model, cs)
    depot = cs.find_depot(statements)

    dose_comp = cs.dosing_compartment
    have_ZO = has_zero_order_absorption(model)
    if depot and not have_ZO:
        _add_zero_order_absorption(model, dose_comp.amount, depot, 'MDT')
    elif not depot and have_ZO:
        _add_first_order_absorption(model, dose_comp.dose, dose_comp)
    elif not depot and not have_ZO:
        amount = dose_comp.dose.amount
        depot = _add_first_order_absorption(model, Bolus(amount), dose_comp)
        _add_zero_order_absorption(model, amount, depot, 'MDT')
    return model


def _disallow_infusion(model, odes):
    dose_comp = odes.dosing_compartment
    if isinstance(dose_comp.dose, Infusion):
        if dose_comp.dose.rate is not None:
            ex = dose_comp.dose.rate
        else:
            ex = dose_comp.dose.duration
        assert ex is not None

        for s in ex.free_symbols:
            if s.name in model.datainfo.names:
                raise ModelError("Model already has an infusion given in the dataset")


def has_zero_order_absorption(model: Model):
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
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    cs = odes.to_compartmental_system()
    dosing = cs.dosing_compartment
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
            assert value is not None
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
    cb = CompartmentalSystemBuilder(model.statements.ode_system)
    cb.set_dose(to_comp, new_dose)
    if lag_time is not None and lag_time != 0:
        cb.set_lag_time(model.statements.ode_system.dosing_compartment, lag_time)
    model.statements = (
        model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
    )


def _add_first_order_absorption(model, dose, to_comp, lag_time=None):
    """Add first order absorption
    Disregards what is currently in the model.
    """
    odes = model.statements.ode_system
    cb = CompartmentalSystemBuilder(odes)
    depot = Compartment('DEPOT', dose, sympy.Integer(0) if lag_time is None else lag_time)
    cb.add_compartment(depot)
    to_comp = cb.set_dose(to_comp, None)
    to_comp = cb.set_lag_time(to_comp, sympy.Integer(0))

    mat_assign = model.statements.find_assignment('MAT')
    if mat_assign:
        mat_symb = mat_assign.symbol
    else:
        mat_symb = _add_parameter(model, 'MAT', _get_absorption_init(model, 'MAT'))
    cb.add_flow(depot, to_comp, 1 / mat_symb)
    model.statements = (
        model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
    )
    return depot


def _get_absorption_init(model, param_name) -> float:
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
    time = obs.index.get_level_values(level=time_label)
    time_min = time[time != 0].min()

    if param_name == 'MDT':
        return float(time_min) / 2
    elif param_name == 'MAT':
        return float(time_min) * 2

    raise NotImplementedError('param_name must be MDT or MAT')


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
    odes = model.statements.ode_system.to_compartmental_system()

    try:
        n = _as_integer(n)
    except TypeError:
        raise TypeError(f'Number of compartments must be integer: {n}')

    per = len(odes.peripheral_compartments)
    if per < n:
        for _ in range(n - per):
            add_peripheral_compartment(model)
    elif per > n:
        for _ in range(per - n):
            remove_peripheral_compartment(model)
    return model


def add_peripheral_compartment(model: Model):
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
    2   :math:`\mathsf{QP1} = \mathsf{QP1' * 0.1}`, :math:`\mathsf{VP1} = \mathsf{VP1'}`,
        :math:`\mathsf{QP2} = \mathsf{QP1' * 0.9}` and :math:`\mathsf{VP2} = \mathsf{VP1'}`
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
    odes = statements.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    cs = odes.to_compartmental_system()
    per = cs.peripheral_compartments
    n = len(per) + 1

    central = cs.central_compartment
    output = cs.output_compartment
    elimination_rate = cs.get_flow(central, output)
    assert elimination_rate is not None
    cl, vc = elimination_rate.as_numer_denom()
    if cl.is_Symbol and vc == 1:
        # If K = CL / V
        s = statements.find_assignment(cl.name)
        assert s is not None
        cl, vc = s.expression.as_numer_denom()
    # Heuristic to handle the MM case
    if vc.is_Mul:
        vc = vc.args[0]

    cb = CompartmentalSystemBuilder(cs)

    # NOTE Only used if vc != 1
    qp_init = 0.1
    vp_init = 0.1

    if n == 1:
        if vc == 1:
            kpc = _add_parameter(model, f'KPC{n}', init=0.1)
            kcp = _add_parameter(model, f'KCP{n}', init=0.1)
            peripheral = cb.add_compartment(f'PERIPHERAL{n}')
            cb.add_flow(central, peripheral, kcp)
            cb.add_flow(peripheral, central, kpc)
        else:
            # Heurstic to handle the Mixed MM-FO case
            if cl.is_Add:
                cl1 = cl.args[0]
                if cl1.is_Mul:
                    cl = cl1.args[0]
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
        from_rate = cs.get_flow(per1, central)
        assert from_rate is not None
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
        set_initial_estimates(model, {pop_qp1.name: pop_qp1_init * 0.10})
        qp_init = pop_qp1_init * 0.90
        vp_init = pop_vp1_init

    if vc != 1:
        qp = _add_parameter(model, f'QP{n}', init=qp_init)
        vp = _add_parameter(model, f'VP{n}', init=vp_init)
        peripheral = Compartment(f'PERIPHERAL{n}')
        cb.add_compartment(peripheral)
        cb.add_flow(central, peripheral, qp / vc)
        cb.add_flow(peripheral, central, qp / vp)

    model.statements = Statements(
        model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
    )

    return model


def remove_peripheral_compartment(model: Model):
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
    odes = statements.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    cs = odes.to_compartmental_system()
    peripherals = cs.peripheral_compartments
    if peripherals:
        last_peripheral = peripherals[-1]
        central = cs.central_compartment
        if len(peripherals) == 1:
            output = cs.output_compartment
            elimination_rate = cs.get_flow(central, output)
            assert elimination_rate is not None
            cl, vc = elimination_rate.as_numer_denom()
            from_rate = cs.get_flow(last_peripheral, central)
            assert from_rate is not None
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
            set_initial_estimates(model, {pop_vc.name: new_vc_init})
        elif len(peripherals) == 2:
            first_peripheral = peripherals[0]
            from1_rate = cs.get_flow(first_peripheral, central)
            assert from1_rate is not None
            qp1, vp1 = from1_rate.as_numer_denom()
            from2_rate = cs.get_flow(last_peripheral, central)
            assert from2_rate is not None
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
            set_initial_estimates(model, {pop_qp1.name: new_qp1_init, pop_vp1.name: new_vp1_init})

        rate1 = cs.get_flow(central, last_peripheral)
        assert rate1 is not None
        rate2 = cs.get_flow(last_peripheral, central)
        assert rate2 is not None
        symbols = rate1.free_symbols | rate2.free_symbols
        cb = CompartmentalSystemBuilder(cs)
        cb.remove_compartment(last_peripheral)
        model.statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
        model.statements = model.statements.remove_symbol_definitions(
            symbols, model.statements.ode_system
        )
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
    new_steps = []
    for step in model.estimation_steps:
        new = step.derive(solver=solver)
        new_steps.append(new)
    model.estimation_steps = EstimationSteps(new_steps)
    return model


def find_clearance_parameters(model: Model):
    """Find clearance parameters in model

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    list
        A list of clearance parameters

    Examples
    --------
     >>> from pharmpy.modeling import *
     >>> model = load_example_model("pheno")
     >>> find_clearance_parameters(model)
     [CL]
    """
    cls = set()
    sset = model.statements
    odes = sset.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    t = odes.t
    rate_list = _find_rate(sset)
    for rate in rate_list:
        if isinstance(rate, sympy.Symbol):
            assignment = sset.find_assignment(rate)
            assert assignment is not None
            rate = assignment.expression
        a, b = map(lambda x: x.free_symbols, rate.as_numer_denom())
        if b:
            clearance_symbols = a - b - {t}
            for clearance in clearance_symbols:
                clearance = _find_real_symbol(sset, clearance)
                cls.add(clearance)
    return sorted(cls, key=str)


def find_volume_parameters(model: Model):
    """Find volume parameters in model

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    list
        A list of volume parameters

    Examples
    --------
     >>> from pharmpy.modeling import *
     >>> model = load_example_model("pheno")
     >>> find_volume_parameters(model)
     [V]
    """
    vcs = set()
    sset = model.statements
    odes = sset.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    t = odes.t
    rate_list = _find_rate(sset)
    for rate in rate_list:
        if isinstance(rate, sympy.Symbol):
            assignment = sset.find_assignment(rate)
            assert assignment is not None
            rate = assignment.expression
        rate = sympy.cancel(rate)
        a, b = map(lambda x: x.free_symbols, rate.as_numer_denom())
        volume_symbols = b - a - {t}
        for volume in volume_symbols:
            volume = _find_real_symbol(sset, volume)
            vcs.add(volume)
    return sorted(vcs, key=str)


def _find_rate(sset: Statements):
    rate_list = []
    odes = sset.ode_system
    assert odes is not None
    odes = odes.to_compartmental_system()
    central = odes.central_compartment
    output = odes.output_compartment
    elimination_rate = odes.get_flow(central, output)
    rate_list.append(elimination_rate)
    for periph in odes.peripheral_compartments:
        rate1 = odes.get_flow(central, periph)
        rate_list.append(rate1)
        rate2 = odes.get_flow(periph, central)
        rate_list.append(rate2)
    return rate_list


def _find_real_symbol(sset, symbol):
    assign = sset.find_assignment(symbol)
    if len(assign.rhs_symbols) == 1:
        dep_assigns = _get_dependent_assignments(sset, assign)
        for dep_assign in dep_assigns:
            symbol = dep_assign.symbol
    return symbol


def _get_dependent_assignments(sset, assignment):
    """Finds dependent assignments one layer deep"""
    return list(
        filter(
            None,  # NOTE filters out falsy values
            (sset.find_assignment(symb) for symb in assignment.expression.free_symbols),
        )
    )
