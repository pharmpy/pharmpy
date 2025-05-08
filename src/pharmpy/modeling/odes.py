"""
:meta private:
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Literal, Optional, Union

from pharmpy.basic import Expr, Matrix, TExpr
from pharmpy.deps import sympy
from pharmpy.internals.unicode import bracket
from pharmpy.model import (
    Assignment,
    Bolus,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    ExecutionSteps,
    Infusion,
    Model,
    ModelError,
    Parameter,
    Parameters,
    Statements,
    get_and_check_odes,
    output,
)
from pharmpy.modeling.help_functions import _as_integer

from .common import remove_unused_parameters_and_rvs, rename_symbols
from .data import add_time_after_dose, get_observations
from .expressions import create_symbol, is_real
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
        for symb in statements.before_odes.full_expression(Expr.symbol(symbol_name)).free_symbols
    }
    theta_name = terms.intersection(pset.names).pop()
    return pset[theta_name]


def _find_noncov_theta(model, paramsymb, full=False):
    # Deterministic function to find the main theta in the expression for parasymb
    # Set full to True if already providing a full expression
    # Find subexpression with as few thetas and covariates as possible
    # Stop if one theta found with no covs.
    thetas = ()
    if full:
        start_expr = paramsymb
    else:
        start_expr = model.statements.before_odes.full_expression(paramsymb)

    exprs = [start_expr]
    all_popparams = set(model.parameters.symbols)
    all_covs = set(Expr.symbol(name) for name in model.datainfo.names)

    while exprs:
        nthetas = float("Inf")
        ncovs = float("Inf")
        next_expr = None
        for expr in exprs:
            if isinstance(expr, tuple):
                # This is for Piecewise pairs
                symbs = set()
                for e in expr:
                    symbs |= e.free_symbols
            else:
                symbs = expr.free_symbols
            curthetas = symbs & all_popparams
            ncurthetas = len(curthetas)
            curcovs = symbs & all_covs
            ncurcovs = len(curcovs)
            if ncurthetas != 0 and (
                ncurthetas < nthetas or (ncurthetas == nthetas and ncurcovs < ncovs)
            ):
                next_expr = expr
                thetas = curthetas
                nthetas = ncurthetas
                ncovs = ncurcovs

        if next_expr is None:
            break
        else:
            if nthetas == 1 and ncovs == 0:
                return next(iter(thetas))
            else:
                if isinstance(next_expr, tuple):
                    exprs = next_expr
                else:
                    exprs = next_expr.args
    raise ValueError(f"Could not find theta connected to {paramsymb}")


def add_individual_parameter(model: Model, name: str, init: float = 0.1, lower: float = 0.0):
    """Add an individual or pk parameter to a model

    Parameters
    ----------
    model : Model
        Pharmpy model
    name : str
        Name of individual/pk parameter
    init : float
        Initial estimate of the population parameter
    lower : float
        Lower bound for the population parameter

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = add_individual_parameter(model, "KA")
    >>> model.statements.find_assignment("KA")
    KA = POP_KA

    """
    model, _ = _add_parameter(model, name, init=init, lower=lower)
    model = model.update_source()
    return model


def _add_parameter(
    model: Model, name: str, init: float = 0.1, lower: float = 0, upper: Union[float, None] = None
):
    pops = create_symbol(model, f'POP_{name}')
    model = add_population_parameter(model, pops.name, init, lower=lower, upper=upper)
    symb = create_symbol(model, name)
    ass = Assignment.create(symb, pops)
    model = model.replace(statements=ass + model.statements)
    return model, symb


def set_first_order_elimination(model: Model):
    """Sets elimination to first order

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_first_order_elimination(model)
    >>> model.statements.ode_system
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘

    See also
    --------
    set_zero_order_elimination
    set_michaelis_menten_elimination

    """
    if has_first_order_elimination(model):
        pass
    elif has_zero_order_elimination(model) or has_michaelis_menten_elimination(model):
        model = rename_symbols(model, {'POP_CLMM': 'POP_CL', 'IIV_CLMM': 'IIV_CL'})
        ind = model.statements.find_assignment_index('CLMM')
        assert ind is not None
        clmm_assignment = model.statements[ind]
        assert isinstance(clmm_assignment, Assignment)
        cl_ass = Assignment.create(Expr.symbol('CL'), clmm_assignment.expression)
        statements = model.statements[0:ind] + cl_ass + model.statements[ind + 1 :]
        odes = statements.ode_system
        assert isinstance(odes, CompartmentalSystem)
        central = odes.central_compartment
        v = Expr.symbol('V')
        rate = odes.get_flow(central, output)
        if v not in rate.free_symbols:
            # take first parameter that starts with 'V' and is no longer than 2 characters
            v = [
                idx
                for idx in list(map(lambda x: str(x), rate.free_symbols))
                if idx[0] == 'V' and len(idx) <= 2
            ]
            v = Expr.symbol(v[0])
        cb = CompartmentalSystemBuilder(odes)
        cb.remove_flow(central, output)
        cb.add_flow(central, output, Expr.symbol('CL') / v)
        new_odes = CompartmentalSystem(cb)
        statements = statements.before_odes + new_odes + statements.after_odes
        statements = statements.remove_symbol_definitions({Expr.symbol('KM')}, new_odes)
        model = model.replace(statements=statements)
        model = remove_unused_parameters_and_rvs(model)
    elif has_mixed_mm_fo_elimination(model):
        odes = model.statements.ode_system
        assert odes is not None
        central = odes.central_compartment
        v = Expr.symbol('V')
        rate = odes.get_flow(central, output)
        if v not in rate.free_symbols:
            v = [
                idx
                for idx in list(map(lambda x: str(x), rate.free_symbols))
                if idx[0] == 'V' and len(idx) <= 2
            ]
            v = Expr.symbol(v[0])
        cb = CompartmentalSystemBuilder(odes)
        cb.remove_flow(central, output)
        cb.add_flow(central, output, Expr.symbol('CL') / v)
        new_odes = CompartmentalSystem(cb)
        statements = model.statements.before_odes + new_odes + model.statements.after_odes
        statements = statements.remove_symbol_definitions(
            {Expr.symbol('KM'), Expr.symbol('CLMM')}, new_odes
        )
        model = model.replace(statements=statements)
        model = remove_unused_parameters_and_rvs(model)
    return model


def add_bioavailability(model: Model, add_parameter: bool = True, logit_transform: bool = False):
    """Add bioavailability statement for the first dose compartment of the model.
    Can be added as a new parameter or otherwise it will be set to 1. If added as a parameter,
    a logit transformation can also be applied.

    Parameters
    ----------
    model : Model
        Pharmpy model
    add_parameter : bool
        Add new parameter representing bioavailability or not
    logit_transform : bool
        Logit transform the added bioavailability parameter.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = add_bioavailability(model)

    See also
    --------
    remove_bioavailability

    """
    odes = get_and_check_odes(model)

    dose_comp = odes.dosing_compartments[0]
    bio = dose_comp.bioavailability

    if bio.is_number():
        # Bio not defined
        if add_parameter:
            model, bio_symb = _add_parameter(model, 'BIO', init=float(bio), upper=1.0)
            if logit_transform:
                model = model.replace(
                    statements=model.statements.reassign(
                        bio_symb, (Expr.symbol("POP_BIO") / (1 - Expr.symbol("POP_BIO"))).log()
                    )
                )
                f_ass = Assignment(Expr.symbol('F_BIO'), 1 / (1 + (-bio_symb).exp()))
            else:
                f_ass = Assignment(Expr.symbol('F_BIO'), bio_symb)

            new_before_odes = model.statements.before_odes + f_ass

        else:
            # Add as a number
            bio_ass = Assignment(Expr.symbol("BIO"), Expr.integer(1))
            f_ass = Assignment(Expr.symbol("F_BIO"), bio_ass.symbol)
            new_before_odes = bio_ass + model.statements.before_odes + f_ass

        # Add statement to code
        cb = CompartmentalSystemBuilder(odes)
        cb.set_bioavailability(dose_comp, f_ass.symbol)

        model = model.replace(
            statements=(new_before_odes + CompartmentalSystem(cb) + model.statements.after_odes)
        )

    else:
        # BIO already defined, leave it alone?
        pass

    return model.update_source()


def remove_bioavailability(model: Model):
    """Remove bioavailability from the first dose compartment of model.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = remove_bioavailability(model)

    See also
    --------
    set_bioavailability
    """
    odes = get_and_check_odes(model)
    dosing_comp = odes.dosing_compartments[0]
    bio = dosing_comp.bioavailability
    if bio:
        symbols = bio.free_symbols
        cb = CompartmentalSystemBuilder(odes)
        cb.set_bioavailability(dosing_comp, Expr.integer(1))
        new_odes = CompartmentalSystem(cb)
        statements = model.statements.before_odes + new_odes + model.statements.after_odes
        statements = statements.remove_symbol_definitions(symbols, new_odes)
        model = model.replace(statements=statements)
        model = remove_unused_parameters_and_rvs(model)
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
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_zero_order_elimination(model)
    >>> model.statements.ode_system
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────┐
    │CENTRAL│──CLMM*KM/(V*(KM + A_CENTRAL(t)/V))→
    └───────┘

    See also
    --------
    set_first_order_elimination
    set_michaelis_menten_elimination

    """
    if has_zero_order_elimination(model):
        pass
    elif has_michaelis_menten_elimination(model):
        model = fix_parameters(model, 'POP_KM')
    elif has_mixed_mm_fo_elimination(model):
        model = fix_parameters(model, 'POP_KM')
        odes = get_and_check_odes(model)
        central = odes.central_compartment
        rate = odes.get_flow(central, output)
        rate = rate.subs({'CL': 0})
        cb = CompartmentalSystemBuilder(odes)
        cb.remove_flow(central, output)
        cb.add_flow(central, output, rate)
        new_odes = CompartmentalSystem(cb)
        statements = model.statements.before_odes + new_odes + model.statements.after_odes
        statements = statements.remove_symbol_definitions({Expr.symbol('CL')}, new_odes)
        model = model.replace(statements=statements)
        model = remove_unused_parameters_and_rvs(model)
    else:
        model = _do_michaelis_menten_elimination(model)
        if model.dataset is not None:
            obs = get_observations(model)
            init = obs.min() / 100  # 1% of smallest observation
        else:
            init = 0.01
        if init < 0:
            init = 0.01
        model = fix_parameters_to(model, {'POP_KM': init})
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
    >>> model = set_michaelis_menten_elimination(model)
    >>> has_michaelis_menten_elimination(model)
    True
    """
    odes = model.statements.ode_system
    if odes is None:
        return False
    central = odes.central_compartment
    rate = odes.get_flow(central, output)
    is_nonlinear = odes.t in rate.free_symbols
    is_zero_order = 'POP_KM' in model.parameters and model.parameters['POP_KM'].fix
    could_be_mixed = Expr.symbol('CL') in rate.free_symbols
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
    >>> model = set_zero_order_elimination(model)
    >>> has_zero_order_elimination(model)
    True
    """
    odes = model.statements.ode_system
    if odes is None:
        return False
    central = odes.central_compartment
    rate = odes.get_flow(central, output)
    is_nonlinear = odes.t in rate.free_symbols
    is_zero_order = 'POP_KM' in model.parameters and model.parameters['POP_KM'].fix
    could_be_mixed = Expr.symbol('CL') in rate.free_symbols
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
    >>> model = set_mixed_mm_fo_elimination(model)
    >>> has_mixed_mm_fo_elimination(model)
    True
    """
    odes = model.statements.ode_system
    if odes is None:
        return False
    central = odes.central_compartment
    rate = odes.get_flow(central, output)
    is_nonlinear = odes.t in rate.free_symbols
    is_zero_order = 'POP_KM' in model.parameters and model.parameters['POP_KM'].fix
    could_be_mixed = Expr.symbol('CL') in rate.free_symbols
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
        return False
    central = odes.central_compartment
    rate = odes.get_flow(central, output)
    is_nonlinear = odes.t in rate.free_symbols
    return not is_nonlinear


def set_michaelis_menten_elimination(model: Model):
    """Sets elimination to Michaelis-Menten.

    Note that the parametrization is not the usual, but is instead using a CLMM parameter.

    Initial estimate for CLMM is set to CL and KM is set to :math:`max(DV)/2`.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_michaelis_menten_elimination(model)
    >>> model.statements.ode_system
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────┐
    │CENTRAL│──CLMM*KM/(V*(KM + A_CENTRAL(t)/V))→
    └───────┘

    See also
    --------
    set_first_order_elimination
    set_zero_order_elimination

    """
    if has_michaelis_menten_elimination(model):
        pass
    elif has_zero_order_elimination(model):
        model = unfix_parameters(model, 'POP_KM')
    elif has_mixed_mm_fo_elimination(model):
        odes = get_and_check_odes(model)
        central = odes.central_compartment
        rate = odes.get_flow(central, output)
        rate = rate.subs({'CL': 0})
        cb = CompartmentalSystemBuilder(odes)
        cb.remove_flow(central, output)
        cb.add_flow(central, output, rate)
        new_odes = CompartmentalSystem(cb)
        statements = model.statements.before_odes + new_odes + model.statements.after_odes
        statements = statements.remove_symbol_definitions({Expr.symbol('CL')}, new_odes)
        model = model.replace(statements=statements)
        model = remove_unused_parameters_and_rvs(model)
    else:
        model = _do_michaelis_menten_elimination(model)
    return model


def set_mixed_mm_fo_elimination(model: Model):
    """Sets elimination to mixed Michaelis-Menten and first order.

    Initial estimate for CLMM is set to CL/2 and KM is set to :math:`max(DV)/2`.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_mixed_mm_fo_elimination(model)
    >>> model.statements.ode_system
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────┐
    │CENTRAL│──(CL + CLMM*KM/(KM + A_CENTRAL(t)/V))/V→
    └───────┘

    See also
    --------
    set_first_order_elimination
    set_zero_order_elimination
    set_michaelis_menten_elimination

    """
    if has_mixed_mm_fo_elimination(model):
        pass
    elif has_michaelis_menten_elimination(model) or has_zero_order_elimination(model):
        model = unfix_parameters(model, 'POP_KM')
        odes = get_and_check_odes(model)
        central = odes.central_compartment
        model = add_individual_parameter(model, 'CL')
        rate = odes.get_flow(central, output)
        assert rate is not None
        cb = CompartmentalSystemBuilder(odes)
        cb.remove_flow(central, output)
        v = Expr.symbol('V')
        if v not in rate.free_symbols:
            v = Expr.symbol('VC')
        cb.add_flow(central, output, Expr.symbol('CL') / v + rate)
        statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
        model = model.replace(statements=statements)
        model = model.update_source()
    else:
        model = _do_michaelis_menten_elimination(model, combined=True)
    return model


def _do_michaelis_menten_elimination(model: Model, combined: bool = False):
    sset = model.statements
    odes = sset.ode_system
    assert isinstance(odes, CompartmentalSystem)
    central = odes.central_compartment
    old_rate = odes.get_flow(central, output)
    assert old_rate is not None
    numer, denom = old_rate.as_numer_denom()

    km_init, clmm_init = _get_mm_inits(model, numer, combined)

    model, km = _add_parameter(model, 'KM', init=km_init)
    if model.dataset is not None and 'idv' in model.datainfo.types and 'dv' in model.datainfo.types:
        maxobs = get_observations(model).max()
    else:
        maxobs = 1.0
    model = set_upper_bounds(model, {'POP_KM': 1.5 * maxobs})

    if denom != 1:
        if combined:
            cl = numer
            model, clmm = _add_parameter(model, 'CLMM', init=clmm_init)
        else:
            model = _rename_parameter(model, 'CL', 'CLMM')
            clmm = Expr.symbol('CLMM')
            cl = 0
        vc = denom
    else:
        if combined:
            if model.statements.find_assignment('CL'):
                assignment = model.statements.find_assignment('CL')
                assert assignment is not None
                cl = assignment.symbol
            else:
                model, cl = _add_parameter(model, 'CL', clmm_init)
        else:
            cl = 0
        if model.statements.find_assignment('VC'):
            assignment = sset.find_assignment('VC')
            assert assignment is not None
            vc = assignment.symbol
        else:
            model, vc = _add_parameter(model, 'VC')  # FIXME: decide better initial estimate
        if not combined and model.statements.find_assignment('CL'):
            model = _rename_parameter(model, 'CL', 'CLMM')
            assignment = model.statements.find_assignment('CLMM')
            assert assignment is not None
            clmm = assignment.symbol
        else:
            model, clmm = _add_parameter(model, 'CLMM', init=clmm_init)

    rate = (clmm * km / (km + central.amount / vc) + cl) / vc
    cb = CompartmentalSystemBuilder(odes)
    cb.add_flow(central, output, rate)
    new_odes = CompartmentalSystem(cb)
    statements = model.statements.before_odes + new_odes + model.statements.after_odes
    statements = statements.remove_symbol_definitions(numer.free_symbols, new_odes)
    model = model.replace(statements=statements)
    model = remove_unused_parameters_and_rvs(model)
    return model


def _rename_parameter(model: Model, old_name, new_name):
    statements = model.statements
    rvs = model.random_variables
    a = statements.find_assignment(old_name)
    assert a is not None
    d = {}
    for s in a.rhs_symbols:
        if s in model.parameters:
            old_par = s
            d[model.parameters[s].symbol] = f'POP_{new_name}'
            new_par = Expr.symbol(f'POP_{new_name}')
            statements = statements.subs({old_par: new_par})
            break
    for s in a.rhs_symbols:
        iivs = rvs.iiv
        if s.name in iivs.names:
            cov = iivs.covariance_matrix
            ind = iivs.names.index(s.name)
            pars = []
            for i in range(cov.rows):
                e = cov[ind, i]
                if e.is_symbol():
                    pars.append(e)
            diag = cov[ind, ind]
            d[diag] = f'IIV_{new_name}'
            for p in pars:
                if p != diag:
                    if p.name.startswith('IIV'):
                        d[p] = p.name.replace(f'IIV_{old_name}', f'IIV_{new_name}')
            rvs = rvs.subs(d)
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
    parameters = Parameters.create(new)
    statements = statements.subs({old_name: new_name})
    model = model.replace(statements=statements, parameters=parameters, random_variables=rvs)
    return model


def _get_mm_inits(model: Model, rate_numer, combined):
    pset, sset = model.parameters, model.statements
    parameter = _extract_params_from_symb(sset, rate_numer.name, pset)
    assert parameter is not None
    clmm_init = parameter.init

    if combined:
        clmm_init /= 2

    if model.dataset is not None and 'idv' in model.datainfo.types and 'dv' in model.datainfo.types:
        dv_max = get_observations(model).max()
    else:
        dv_max = 1.0
    km_init = dv_max / 2
    # FIXME: Cap initial estimate, this is NONMEM specific and should be handled more generally
    #  (https://github.com/pharmpy/pharmpy/issues/1395)
    if km_init >= 10**6:
        km_init = 5 * 10**5

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
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_transit_compartments(model, 3)
    >>> model.statements.ode_system
    Bolus(AMT, admid=1) → TRANSIT1
    ┌────────┐      ┌────────┐      ┌────────┐      ┌───────┐
    │TRANSIT1│──K12→│TRANSIT2│──K23→│TRANSIT3│──K34→│CENTRAL│──K40→
    └────────┘      └────────┘      └────────┘      └───────┘

    See also
    --------
    add_lag_time

    """
    statements = model.statements
    cs = get_and_check_odes(model)
    transits = cs.find_transit_compartments(statements)
    try:
        n = _as_integer(n)
    except ValueError:
        raise ValueError(f'Number of compartments must be integer: {n}')

    model = remove_lag_time(model)

    # Handle keep_depot option
    depot = cs.find_depot(statements)
    mdt_init = None
    mdt_assign = None
    if not keep_depot and depot:
        central = cs.central_compartment
        rate = cs.get_flow(depot, central)
        assert rate is not None
        if not rate.is_symbol():
            num, den = rate.as_numer_denom()
            if num == 1 and den.is_symbol():
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
            cb.set_dose(central, depot.doses[0])
        if statements.find_assignment('MAT'):
            if not statements.find_assignment('MDT'):
                model = _rename_parameter(model, 'MAT', 'MDT')
            statements = model.statements
            mdt_assign = statements.find_assignment('MDT')
        cb.remove_compartment(depot)
        new_odes = CompartmentalSystem(cb)
        statements = statements.before_odes + new_odes + statements.after_odes
        statements = statements.remove_symbol_definitions(rate.free_symbols, new_odes)
        if mdt_assign and not statements.find_assignment('MDT'):
            statements = mdt_assign + statements
        model = model.replace(statements=statements)
        model = remove_unused_parameters_and_rvs(model)
        odes = statements.ode_system
        assert odes is not None
        # Since update_source() is used after removing the depot and statements are immutable, we need to
        # reset to get the correct rate names
        cs = get_and_check_odes(model)

    if len(transits) == n:
        return model
    elif n == 1 and has_instantaneous_absorption(model):
        raise ValueError(
            "Cannot set the number of transits to 1 for model with instantaneous "
            "absorption. The resulting model cannot be distinguished from first order absorption"
        )
    elif len(transits) == 0:
        if mdt_assign:
            mdt_symb = mdt_assign.symbol
        else:
            if mdt_init is not None:
                init = mdt_init
            else:
                init = _get_absorption_init(model, 'MDT')
            model, mdt_symb = _add_parameter(model, 'MDT', init=init)
        rate = n / mdt_symb
        dosing_comp = cs.dosing_compartments[0]
        comp = dosing_comp
        cb = CompartmentalSystemBuilder(cs)
        while n > 0:
            new_comp = Compartment.create(f'TRANSIT{n}')
            cb.add_compartment(new_comp)
            n -= 1
            cb.add_flow(new_comp, comp, rate)
            comp = new_comp
        comp = cb.set_bioavailability(comp, dosing_comp.bioavailability)
        dosing_comp = cb.set_bioavailability(dosing_comp, Expr.integer(1))

        if len(dosing_comp.doses) == 1 and dosing_comp.doses[0].admid == 2:
            dosing_comp = cb.set_dose(dosing_comp, dosing_comp.doses[0].replace(admid=1))

        cb.move_dose(dosing_comp, comp, admid=1)

        if len(dosing_comp.doses) == 1:
            cb.set_dose(comp, dosing_comp.doses)
        else:
            dose = _sorted_doses(dosing_comp, model)[0]
            cb.add_dose(comp, dose)
            cb.set_dose(dosing_comp, _sorted_doses(dosing_comp, model)[1:])

        statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
        model = model.replace(statements=statements)
        model = model.update_source()
    elif len(transits) > n:
        nremove = len(transits) - n
        removed_symbols = set()
        remaining = set(transits)
        trans, destination, flow = _find_last_transit(cs, remaining)

        cb = CompartmentalSystemBuilder(cs)

        while nremove > 0:
            inflows = cs.get_compartment_inflows(trans)
            if len(inflows) == 0:
                from_comp = None
                from_flow = None
            else:
                from_comp, from_flow = inflows[0]
                cb.add_flow(from_comp, destination, from_flow)
            cb.remove_compartment(trans)
            remaining.remove(trans)
            removed_symbols |= flow.free_symbols
            if from_comp is not None and from_flow is not None:
                trans = from_comp
                flow = from_flow
            nremove -= 1

        if n == 0:
            dose = cs.dosing_compartments[0].doses[0]
            cb.set_dose(destination, dose)

        statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
        model = model.replace(statements=statements)
        model = _update_numerators(model)
        statements = model.statements.remove_symbol_definitions(
            removed_symbols, get_and_check_odes(model)
        )
        model = model.replace(statements=statements)
        model = remove_unused_parameters_and_rvs(model)
    else:
        nadd = n - len(transits)
        last, destination, rate = _find_last_transit(cs, set(transits))
        cb = CompartmentalSystemBuilder(cs)
        cb.remove_flow(last, destination)
        while nadd > 0:
            new_comp = Compartment.create(f'TRANSIT{n - nadd + 1}')
            cb.add_compartment(new_comp)
            cb.add_flow(last, new_comp, rate)
            if rate.is_symbol():
                ass = statements.find_assignment(rate.name)
                if ass is not None:
                    rate = ass.expression
            last = new_comp
            nadd -= 1
        cb.add_flow(last, destination, rate)
        statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
        model = model.replace(statements=statements)
        model = _update_numerators(model)
        model = model.update_source()
    return model


def _find_last_transit(odes: CompartmentalSystem, transits: set[Compartment]):
    for trans in transits:
        destination, flow = odes.get_compartment_outflows(trans)[0]
        assert isinstance(destination, Compartment)
        if destination not in transits:
            return trans, destination, flow

    raise ValueError('Could not find last transit')


def _update_numerators(model: Model):
    # update numerators for transit compartment rates
    statements = model.statements
    odes = statements.ode_system
    assert odes is not None
    transits = odes.find_transit_compartments(statements)
    new_numerator = Expr.integer(len(transits))
    cb = CompartmentalSystemBuilder(odes)
    for comp in transits:
        to_comp, rate = odes.get_compartment_outflows(comp)[0]
        numer, denom = rate.as_numer_denom()
        if numer.is_integer() and numer != new_numerator:
            new_rate = new_numerator / denom
            cb.add_flow(comp, to_comp, new_rate)
        elif numer.is_symbol():
            ass = statements.find_assignment(numer.name)
            if ass is not None:
                ass_numer, ass_denom = ass.expression.as_numer_denom()
                if ass_numer.is_integer() and ass_numer != new_numerator:
                    new_rate = new_numerator / ass_denom
                    statements = statements.reassign(numer, new_rate)
    model = model.replace(
        statements=statements.before_odes + CompartmentalSystem(cb) + statements.after_odes
    )
    return model


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
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = add_lag_time(model)

    See also
    --------
    set_transit_compartments
    remove_lag_time

    """
    odes = get_and_check_odes(model)
    dosing_comp = odes.dosing_compartments[0]
    old_lag_time = dosing_comp.lag_time
    model, mdt_symb = _add_parameter(model, 'MDT', init=_get_absorption_init(model, 'MDT'))
    cb = CompartmentalSystemBuilder(odes)
    dosing_comp = cb.set_lag_time(dosing_comp, mdt_symb)

    # FIXME: Very temporary until new zo absorption logic is implemented
    if len(dosing_comp.doses) > 1:
        cb.set_lag_time(dosing_comp, Expr.symbol("lag_time"))
        doses = _sorted_doses(dosing_comp, model)
        assert len(doses) > 0
        oral_admid = doses[0].admid
        admid = Expr.symbol("ADMID")
        model = model.replace(
            statements=(
                model.statements.before_odes
                + Assignment.create(
                    Expr.symbol("lag_time"),
                    Expr.piecewise((mdt_symb, sympy.Eq(admid, oral_admid)), (0, sympy.true)),
                )
                + CompartmentalSystem(cb)
                + model.statements.after_odes
            )
        )
    else:
        model = model.replace(
            statements=(
                model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
            )
        )
    if old_lag_time:
        odes = get_and_check_odes(model)
        model = model.replace(
            statements=model.statements.remove_symbol_definitions(old_lag_time.free_symbols, odes)
        )
        model = remove_unused_parameters_and_rvs(model)
    else:
        model = model.update_source()
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
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = remove_lag_time(model)

    See also
    --------
    set_transit_compartments
    add_lag_time


    """
    odes = get_and_check_odes(model)
    dosing_comp = odes.dosing_compartments[0]
    lag_time = dosing_comp.lag_time
    if lag_time:
        symbols = lag_time.free_symbols
        cb = CompartmentalSystemBuilder(odes)
        cb.set_lag_time(dosing_comp, Expr.integer(0))
        new_odes = CompartmentalSystem(cb)
        statements = model.statements.before_odes + new_odes + model.statements.after_odes
        statements = statements.remove_symbol_definitions(symbols, new_odes)
        model = model.replace(statements=statements)
        model = remove_unused_parameters_and_rvs(model)
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
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_zero_order_absorption(model)
    >>> model.statements.ode_system
    Infusion(AMT, admid=1, duration=D1) → CENTRAL
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘

    See also
    --------
    set_instantaneous_absorption
    set_first_order_absorption

    """
    statements = model.statements
    odes = get_and_check_odes(model)
    _disallow_infusion(model, odes)

    if has_zero_order_absorption(model) and not has_seq_zo_fo_absorption(model):
        pass
    else:
        _disallow_infusion(model, odes)
        depot = odes.find_depot(statements)

        dose_comp = odes.dosing_compartments[0]
        symbols = dose_comp.free_symbols
        dose = _sorted_doses(dose_comp, model)[0]
        lag_time = dose_comp.lag_time
        if depot:
            to_comp, _ = odes.get_compartment_outflows(depot)[0]
            assert isinstance(to_comp, Compartment)
            ka = odes.get_flow(depot, odes.central_compartment)
            assert ka is not None
            cb = CompartmentalSystemBuilder(odes)
            cb.remove_compartment(depot)
            to_comp = cb.add_dose(to_comp, dose)
            to_comp = cb.set_lag_time(to_comp, depot.lag_time)
            cb.set_bioavailability(to_comp, depot.bioavailability)
            statements = statements.before_odes + CompartmentalSystem(cb) + statements.after_odes
            symbols = ka.free_symbols
        odes = statements.ode_system
        assert odes is not None
        new_statements = statements.remove_symbol_definitions(symbols, odes)
        mat_idx = statements.find_assignment_index('MAT')
        if mat_idx is not None:
            # FIXME : Causes issue if mat_assign statement is dependent on previously
            # removed parameters/statements
            mat_assign = statements[mat_idx]
            new_statements = new_statements[0:mat_idx] + mat_assign + new_statements[mat_idx:]
        model = model.replace(statements=new_statements)

        model = remove_unused_parameters_and_rvs(model)
        if not has_zero_order_absorption(model):
            odes = model.statements.ode_system
            assert odes is not None
            model = _add_zero_order_absorption(
                model, dose, odes.dosing_compartments[0], 'MAT', lag_time
            )
            model = model.update_source()
        # FIXME : Very temporary until new zo absorption logic is implemented
        odes = get_and_check_odes(model)
        if lag_time != 0 and len(odes.dosing_compartments[0].doses) > 1:
            model = remove_lag_time(model)
            model = add_lag_time(model)
    return model


def set_first_order_absorption(model: Model):
    """Set or change to first order absorption rate.

    Initial estimate for absorption rate is set to
    the previous rate if available, otherwise it is set to the time of first observation/2.

    If multiple doses is set to the affected compartment, currently only iv+oral
    doses (one of each) is supported

    Parameters
    ----------
    model : Model
        Model to set or change to use first order absorption rate

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_first_order_absorption(model)
    >>> model.statements.ode_system
    Bolus(AMT, admid=1) → DEPOT
    ┌─────┐     ┌───────┐
    │DEPOT│──KA→│CENTRAL│──CL/V→
    └─────┘     └───────┘

    See also
    --------
    set_instantaneous_absorption
    set_zero_order_absorption

    """
    statements = model.statements
    cs = get_and_check_odes(model)

    remove_dose = False
    dose_admid = None
    if has_first_order_absorption(model) and not has_seq_zo_fo_absorption(model):
        pass
    else:
        depot = cs.find_depot(statements)
        dose_comp = cs.dosing_compartments[0]

        amount = dose_comp.doses[0].amount
        symbols = dose_comp.free_symbols
        lag_time = dose_comp.lag_time
        bio = dose_comp.bioavailability
        cb = CompartmentalSystemBuilder(cs)
        if depot and depot == dose_comp:
            dose = dose_comp.doses[0]
            dose_comp = cb.remove_dose(dose_comp, admid=dose.admid)
            dose_comp = cb.set_dose(dose_comp, Bolus(dose.amount))
            dose_comp = cb.set_lag_time(dose_comp, Expr.integer(0))
        if not depot:
            # TODO : Add another way of removing dependencies
            dose_admid = _sorted_doses(dose_comp, model)[0].admid
            if len(dose_comp.doses) == 1:
                dose_comp = cb.set_dose(dose_comp, Bolus(amount))
                remove_dose = True
            else:
                dose_comp = cb.set_dose(dose_comp, _sorted_doses(dose_comp, model)[1:])

        new_odes = CompartmentalSystem(cb)
        statements = statements.before_odes + new_odes + statements.after_odes
        new_statements = statements.remove_symbol_definitions(symbols, new_odes)
        mat_idx = statements.find_assignment_index('MAT')
        if mat_idx is not None:
            mat_assign = statements[mat_idx]
            new_statements = new_statements[0:mat_idx] + mat_assign + new_statements[mat_idx:]

        model = model.replace(statements=new_statements)

        model = remove_unused_parameters_and_rvs(model)
        if dose_admid is not None:
            # The new dose is created here
            model, _ = _add_first_order_absorption(
                model,
                Bolus(amount, admid=dose_admid),
                dose_comp,
                lag_time,
                bio,
                remove_dose=remove_dose,
            )
            model = model.update_source()
    return model


def set_instantaneous_absorption(model: Model):
    """Set or change to instantaneous absorption rate.

    Currently lagtime together with instantaneous absorption is not supported.

    Parameters
    ----------
    model : Model
        Model to set or change absorption rate

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_instantaneous_absorption(model)
    >>> model.statements.ode_system
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘

    See also
    --------
    set_zero_order_absorption
    set_first_order_absorption

    """
    statements = model.statements
    cs = get_and_check_odes(model)
    if has_instantaneous_absorption(model):
        pass
    else:
        depot = cs.find_depot(statements)
        if depot:
            to_comp, _ = cs.get_compartment_outflows(depot)[0]
            assert isinstance(to_comp, Compartment)
            cb = CompartmentalSystemBuilder(cs)
            cb.set_dose(to_comp, depot.doses[0])
            ka = cs.get_flow(depot, cs.central_compartment)
            cb.remove_compartment(depot)
            symbols = ka.free_symbols
            new_odes = CompartmentalSystem(cb)
            statements = statements.before_odes + new_odes + statements.after_odes
            model = model.replace(
                statements=statements.remove_symbol_definitions(symbols, new_odes)
            )
            model = remove_unused_parameters_and_rvs(model)
        if has_zero_order_absorption(model):
            dose_comp = cs.dosing_compartments[0]
            old_symbols = dose_comp.free_symbols
            cb = CompartmentalSystemBuilder(cs)
            new_dose = Bolus(_sorted_doses(dose_comp, model)[0].amount)
            if len(dose_comp.doses) > 1:
                cb.set_dose(dose_comp, (new_dose,) + _sorted_doses(dose_comp, model)[1:])
            else:
                cb.set_dose(dose_comp, new_dose)
            unneeded_symbols = old_symbols - new_dose.free_symbols
            new_odes = CompartmentalSystem(cb)
            statements = statements.before_odes + new_odes + statements.after_odes
            model = model.replace(
                statements=statements.remove_symbol_definitions(unneeded_symbols, new_odes)
            )
            model = remove_unused_parameters_and_rvs(model)
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
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_seq_zo_fo_absorption(model)
    >>> model.statements.ode_system
    Infusion(AMT, admid=1, duration=D1) → DEPOT
    ┌─────┐     ┌───────┐
    │DEPOT│──KA→│CENTRAL│──CL/V→
    └─────┘     └───────┘

    See also
    --------
    set_instantaneous_absorption
    set_zero_order_absorption
    set_first_order_absorption

    """
    statements = model.statements
    cs = get_and_check_odes(model)

    if has_seq_zo_fo_absorption(model):
        pass
    else:
        _disallow_infusion(model, cs)
        depot = cs.find_depot(statements)

        dose_comp = cs.dosing_compartments[0]
        have_ZO = has_zero_order_absorption(model)
        if depot and not have_ZO:
            model = _add_zero_order_absorption(model, dose_comp.doses[0], depot, 'MDT')
        elif not depot and have_ZO:
            if len(dose_comp.doses) == 1:
                fo_dose = dose_comp.doses[0]
                remove_dose = True
            else:
                fo_dose = _sorted_doses(dose_comp, model)[0]
                cb = CompartmentalSystemBuilder(model.statements.ode_system)
                dose_comp = cb.set_dose(dose_comp, _sorted_doses(dose_comp, model)[1:])
                model = model.replace(
                    statements=model.statements.before_odes
                    + CompartmentalSystem(cb)
                    + model.statements.after_odes
                )
                remove_dose = False
            model, _ = _add_first_order_absorption(
                model, fo_dose, dose_comp, remove_dose=remove_dose
            )
        elif not depot and not have_ZO:
            model = set_first_order_absorption(model)
            odes = get_and_check_odes(model)
            depot = odes.find_depot(model.statements)
            model = _add_zero_order_absorption(
                model, Bolus(dose_comp.doses[0].amount), depot, 'MDT'
            )
        model = model.update_source()
    return model


def has_weibull_absorption(model: Model) -> bool:
    """Check if ode system describes a weibull type absorption

    .. warning::
        This function is still under development.

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    -------
        Bool : True if model has weibull type absorption

    """

    odes = model.statements.ode_system
    if odes is None:
        return False
    central = odes.central_compartment
    depot = odes.find_depot(model.statements)
    if depot is None:
        return False

    rate = odes.get_flow(depot, central)

    beta = sympy.Wild("beta")
    alpha = sympy.Wild("alpha")
    tad = sympy.Wild("tad")
    ka = (beta / alpha) * (tad / alpha) ** (beta - 1)  # pyright: ignore [reportOperatorIssue]
    matches = sympy.sympify(rate).match(ka) is not None
    return matches


def set_weibull_absorption(model: Model) -> Model:
    """Set or change to Weibull type absorption

    Initial estimate for absorption rate is set to??

    If multiple doses is set to the affected compartment, currently only iv+oral
    doses (one of each) is supported

    Weibull absorption cannot be used together with lag time and transit compartments.

    Assumes that absorption of one does is done when next dose is given.

    .. warning::
        This function is still under development.

    Parameters
    ----------
    model : Model
        Model to set or change to use Weibull absorption rate

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_weibull_absorption(model)
    >>> model.statements.ode_system
    Bolus(AMT, admid=1) → DEPOT
    ┌─────┐     ┌───────┐
    │DEPOT│──KA→│CENTRAL│──CL/V→
    └─────┘     └───────┘

    See also
    --------
    set_zero_order_absorption
    set_first_order_absorption

    """
    if has_weibull_absorption(model):
        return model

    # FIXME: Using names of parameters here. Could find parameters structurally instead.
    if 'POP_MAT' in model.parameters.names:
        init_mat = model.parameters['POP_MAT'].init
    else:
        init_mat = None
    if 'POP_MDT' in model.parameters.names:
        init_mdt = model.parameters['POP_MDT'].init
    else:
        init_mdt = None

    if init_mat is not None and init_mdt is None:
        init_beta = 1.0
        init_alpha = init_mat
    elif init_mdt is not None and init_mat is None:
        init_beta = 1.0
        init_alpha = init_mdt
    else:
        if init_mat is None:
            init_mat = _get_absorption_init(model, "MAT")
        init_beta = 1.5
        init_alpha = init_mat / math.gamma(1.0 + 1.0 / init_beta)

    model = remove_lag_time(model)
    model = set_transit_compartments(model, 0, keep_depot=True)
    model = set_first_order_absorption(model)
    model = add_time_after_dose(model)
    tad = model.datainfo.descriptorix['time after dose'][0].symbol

    alpha = Expr.symbol("ALPHA")
    beta = Expr.symbol("BETA")
    model = add_individual_parameter(model, alpha.name, init=init_alpha, lower=0.0)
    model = add_individual_parameter(model, beta.name, init=init_beta, lower=0.0)

    ka = (beta / alpha) * (tad / alpha) ** (beta - 1)

    odes = get_and_check_odes(model)

    cb = CompartmentalSystemBuilder(odes)
    central = odes.central_compartment
    assert isinstance(central, Compartment)
    depot = odes.find_depot(model.statements)
    assert isinstance(depot, Compartment)
    oldrate = odes.get_flow(depot, central)
    cb.add_flow(depot, central, ka)
    odes = CompartmentalSystem(cb)

    statements = model.statements.before_odes + odes + model.statements.after_odes
    statements = statements.remove_symbol_definitions(oldrate.free_symbols, odes)
    model = model.replace(statements=statements)

    model = remove_unused_parameters_and_rvs(model)
    return model


def _disallow_infusion(model, odes):
    dose_comp = odes.dosing_compartments[0]
    if isinstance(dose_comp.doses[0], Infusion):
        if dose_comp.doses[0].rate is not None:
            ex = dose_comp.doses[0].rate
        else:
            ex = dose_comp.doses[0].duration
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
    bool
        Whether the model has zero order absorption or not

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> has_zero_order_absorption(model)
    False

    """
    odes = model.statements.ode_system
    if odes is None:
        return False
    dosing = odes.dosing_compartments[0]
    dose = dosing.doses[0]
    return _dose_zo(model, dose)


def _dose_zo(model, dose):
    if isinstance(dose, Infusion):
        if dose.rate is None:
            value = dose.duration
        else:
            value = dose.rate
        if isinstance(value, str) or isinstance(value, Expr) and value.is_symbol():
            name = str(value)
            if name not in model.datainfo.names:
                return True
        elif isinstance(value, Expr):
            assert value is not None
            names = {symb.name for symb in value.free_symbols}
            if not all(name in model.datainfo.names for name in names):
                return True
    return False


def _sorted_doses(comp, model):
    """Return doses to compartment where oral doses are located first"""
    doses = comp.doses
    assert len(doses) > 0
    if len(doses) > 1:
        return tuple(sorted(doses, key=lambda d: _dose_zo(model, d), reverse=True))
    else:
        return doses


def has_first_order_absorption(model: Model):
    """Check if ode system describes a first order absorption

    Currently defined as the central compartment having a unidirectional input
    flow from another compartment (such as depot or transit)

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    -------
        Bool : True if model has first order absorption

    """

    odes = model.statements.ode_system
    if odes is None:
        return False
    dosing = odes.dosing_compartments[0]
    central = odes.central_compartment
    if dosing == central:
        return False

    in_flow = [flow[0] for flow in odes.get_compartment_inflows(central) if flow[0] != output]
    out_flow = [flow[0] for flow in odes.get_compartment_outflows(central) if flow[0] != output]

    unidirectional_flow = [uni_flow for uni_flow in in_flow if uni_flow not in out_flow]

    if len(unidirectional_flow) == 1:
        return True
    return False


def has_instantaneous_absorption(model: Model):
    """Check if ode system describes a instantaneous absorption

    Defined as being a instantaneous dose directly into the central compartment

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    -------
        Bool : True if model has instantaneous absorption
    """
    odes = model.statements.ode_system
    if odes is None:
        return False
    dosing = odes.dosing_compartments[0]
    central = odes.central_compartment
    if dosing != central:
        return False
    if isinstance(dosing.doses[0], Bolus):
        return True
    return False


def has_seq_zo_fo_absorption(model: Model):
    """Check if ode system describes a sequential zero-order, first-order absorption

    Defined as the model having both zero- and first-order absorption.

    Parameters
    ----------
    model : Model
        DPharmpy model

    See also
    --------
    has_zero_order_absorption
    has_first_order_absorption

    """
    if has_zero_order_absorption(model) and has_first_order_absorption(model):
        return True
    else:
        return False


def get_number_of_peripheral_compartments(model: Model):
    """Return the number of peripherals compartments connected to the central
    compartment

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    int
        Number of peripherals compartments
    """
    # Redundant function ?
    odes = get_and_check_odes(model)
    return len(odes.find_peripheral_compartments())


def get_number_of_transit_compartments(model: Model):
    """Return the number of transit compartments in the model

    Parameters
    ----------
    model : Model
        Pharmpy model

    Returns
    -------
    int
        Number of transit compartments
    """
    # Redundant function ?
    odes = get_and_check_odes(model)
    return len(odes.find_transit_compartments(model.statements))


def has_lag_time(model: Model):
    """Check if ode system is defined with absorption lag time

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    -------
        Bool : True if model is defined with lagtime
    """
    odes = model.statements.ode_system
    if odes is None:
        return False

    dosing = odes.dosing_compartments[0]
    if dosing.lag_time:
        return True
    return False


def _add_zero_order_absorption(
    model, old_dose, to_comp, parameter_name, lag_time=None, replace=True
):
    """Add zero order absorption to a compartment. Initial estimate for absorption rate is set
    the previous rate if available, otherwise it is set to the time of first observation/2 is used.
    Disregards what is currently in the model.
    """
    mat_assign = model.statements.find_assignment(parameter_name)
    if mat_assign:
        mat_symb = mat_assign.symbol
    else:
        model, mat_symb = _add_parameter(
            model, parameter_name, init=_get_absorption_init(model, parameter_name)
        )
    new_dose = Infusion(old_dose.amount, admid=old_dose.admid, duration=mat_symb * 2)
    cb = CompartmentalSystemBuilder(model.statements.ode_system)
    dose_list = [new_dose] + list(to_comp.doses)
    dose_list.remove(old_dose)
    if replace:
        cb.set_dose(to_comp, tuple(dose_list))
    else:
        cb.add_dose(to_comp, tuple(dose_list))
    if lag_time is not None and lag_time != 0:
        odes = get_and_check_odes(model)
        cb.set_lag_time(odes.dosing_compartments[0], lag_time)
    model = model.replace(
        statements=model.statements.before_odes
        + CompartmentalSystem(cb)
        + model.statements.after_odes
    )
    return model


def _add_first_order_absorption(
    model, dose, to_comp, lag_time=None, bioavailability=None, remove_dose=True
):
    """Add first order absorption
    Disregards what is currently in the model.
    """
    odes = model.statements.ode_system
    cb = CompartmentalSystemBuilder(odes)
    depot = Compartment.create(
        'DEPOT',
        doses=(dose,),
        lag_time=Expr.integer(0) if lag_time is None else lag_time,
        bioavailability=Expr.integer(1) if bioavailability is None else bioavailability,
    )
    cb.add_compartment(depot)
    if remove_dose:
        to_comp = cb.set_dose(to_comp, tuple())
    to_comp = cb.set_lag_time(to_comp, Expr.integer(0))
    to_comp = cb.set_bioavailability(to_comp, Expr.integer(1))

    mat_assign = model.statements.find_assignment('MAT')
    if mat_assign:
        mat_symb = mat_assign.symbol
    else:
        model, mat_symb = _add_parameter(model, 'MAT', _get_absorption_init(model, 'MAT'))
    cb.add_flow(depot, to_comp, 1 / mat_symb)
    model = model.replace(
        statements=model.statements.before_odes
        + CompartmentalSystem(cb)
        + model.statements.after_odes
    )
    return model, depot


def _get_min_value(x) -> float:
    return float(x.min())


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
    if model.dataset is None:
        time_min = 1.0
    else:
        obs = get_observations(model)
        time = obs.index.get_level_values(level=time_label)
        time_min = _get_min_value(time[time > 0])

    if param_name == 'MDT':
        init = time_min / 2
    elif param_name == 'MAT':
        init = time_min * 2
    else:
        raise NotImplementedError('param_name must be MDT or MAT')

    if init < 0:
        init = 0.1

    return init


def set_peripheral_compartments(model: Model, n: int, name: Optional[str] = None):
    """Sets the number of peripheral compartments for central compartment to a specified number.

    If name is set, the peripheral compartment will be added to the compartment
    with the specified name instead.

    Parameters
    ----------
    model : Model
        Pharmpy model
    n : int
        Number of transit compartments
    name : Optional[str]
        Name of compartment to add peripheral to.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_peripheral_compartments(model, 2)
    >>> model.statements.ode_system
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────────┐
    │PERIPHERAL1│
    └───────────┘
      ↑      │
    Q2/V1  Q2/V2
      │      ↓
    ┌───────────┐
    │  CENTRAL  │──CL/V1→
    └───────────┘
       ↑      │
     Q3/V3  Q3/V1
       │      ↓
    ┌───────────┐
    │PERIPHERAL2│
    └───────────┘

    See also
    --------
    add_peripheral_compartment
    remove_peripheral_compartment

    """
    odes = get_and_check_odes(model)

    try:
        n = _as_integer(n)
    except TypeError:
        raise TypeError(f'Number of compartments must be integer: {n}')

    per = len(odes.find_peripheral_compartments())
    if per < n:
        for _ in range(n - per):
            model = add_peripheral_compartment(model, name=name)
    elif per > n:
        for _ in range(per - n):
            model = remove_peripheral_compartment(model, name=name)
    return model


def add_peripheral_compartment(model: Model, name: Optional[str] = None):
    r"""Add a peripheral distribution compartment to model

    The rate of flow from the central to the peripheral compartment
    will be parameterized as QPn / VC where VC is the volume of the central compartment.
    The rate of flow from the peripheral to the central compartment
    will be parameterized as QPn / VPn where VPn is the volumne of the added peripheral
    compartment.

    If name is set, the peripheral compartment will be added to the compartment
    with the specified name instead.

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
    name : Optional[str]
        Name of compartment to add peripheral to.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = add_peripheral_compartment(model)
    >>> model.statements.ode_system
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────────┐
    │PERIPHERAL1│
    └───────────┘
      ↑      │
     Q/V1   Q/V2
      │      ↓
    ┌───────────┐
    │  CENTRAL  │──CL/V1→
    └───────────┘

    See also
    --------
    set_peripheral_compartment
    remove_peripheral_compartment

    """
    statements = model.statements
    odes = get_and_check_odes(model)
    if name:
        other_compartment = odes.find_compartment(name)
        if other_compartment is None:
            raise ValueError(f'{name} is not a compartment name.')
        per = odes.find_peripheral_compartments(name)
        central = other_compartment
    else:
        per = odes.find_peripheral_compartments()
        central = odes.central_compartment

    n = len(per) + 1

    elimination_rate = odes.get_flow(central, output)
    assert elimination_rate is not None
    if has_mixed_mm_fo_elimination(model):
        elimination_rate = Expr(sympy.expand(elimination_rate).args[0])
    cl, vc = elimination_rate.as_numer_denom()
    if cl.is_symbol() and vc == 1:
        # If K = CL / V
        s = statements.find_assignment(cl.name)
        assert s is not None
        cl, vc = s.expression.as_numer_denom()
    # Heuristic to handle the MM case
    if vc.is_mul():
        vc = vc.args[0]

    cb = CompartmentalSystemBuilder(odes)

    # NOTE: Only used if vc != 1
    qp_init = 0.1
    vp_init = 0.1

    if n == 1:
        if vc != 1:
            # Heuristic to handle the Mixed MM-FO case
            if cl.is_add():
                cl1 = cl.args[0]
                assert isinstance(cl1, Expr)
                if cl1.is_mul():
                    cl = cl1.args[0]
            pop_cl = _find_noncov_theta(model, cl)
            pop_vc = _find_noncov_theta(model, vc)
            pop_cl_init = model.parameters[pop_cl].init
            pop_vc_init = model.parameters[pop_vc].init
            qp_init = pop_cl_init
            vp_init = pop_vc_init * 0.05
    elif n == 2:
        if vc != 1:
            per1 = per[0]
            from_rate = odes.get_flow(per1, central)
            assert from_rate is not None
            qp1, vp1 = from_rate.as_numer_denom()
            if qp1.is_symbol() and vc == 1:
                # If K = CL / V
                s = statements.find_assignment(qp1.name)
                assert s is not None
                qp1, vp1 = s.expression.as_numer_denom()
            full_qp1 = statements.before_odes.full_expression(qp1)
            full_vp1 = statements.before_odes.full_expression(vp1)
            if full_vp1 == 1:
                full_qp1, full_vp1 = full_qp1.as_numer_denom()
            pop_qp1 = _find_noncov_theta(model, full_qp1, full=True)
            pop_vp1 = _find_noncov_theta(model, full_vp1, full=True)
            pop_qp1_init = model.parameters[pop_qp1].init
            pop_vp1_init = model.parameters[pop_vp1].init
            model = set_initial_estimates(model, {pop_qp1.name: pop_qp1_init * 0.10})
            qp_init = pop_qp1_init * 0.90
            vp_init = pop_vp1_init

    if vc != 1:
        model, qp = _add_parameter(model, f'QP{n}', init=qp_init)
        model, vp = _add_parameter(model, f'VP{n}', init=vp_init)
        if name:
            peripheral = Compartment.create(f'{name}_PERIPHERAL{n}')
        else:
            peripheral = Compartment.create(f'PERIPHERAL{n}')
        cb.add_compartment(peripheral)
        cb.add_flow(central, peripheral, qp / vc)
        cb.add_flow(peripheral, central, qp / vp)
    elif vc == 1:
        model, kpc = _add_parameter(model, f'KPC{n}', init=0.1)
        model, kcp = _add_parameter(model, f'KCP{n}', init=0.1)
        if name:
            peripheral = Compartment.create(f'{name}_PERIPHERAL{n}')
        else:
            peripheral = Compartment.create(f'PERIPHERAL{n}')
        cb.add_compartment(peripheral)
        cb.add_flow(central, peripheral, kcp)
        cb.add_flow(peripheral, central, kpc)

    model = model.replace(
        statements=Statements(
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )
    )

    return model.update_source()


def remove_peripheral_compartment(model: Model, name: Optional[str] = None):
    r"""Remove a peripheral distribution compartment from model

    If name is set, a peripheral compartment will be removed from the compartment
    with the specified name.

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
    name : str
        Name of compartment to remove peripheral compartment from.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_peripheral_compartments(model, 2)
    >>> model = remove_peripheral_compartment(model)
    >>> model.statements.ode_system
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────────┐
    │PERIPHERAL1│
    └───────────┘
      ↑       │
    Q/V1     Q/V2
      │       ↓
    ┌───────────┐
    │  CENTRAL  │──CL/V1→
    └───────────┘

    See also
    --------
    set_peripheral_compartment
    add_peripheral_compartment

    """
    odes = get_and_check_odes(model)
    peripherals = odes.find_peripheral_compartments(name)

    if peripherals:
        last_peripheral = peripherals[-1]
        if name:
            central = odes.find_compartment(name)
        else:
            central = odes.central_compartment

        if len(peripherals) == 1:
            # TODO: Elimination can be zero (drug metabolite)
            elimination_rate = odes.get_flow(central, output)
            if elimination_rate == 0:
                pass
            else:
                cl, vc = elimination_rate.as_numer_denom()
                if vc != 1:
                    from_rate = odes.get_flow(last_peripheral, central)
                    assert from_rate is not None

                    qp1, vp1 = from_rate.as_numer_denom()
                    pop_cl = _find_noncov_theta(model, cl)
                    pop_vc = _find_noncov_theta(model, vc)
                    pop_qp1 = _find_noncov_theta(model, qp1)
                    pop_vp1 = _find_noncov_theta(model, vp1)
                    pop_vc_init = model.parameters[pop_vc].init
                    pop_cl_init = model.parameters[pop_cl].init
                    pop_qp1_init = model.parameters[pop_qp1].init
                    pop_vp1_init = model.parameters[pop_vp1].init
                    new_vc_init = pop_vc_init + pop_qp1_init / pop_cl_init * pop_vp1_init
                    model = set_initial_estimates(model, {pop_vc.name: new_vc_init})
        elif len(peripherals) == 2:
            first_peripheral = peripherals[0]
            from1_rate = odes.get_flow(first_peripheral, central)
            assert from1_rate is not None
            qp1, vp1 = from1_rate.as_numer_denom()
            from2_rate = odes.get_flow(last_peripheral, central)
            assert from2_rate is not None
            qp2, vp2 = from2_rate.as_numer_denom()
            pop_qp2 = _find_noncov_theta(model, qp2)
            pop_vp2 = _find_noncov_theta(model, vp2)
            pop_qp1 = _find_noncov_theta(model, qp1)
            pop_vp1 = _find_noncov_theta(model, vp1)
            pop_qp2_init = model.parameters[pop_qp2].init
            pop_vp2_init = model.parameters[pop_vp2].init
            pop_qp1_init = model.parameters[pop_qp1].init
            pop_vp1_init = model.parameters[pop_vp1].init
            new_qp1_init = (pop_qp1_init + pop_qp2_init) / 2
            new_vp1_init = pop_vp1_init + pop_vp2_init
            model = set_initial_estimates(
                model, {pop_qp1.name: new_qp1_init, pop_vp1.name: new_vp1_init}
            )

        rate1 = odes.get_flow(central, last_peripheral)
        assert rate1 is not None
        rate2 = odes.get_flow(last_peripheral, central)
        assert rate2 is not None
        symbols = rate1.free_symbols | rate2.free_symbols
        cb = CompartmentalSystemBuilder(odes)
        cb.remove_compartment(last_peripheral)
        new_odes = CompartmentalSystem(cb)
        model = model.replace(
            statements=(model.statements.before_odes + new_odes + model.statements.after_odes)
        )
        model = model.replace(
            statements=model.statements.remove_symbol_definitions(symbols, new_odes)
        )
        model = remove_unused_parameters_and_rvs(model)
    return model


def set_ode_solver(
    model: Model, solver: Literal['CVODES', 'DGEAR', 'DVERK', 'IDA', 'LSODA', 'LSODI']
):
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
    solver : {'CVODES', 'DGEAR', 'DVERK', 'IDA', 'LSODA', 'LSODI'}
        Solver to use or None for no preference

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_ode_solver(model, 'LSODA')

    """
    new_steps = []
    for step in model.execution_steps:
        new = step.replace(solver=solver)
        new_steps.append(new)
    newsteps = ExecutionSteps.create(new_steps)
    model = model.replace(execution_steps=newsteps).update_source()
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
    odes = get_and_check_odes(model)
    t = odes.t
    rate_list = _find_rate(model, sset)
    for rate in rate_list:
        if rate.is_symbol():
            assignment = sset.find_assignment(rate)
            assert assignment is not None
            rate = assignment.expression
        a, b = map(lambda x: x.free_symbols, rate.as_numer_denom())
        if b:
            clearance_symbols = a - b - {t}
            for clearance in clearance_symbols:
                clearance = _find_real_symbol(sset, clearance)
                if str(clearance) not in ['LAFREE', 'KINT', 'KON']:  # exclude TMDD parameters
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
    [VC]
    """
    vcs = set()
    sset = model.statements
    odes = sset.ode_system
    if odes is None:
        raise ValueError(f'Model {model.name} has no ODE system')
    t = odes.t
    rate_list = _find_rate(model, sset)
    for rate in rate_list:
        if rate.is_symbol():
            assignment = sset.find_assignment(rate)
            assert assignment is not None
            rate = assignment.expression
        rate = Expr(sympy.cancel(rate))
        a, b = map(lambda x: x.free_symbols, rate.as_numer_denom())
        volume_symbols = b - a - {t}
        for volume in volume_symbols:
            volume = _find_real_symbol(sset, volume)
            vcs.add(volume)
    return sorted(vcs, key=str)


def _find_rate(model: Model, sset: Statements):
    rate_list = []
    odes = sset.ode_system
    assert isinstance(odes, CompartmentalSystem)
    central = odes.central_compartment
    # FIXME : How to specify metabolite compartment?
    metacomp = odes.find_compartment("METABOLITE")
    if metacomp:
        central_to_metabolite = odes.get_flow(central, metacomp)
        rate_list.append(central_to_metabolite)
        elimination_rate = odes.get_flow(metacomp, output)
        rate_list.append(elimination_rate)

    else:
        elimination_rate = odes.get_flow(central, output)
        rate_list.append(elimination_rate)
        if has_mixed_mm_fo_elimination(model):
            elimination_rate = Expr(sympy.expand(elimination_rate))
            rate_list.append(elimination_rate.args[0])
            rate_list.append(elimination_rate.args[1])

    comp_peripherals = {}
    comp_peripherals[central] = odes.find_peripheral_compartments()
    if metacomp:
        comp_peripherals[metacomp] = odes.find_peripheral_compartments(name=metacomp.name)

    for comp, peripherals in comp_peripherals.items():
        for periph in peripherals:
            rate1 = odes.get_flow(comp, periph)
            rate_list.append(rate1)
            rate2 = odes.get_flow(periph, comp)
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
            None,  # NOTE: filters out falsy values
            (sset.find_assignment(symb) for symb in assignment.expression.free_symbols),
        )
    )


def has_odes(model: Model) -> bool:
    """Check if model has an ODE system

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    bool
        True if model has an ODE system

    See also
    --------
    has_linear_odes
        has_linear_odes_with_real_eigenvalues

    Examples
    --------
    >>> from pharmpy.modeling import has_odes, load_example_model
    >>> model = load_example_model("pheno")
    >>> has_odes(model)
    True
    """

    odes = model.statements.ode_system
    return bool(odes)


def has_linear_odes(model: Model) -> bool:
    """Check if model has a linear ODE system

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    bool
        True if model has an ODE system that is linear

    See also
    --------
    has_odes
        has_linear_odes_with_real_eigenvalues

    Examples
    --------
    >>> from pharmpy.modeling import has_linear_odes, load_example_model
    >>> model = load_example_model("pheno")
    >>> has_linear_odes(model)
    True
    """

    if not has_odes(model):
        return False

    odes = model.statements.ode_system
    assert isinstance(odes, CompartmentalSystem)
    symbs = odes.compartmental_matrix.free_symbols | odes.zero_order_inputs.free_symbols
    return odes.t not in symbs


def has_linear_odes_with_real_eigenvalues(model: Model) -> bool:
    """Check if model has a linear ode system with real eigenvalues

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    bool
        True if model has an ODE system that is linear

    See also
    --------
    has_odes
        has_linear_odes

    Examples
    --------
    >>> from pharmpy.modeling import has_linear_odes_with_real_eigenvalues, load_example_model
    >>> model = load_example_model("pheno")
    >>> has_linear_odes_with_real_eigenvalues(model)
    True
    """

    odes = model.statements.ode_system
    if odes is None:
        return False
    if not has_linear_odes(model):
        return False
    if _check_all_cycles(model):
        return True
    M = odes.compartmental_matrix
    eigs = M.eigenvals().keys()
    for eig in eigs:
        real = is_real(model, eig)
        if real is None or not real:
            return bool(real)
    return True


def _check_all_cycles(model):
    # This checks a sufficient but not necessary condition on compartmental systems
    # to have all eigenvalues real that is faster to compute than the
    # direct calculation of eigenvalues.
    # If for all cycles (assuming simple cycles) in the compartmental system
    # the product of all rates in one direction is equal to the product of all rates
    # in the opposite direction the compartmental system has only real valued eigenvalues
    # Source: Keith Godfrey, "Compartmental Models and Their Applications", 1983, p30-31

    import networkx as nx
    import sympy

    # FIXME: Implement methods instead of using networkx and sympy directly
    odes = model.statements.ode_system
    g = model.statements.ode_system._g
    for cycle in nx.simple_cycles(g):
        prod1 = prod2 = Expr.integer(1)  # Products for the two directions
        for i, comp in enumerate(cycle):
            if i != len(cycle) - 1:  # Not final element
                flow1 = odes.get_flow(cycle[i], cycle[i + 1])
                flow2 = odes.get_flow(cycle[i + 1], cycle[i])
            else:  # Final element so we go back to first compartment in cycle
                flow1 = odes.get_flow(cycle[i], cycle[0])
                flow2 = odes.get_flow(cycle[0], cycle[i])
            prod1 *= flow1
            prod2 *= flow2
        eq = sympy.sympify(prod1).equals(sympy.sympify(prod2))
        if not eq:
            return False
    return True


def get_initial_conditions(model: Model, dosing: bool = False) -> Mapping[Expr, Expr]:
    """Get initial conditions for the ode system

    Default initial conditions at t=0 for amounts is 0

    Parameters
    ----------
    model : Model
        Pharmpy model
    dosing : bool
        Set to True to add dosing as initial conditions

    Return
    ------
    dict
        Initial conditions

    Examples
    --------
    >>> from pharmpy.modeling import get_initial_conditions, load_example_model
    >>> model = load_example_model("pheno")
    >>> get_initial_conditions(model)
    {A_CENTRAL(0): 0}
    >>> get_initial_conditions(model, dosing=True)
    {A_CENTRAL(0): AMT}
    """
    d = {}
    odes = model.statements.ode_system
    if odes is None:
        return d
    assert isinstance(odes, CompartmentalSystem)
    for amt in odes.amounts:
        d[Expr.function(amt.name, 0)] = Expr.integer(0)
    for s in model.statements:
        if isinstance(s, Assignment) and s.symbol.is_function():
            if not (s.symbol.args[0].free_symbols):  # pyright: ignore[reportAttributeAccessIssue]
                d[s.symbol] = s.expression

    if dosing:
        for name in odes.compartment_names:
            comp = odes.find_compartment(name)
            assert comp is not None
            if comp.doses and isinstance(comp.doses[0], Bolus):
                if comp.lag_time:
                    time = comp.lag_time
                else:
                    time = 0
                d[Expr.function(comp.amount.name, time)] = comp.doses[0].amount

    return d


def set_initial_condition(
    model: Model,
    compartment: str,
    expression: TExpr,
    time: TExpr = Expr.integer(0),
) -> Model:
    """Set an initial condition for the ode system

    If the initial condition is already set it will be updated. If the initial condition
    is set to zero at time zero it will be removed (since the default is 0).

    Parameters
    ----------
    model : Model
        Pharmpy model
    compartment : str
        Name of the compartment
    expression : Union[str, Expr]
        The expression of the initial condition
    time : Union[str, Expr]
        Time point. Default 0

    Return
    ------
    model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_initial_condition(model, "CENTRAL", 10)
    >>> get_initial_conditions(model)
    {A_CENTRAL(0): 10}
    """
    odes = model.statements.ode_system
    if odes is None:
        raise ValueError("Model has no system of ODEs")
    comp = odes.find_compartment(compartment)
    if comp is None:
        raise ValueError(f"Model has no compartment named {compartment}")
    expr = Expr(expression)
    time = Expr(time)
    amount = Expr.function(comp.amount.name, time)
    assignment = Assignment.create(amount, expr)
    statements = None
    for i, s in enumerate(model.statements.before_odes):
        if s.symbol == amount:
            if time == 0 and expr == 0:
                statements = model.statements[:i] + model.statements[i + 1 :]
            else:
                statements = model.statements[:i] + assignment + model.statements[i + 1 :]
            break
    else:
        if not (time == 0 and expr == 0):
            statements = (
                model.statements.before_odes + assignment + odes + model.statements.after_odes
            )
    if statements is not None:
        model = model.replace(statements=statements)
        model = model.update_source()
    return model


def get_zero_order_inputs(model: Model) -> Matrix:
    """Get zero order inputs for all compartments

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Matrix
        Vector of inputs

    Examples
    --------
    >>> from pharmpy.modeling import get_zero_order_inputs, load_example_model
    >>> model = load_example_model("pheno")
    >>> get_zero_order_inputs(model)
    [0]
    """
    odes = model.statements.ode_system
    if odes is None:
        return Matrix()
    return odes.zero_order_inputs


def set_zero_order_input(model: Model, compartment: str, expression: TExpr) -> Model:
    """Set a zero order input for the ode system

    If the zero order input is already set it will be updated.

    Parameters
    ----------
    model : Model
        Pharmpy model
    compartment : str
        Name of the compartment
    expression : Union[str, Expr]
        The expression of the zero order input

    Return
    ------
    model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_zero_order_input(model, "CENTRAL", 10)
    >>> get_zero_order_inputs(model)
    [10]
    """
    odes = model.statements.ode_system
    if odes is None:
        raise ValueError("Model has no system of ODEs")
    comp = odes.find_compartment(compartment)
    if comp is None:
        raise ValueError(f"Model has no compartment named {compartment}")
    expr = Expr(expression)
    cb = CompartmentalSystemBuilder(odes)
    cb.set_input(comp, expr)
    cs = CompartmentalSystem(cb)
    statements = model.statements.before_odes + cs + model.statements.after_odes
    model = model.replace(statements=statements)
    return model.update_source()


class ODEDisplayer:
    def __init__(self, eqs, ics):
        self._eqs = eqs
        self._ics = ics

    def __repr__(self):
        if self._eqs is None:
            return ""
        a = []
        for ode in self._eqs:
            ode_str = ode.unicode()
            a += ode_str.split('\n')
        for key, value in self._ics.items():
            ics_str = sympy.pretty(sympy.Eq(key, value))
            a += ics_str.split('\n')
        return bracket(a)

    def _repr_latex_(self):
        if self._eqs is None:
            return ""
        rows = []
        for ode in self._eqs:
            ode_repr = sympy.latex(ode, mul_symbol='dot')
            rows.append(ode_repr)
        for k, v in self._ics.items():
            ics_eq = sympy.Eq(k, v)
            ics_repr = sympy.latex(ics_eq, mul_symbol='dot')
            rows.append(ics_repr)
        return r'\begin{cases} ' + r' \\ '.join(rows) + r' \end{cases}'


def display_odes(model: Model):
    """Displays the ordinary differential equation system

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    ODEDisplayer
        A displayable object

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> display_odes(model)
    ⎧d                  -CL⋅A_CENTRAL(t)
    ⎨──(A_CENTRAL(t)) = ─────────────────
    ⎩dt                         V
    <BLANKLINE>

    """

    odes = model.statements.ode_system
    if odes is not None:
        eqs = odes.eqs
        ics = {key: val for key, val in get_initial_conditions(model).items() if val != 0}
    else:
        eqs = None
        ics = None
    return ODEDisplayer(eqs, ics)


def solve_ode_system(model: Model):
    """Replace ODE system with analytical solution if possible

    Warnings
    --------
    This function can currently only handle the most simple of ODE systems.

    Parameters
    ----------
    model : Model
        Pharmpy model object

    Returns
    -------
    Model
        Pharmpy model object

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model.statements.ode_system
    Bolus(AMT, admid=1) → CENTRAL
    ┌───────┐
    │CENTRAL│──CL/V→
    └───────┘
    >>> model = solve_ode_system(model)

    """
    odes = model.statements.ode_system
    if odes is None:
        return model
    ics = get_initial_conditions(model, dosing=True)
    # FIXME: Should set assumptions on symbols before solving
    # FIXME: Need a way to handle systems with no explicit solutions
    sol = sympy.dsolve(sympy.sympify(odes.eqs), ics=sympy.sympify(ics))
    assert isinstance(sol, Iterable)

    new = []
    for s in model.statements:
        if isinstance(s, CompartmentalSystem):
            for eq in sol:
                assert isinstance(eq, sympy.Eq)
                ass = Assignment.create(eq.lhs, eq.rhs)
                new.append(ass)
        else:
            new.append(s)
    model = model.replace(statements=Statements(new)).update_source()
    return model


def get_central_volume_and_clearance(model: Model):
    """Get the volume and clearance parameters

    Parameters
    ----------
    model : Model
        Pharmpy model


    Returns
    -------
    sympy.Symbol
        Volume symbol
    sympy.Symbol
        Clearance symbol

    Example
    -------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> get_central_volume_and_clearance(model)
    (VC, CL)

    """
    vcs = set()
    cls = set()
    sset = model.statements
    odes = get_and_check_odes(model)
    t = odes.t
    central_comp = odes.central_compartment

    from .metabolite import has_presystemic_metabolite  # Circular import issue

    if has_presystemic_metabolite(model):
        metabolite = odes.find_compartment("METABOLITE")
        rate = odes.get_flow(central_comp, metabolite)
    else:
        rate = odes.get_flow(central_comp, output)
    rate = sympy.expand(rate)
    if has_mixed_mm_fo_elimination(model):
        rate = rate.args[0]
    if isinstance(rate, sympy.Symbol):
        assignment = sset.find_assignment(rate)
        assert assignment is not None
        rate = assignment.expression
    rate = Expr(sympy.cancel(rate))
    a, b = map(lambda x: x.free_symbols, rate.as_numer_denom())
    if b:
        # Get volume parameter
        volume_symbols = b - a - {t}
        for volume in volume_symbols:
            volume = _find_real_symbol(sset, volume)
            vcs.add(volume)
        # Get clearance parameter
        clearance_symbols = a - b - {t}
        for clearance in clearance_symbols:
            clearance = _find_real_symbol(sset, clearance)
            cls.add(clearance)
    else:
        raise ValueError('Model is not suitable')
    return list(vcs)[0], list(cls)[0]
