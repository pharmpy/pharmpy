"""
:meta private:
"""

from __future__ import annotations

from pharmpy.basic import Expr
from pharmpy.model import (
    Assignment,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    Model,
    get_and_check_odes,
    output,
)

from .error import set_proportional_error_model
from .odes import _find_noncov_theta, add_individual_parameter, set_first_order_absorption
from .parameters import set_initial_estimates, set_lower_bounds, set_upper_bounds


def add_metabolite(model: Model, drug_dvid: int = 1, presystemic: bool = False) -> Model:
    """Adds a metabolite compartment to a model

    The flow from the central compartment to the metabolite compartment
    will be unidirectional.

    Presystemic indicate that the metabolite compartment will be
    directly connected to the DEPOT. If a depot compartment is not present,
    one will be created.

    Parameters
    ----------
    model : Model
        Pharmpy model
    drug_dvid : int
        DVID for drug (assuming all other DVIDs being for metabolites)
    presystemic : bool
        Decide wether or not to add metabolite as a presystemetic fixed drug.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = add_metabolite(model)

    """
    if presystemic:
        odes = get_and_check_odes(model)
        depot = odes.find_depot(model.statements)
        if not depot:
            model = set_first_order_absorption(model)
            odes = get_and_check_odes(model)
            depot = odes.find_depot(model.statements)
            if not depot:
                transits = odes.find_transit_compartments(model.statements)
                if transits:
                    depot = transits[-1]
                else:
                    raise ValueError(
                        "Pre-systemic metabolite model is not compatible with input model."
                    )
    else:
        depot = None

    # TODO: Implement possibility of converting plain metabolite to presystemic

    clm = Expr.symbol('CLM')
    model = add_individual_parameter(model, clm.name)
    vm = Expr.symbol('VM')
    model = add_individual_parameter(model, vm.name)

    odes = get_and_check_odes(model)
    central = odes.central_compartment
    ke = odes.get_flow(central, output)
    if ke.is_symbol():
        ke_ass = model.statements.find_assignment(ke)
        if ke_ass is not None:
            ke = ke_ass.expression
    cl, vc = ke.as_numer_denom()

    if vc != 1:
        pop_cl = _find_noncov_theta(model, cl)
        pop_vc = _find_noncov_theta(model, vc)
        pop_clm_init = model.parameters[pop_cl].init * 2.0
        pop_vm_init = model.parameters[pop_vc].init * 0.5
        model = set_initial_estimates(model, {'POP_CLM': pop_clm_init, 'POP_VM': pop_vm_init})

    cb = CompartmentalSystemBuilder(odes)
    metacomp = Compartment.create(name="METABOLITE")
    cb.add_compartment(metacomp)
    cb.add_flow(metacomp, output, clm / vm)

    cb.add_flow(central, metacomp, ke)
    cb.remove_flow(central, output)
    amount = metacomp.amount
    if presystemic:
        assert isinstance(depot, Compartment)
        fpre = Expr.symbol('FPRE')
        model = add_individual_parameter(model, fpre.name)
        model = set_lower_bounds(model, {'POP_FPRE': 0.0})
        model = set_upper_bounds(model, {'POP_FPRE': 1.0})
        ka = odes.get_flow(depot, central)
        if ka.is_symbol():
            ka_ass = model.statements.find_assignment(ka)
            if ka_ass is not None:
                ka = ka_ass.expression
        cb.add_flow(depot, metacomp, fpre * ka)
        cb.remove_flow(depot, central)
        cb.add_flow(depot, central, ka * (1 - fpre))

        # FIXME: drug_dvid is never used, use it here?
        # dvid_col = model.datainfo.typeix['dvid'][0]
        # dvids = dvid_col.categories

        # QUESTION: Add bioavailability to depot?
        cb.set_bioavailability(depot, 1 / (1 - fpre))
        model = model.replace(
            statements=model.statements.before_odes
            + CompartmentalSystem(cb)
            + model.statements.after_odes
        )
        model = model.update_source()
        cs = get_and_check_odes(model)
        biocomp = cs.find_compartment(depot.name)
        assert isinstance(biocomp, Compartment)
        bio = biocomp.bioavailability
        conc = Assignment(Expr.symbol('CONC_M'), amount / vm / bio)
    else:
        cs = CompartmentalSystem(cb)
        conc = Assignment(Expr.symbol('CONC_M'), amount / vm)
    y_m = Expr.symbol('Y_M')
    y = Assignment(y_m, conc.symbol)
    original_y = next(iter(model.dependent_variables))
    ind = model.statements.after_odes.find_assignment_index(original_y)
    assert ind is not None
    old_after = model.statements.after_odes
    new_after = old_after[: ind + 1] + y + old_after[ind + 1 :]
    error = conc + new_after
    dvs = model.dependent_variables.replace(y_m, 2)  # FIXME: Should be next DVID in categories
    statements = model.statements.before_odes + cs + error
    model = model.replace(statements=statements, dependent_variables=dvs)
    model = set_proportional_error_model(model, dv=2, zero_protection=False)
    return model


def has_presystemic_metabolite(model: Model):
    """Checks whether a model has a presystemic metabolite

    If pre-systemic drug there will be a flow from DEPOT to METABOLITE as well
    as being a flow from the CENTRAL to METABOLITE

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    bool
        Whether a model has presystemic metabolite

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = add_metabolite(model, presystemic=True)
    >>> has_presystemic_metabolite(model)
    True

    """
    odes = model.statements.ode_system
    if odes is None:
        return False
    central = odes.central_compartment
    metabolite = odes.find_compartment("METABOLITE")
    depot = odes.find_depot(model.statements)

    CM = odes.get_flow(central, metabolite)
    DM = odes.get_flow(depot, metabolite)

    if CM != Expr.integer(0) and DM != Expr.integer(0):
        return True
    else:
        return False
