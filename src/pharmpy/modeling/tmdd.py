"""
:meta private:
"""

from __future__ import annotations

from typing import Literal, Optional

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
from pharmpy.modeling import get_central_volume_and_clearance

from .error import set_proportional_error_model
from .expressions import _replace_trivial_redefinitions
from .odes import add_individual_parameter, set_first_order_elimination, set_initial_condition
from .parameter_variability import add_iiv

TMDD_TYPE = ('full', 'ib', 'cr', 'crib', 'qss', 'wagner', 'mmapp')
DV_TYPES = ('drug', 'drug_tot', 'target', 'target_tot', 'complex')


def set_tmdd(
    model: Model,
    type: Literal['full', 'ib', 'cr', 'crib', 'qss', 'wagner', 'mmapp'],
    dv_types: Optional[
        dict[Literal['drug', 'drug_tot', 'target', 'target_tot', 'complex'], int]
    ] = None,
):
    """Sets target mediated drug disposition

    Implemented target mediated drug disposition (TMDD) models are:

    - Full model
    - Irreversible binding approximation (IB)
    - Constant total receptor approximation (CR)
    - Irreversible binding and constant total receptor approximation (CR+IB)
    - Quasi steady-state approximation (QSS)
    - Wagner
    - Michaelis-Menten approximation (MMAPP)


    Parameters
    ----------
    model : Model
        Pharmpy model
    type : str
        Type of TMDD model
    dv_types: dict
        Dictionary of DV types for TMDD models with multiple DVs (e.g. dv_types = {'drug' : 1, 'target': 2}).
        Default is None which means that all observations are treated as drug observations.
        For dv = 1 the only allowed keys are 'drug' and 'drug_tot'. If no DV for drug is specified then (free) drug
        will have dv = 1.

    Return
    ------
    Model
        Pharmpy model object

    Examples
    --------
    >>> from pharmpy.modeling import *
    >>> model = load_example_model("pheno")
    >>> model = set_tmdd(model, "full")

    """
    if dv_types is not None:
        _validate_dv_types(dv_types)

    uptype = type.upper()

    model = _replace_trivial_redefinitions(model)
    model = set_first_order_elimination(model)

    odes = get_and_check_odes(model)
    central = odes.central_compartment
    cb = CompartmentalSystemBuilder(odes)

    vc, cl = get_central_volume_and_clearance(model)
    r_0 = Expr.symbol('R_0')
    model = add_individual_parameter(model, r_0.name)
    model = add_iiv(model, [r_0.name], 'exp')
    kint = Expr.symbol('KINT')
    model = add_individual_parameter(model, kint.name)

    y_symbol = _get_y_symbol(model)

    lafreef = Expr.symbol("LAFREEF")
    target_comp = Compartment.create(name='TARGET')
    complex_comp = Compartment.create(name='COMPLEX')

    if uptype == "FULL":
        model, kon, koff, kdeg = _create_parameters(model, ['KON', 'KOFF', 'KDEG'])
        cb.add_compartment(target_comp)
        cb.add_compartment(complex_comp)
        ksyn, ksyn_ass = _create_ksyn()

        cb.add_flow(target_comp, complex_comp, kon * central.amount / vc)
        cb.add_flow(complex_comp, target_comp, koff)
        cb.add_flow(target_comp, output, kdeg)
        cb.add_flow(complex_comp, output, kint)
        cb.set_input(target_comp, ksyn * vc)
        cb.set_input(
            central, koff * complex_comp.amount - kon * central.amount * target_comp.amount / vc
        )

        before = model.statements.before_odes + ksyn_ass
        after = model.statements.after_odes
    elif uptype == "IB":
        model, kdeg, kon = _create_parameters(model, ['KDEG', 'KON'])
        cb.add_compartment(target_comp)
        cb.add_compartment(complex_comp)
        ksyn, ksyn_ass = _create_ksyn()

        cb.add_flow(target_comp, complex_comp, kon * central.amount / vc)
        cb.add_flow(target_comp, output, kdeg)
        cb.add_flow(complex_comp, output, kint)
        cb.set_input(target_comp, ksyn * vc)
        cb.set_input(central, -kon * central.amount * target_comp.amount / vc)

        before = model.statements.before_odes + ksyn_ass
        after = model.statements.after_odes
    elif uptype == "CR":
        model, kon, koff = _create_parameters(model, ['KON', 'KOFF'])
        cb.add_compartment(complex_comp)

        cb.add_flow(complex_comp, central, koff + kon * central.amount / vc)
        cb.add_flow(complex_comp, output, kint)
        cb.add_flow(central, complex_comp, kon * r_0)

        before = model.statements.before_odes
        after = model.statements.after_odes
    elif uptype == "CRIB":
        model, kon = _create_parameters(model, ['KON'])
        cb.add_compartment(complex_comp)

        cb.add_flow(complex_comp, output, kint)
        cb.add_flow(central, complex_comp, kon * r_0)
        cb.add_flow(complex_comp, central, kon * central.amount / vc)

        before = model.statements.before_odes
        after = model.statements.after_odes
    elif uptype == "QSS":
        model, kdc, kdeg = _create_parameters(model, ['KDC', 'KDEG'])
        cb.add_compartment(target_comp)

        kd = Expr.symbol('KD')

        ksyn, ksyn_ass = _create_ksyn()
        kd_ass = Assignment.create(kd, kdc * vc)

        lafree_symb = Expr.symbol('LAFREE')
        lafree_expr = (
            central.amount
            - target_comp.amount
            - kd
            + ((central.amount - target_comp.amount - kd) ** 2 + 4 * kd * central.amount).sqrt()
        ) / 2
        lafree_ass = Assignment.create(lafree_symb, lafree_expr)

        num_peripheral_comp = len(odes.find_peripheral_compartments())
        if num_peripheral_comp == 0:
            elimination_rate = odes.get_flow(central, output)
            cb.remove_flow(central, output)
            cb.set_input(
                central,
                -lafree_symb * elimination_rate
                - target_comp.amount * kint * lafree_symb / (kd + lafree_symb),
            )
            cb.set_input(
                target_comp,
                ksyn * vc
                - kdeg * target_comp.amount
                - (kint - kdeg) * target_comp.amount * lafree_symb / (kd + lafree_symb),
            )

            # FIXME: Should others also have flows?
            central = cb.find_compartment('CENTRAL')
            assert isinstance(central, Compartment)
            cb.add_flow(central, output, lafree_symb * elimination_rate)

            lafree_final = Assignment.create(lafreef, lafree_expr)
            before = model.statements.before_odes + (ksyn_ass, kd_ass, lafree_ass)
            after = lafree_final + model.statements.after_odes
            ipred = lafreef / vc
            after = after.reassign(y_symbol, ipred)
        elif num_peripheral_comp > 0 and num_peripheral_comp <= 2:
            peripheral1 = _create_compartment(cb, 'PERIPHERAL1')
            flow_central_peripheral1 = odes.get_flow(central, peripheral1)
            if num_peripheral_comp == 2:
                peripheral2 = _create_compartment(cb, 'PERIPHERAL2')
                flow_central_peripheral2 = odes.get_flow(central, peripheral2)
            else:
                peripheral2 = None
                flow_central_peripheral2 = None

            elimination_rate = odes.get_flow(central, output)
            cb.remove_flow(central, output)
            if num_peripheral_comp == 1:
                cb.set_input(
                    central,
                    -target_comp.amount * kint * lafree_symb / (kd + lafree_symb)
                    - lafree_symb * flow_central_peripheral1
                    + flow_central_peripheral1 * central.amount,
                )
                cb.set_input(
                    target_comp,
                    ksyn * vc
                    - kdeg * target_comp.amount
                    - (kint - kdeg) * target_comp.amount * lafree_symb / (kd + lafree_symb),
                )
                cb.set_input(
                    peripheral1,
                    lafree_symb * flow_central_peripheral1
                    - flow_central_peripheral1 * central.amount,
                )
            elif flow_central_peripheral2 is not None and peripheral2 is not None:
                cb.set_input(
                    central,
                    -target_comp.amount * kint * lafree_symb / (kd + lafree_symb)
                    - lafree_symb * flow_central_peripheral1
                    - lafree_symb * flow_central_peripheral2
                    + flow_central_peripheral2 * central.amount,
                )
                cb.set_input(
                    target_comp,
                    ksyn * vc
                    - kdeg * target_comp.amount
                    - (kint - kdeg) * target_comp.amount * lafree_symb / (kd + lafree_symb),
                )
                cb.set_input(
                    peripheral1,
                    lafree_symb * flow_central_peripheral1
                    - flow_central_peripheral1 * central.amount,
                )
                cb.set_input(
                    peripheral2,
                    lafree_symb * flow_central_peripheral2
                    - flow_central_peripheral2 * central.amount,
                )

            central = cb.find_compartment('CENTRAL')
            assert isinstance(central, Compartment)
            cb.add_flow(central, output, lafree_symb * elimination_rate / central.amount)

            lafree_final = Assignment.create(lafreef, lafree_expr)
            before = model.statements.before_odes + (ksyn_ass, kd_ass, lafree_ass)
            after = lafree_final + model.statements.after_odes
            ipred = lafreef / vc
            after = after.reassign(y_symbol, ipred)
        else:
            raise ValueError('More than 2 peripheral compartments are not supported.')
    elif uptype == 'WAGNER':
        model, km = _create_parameters(model, ['KM'])

        kel = odes.get_flow(central, output)
        kd = km * vc
        rinit = r_0 * vc
        rinit_ass = Assignment(Expr.symbol('RINIT'), rinit)
        kd_ass = Assignment(Expr.symbol('KD'), km * vc)

        lafree_symb = Expr.symbol('LAFREE')
        lafree_expr = (
            central.amount
            - rinit
            - kd
            + ((central.amount - rinit - kd) ** 2 + 4 * kd * central.amount).sqrt()
        ) / 2
        lafree_ass = Assignment.create(lafree_symb, lafree_expr)

        num_peripheral_comp = len(odes.find_peripheral_compartments())
        if num_peripheral_comp == 0:
            cb.add_flow(central, output, kel)
            cb.set_input(
                central,
                kint * lafree_symb
                - kint * central.amount
                - kel * lafree_symb
                + kel * central.amount,
            )
        elif num_peripheral_comp == 1:
            peripheral = _create_compartment(cb, 'PERIPHERAL1')
            kcp = odes.get_flow(central, peripheral)
            cb.add_flow(central, output, kel)
            cb.set_input(
                central,
                kint * lafree_symb
                - kint * central.amount
                - kel * lafree_symb
                + kel * central.amount
                - kcp * lafree_symb
                + kcp * central.amount,
            )
            cb.set_input(peripheral, kcp * lafree_symb - kcp * central.amount)
        elif num_peripheral_comp == 2:
            peripheral1 = _create_compartment(cb, 'PERIPHERAL1')
            kcp1 = odes.get_flow(central, peripheral1)
            peripheral2 = _create_compartment(cb, 'PERIPHERAL2')
            kcp2 = odes.get_flow(central, peripheral2)

            cb.add_flow(central, output, kel)
            cb.set_input(
                central,
                kint * lafree_symb
                - kint * central.amount
                - kel * lafree_symb
                + kel * central.amount
                - kcp1 * lafree_symb
                - kcp2 * lafree_symb,
            )
            cb.set_input(peripheral1, kcp1 * lafree_symb - kcp1 * central.amount)
            cb.set_input(peripheral2, kcp2 * lafree_symb - kcp2 * central.amount)

        lafree_final = Assignment.create(lafreef, lafree_expr)
        before = model.statements.before_odes + lafree_ass + kd_ass + rinit_ass
        after = lafree_final + model.statements.after_odes
        ipred = lafreef / vc
        after = after.reassign(y_symbol, ipred)
    elif uptype == 'MMAPP':
        model, km, kdeg = _create_parameters(model, ['KM', 'KDEG'])
        cb.add_compartment(target_comp)
        ksyn, ksyn_ass = _create_ksyn()

        target_elim = kdeg + (kint - kdeg) * central.amount / vc / (km + central.amount / vc)
        cb.add_flow(target_comp, output, target_elim)
        elim = cl / vc
        cb.add_flow(central, output, elim)
        cb.set_input(target_comp, ksyn)
        cb.set_input(
            central, -target_comp.amount * central.amount * kint / (central.amount / vc + km)
        )

        before = model.statements.before_odes + ksyn_ass
        after = model.statements.after_odes
    else:
        raise ValueError(f'Unknown TMDD type "{type}".')

    model = model.replace(statements=before + CompartmentalSystem(cb) + after)
    if uptype not in ('CR', 'CRIB', 'WAGNER', 'MMAPP'):
        model = set_initial_condition(model, "TARGET", r_0 * vc)
    if uptype == 'MMAPP':
        model = set_initial_condition(model, "TARGET", r_0)

    # Multiple DVs:
    if dv_types is not None:
        if uptype in ('FULL', 'IB'):
            if 'drug_tot' in dv_types.keys():
                new_y = (central.amount + complex_comp.amount) / vc
                after = model.statements.after_odes
                after = after.reassign(y_symbol, new_y)
                model = model.replace(
                    statements=model.statements.before_odes + get_and_check_odes(model) + after
                )
            if 'target' in dv_types.keys():
                y_target = Expr.symbol("Y_TARGET")
                ytarget = Assignment.create(y_target, target_comp.amount / vc)
                dvs = model.dependent_variables.replace(y_target, dv_types['target'])
                model = model.replace(
                    statements=model.statements + ytarget, dependent_variables=dvs
                )
            if 'complex' in dv_types.keys():
                y_complex = Expr.symbol("Y_COMPLEX")
                ycomplex = Assignment.create(y_complex, complex_comp.amount / vc)
                dvs = model.dependent_variables.replace(y_complex, dv_types['complex'])
                model = model.replace(
                    statements=model.statements + ycomplex, dependent_variables=dvs
                )
            if 'target_tot' in dv_types.keys():
                y_target_tot = Expr.symbol("Y_TOTTARGET")
                ytargettot = Assignment.create(
                    y_target_tot, (target_comp.amount + complex_comp.amount) / vc
                )
                dvs = model.dependent_variables.replace(y_target_tot, dv_types['target_tot'])
                model = model.replace(
                    statements=model.statements + ytargettot, dependent_variables=dvs
                )
        elif uptype == 'QSS':
            if 'drug_tot' in dv_types.keys():
                new_y = central.amount / vc
                after = model.statements.after_odes
                after = after.reassign(y_symbol, new_y)
                model = model.replace(
                    statements=model.statements.before_odes + get_and_check_odes(model) + after
                )
            if 'target' in dv_types.keys():
                y_target = Expr.symbol("Y_TARGET")
                ytarget = Assignment.create(
                    y_target, (target_comp.amount - central.amount + lafreef) / vc
                )
                dvs = model.dependent_variables.replace(y_target, dv_types['target'])
                model = model.replace(
                    statements=model.statements + ytarget, dependent_variables=dvs
                )
            if 'complex' in dv_types.keys():
                y_complex = Expr.symbol("Y_COMPLEX")
                ycomplex = Assignment.create(y_complex, (central.amount - lafreef) / vc)
                dvs = model.dependent_variables.replace(y_complex, dv_types['complex'])
                model = model.replace(
                    statements=model.statements + ycomplex, dependent_variables=dvs
                )
            if 'target_tot' in dv_types.keys():
                y_target_tot = Expr.symbol("Y_TOTTARGET")
                ytargettot = Assignment.create(y_target_tot, target_comp.amount / vc)
                dvs = model.dependent_variables.replace(y_target_tot, dv_types['target_tot'])
                model = model.replace(
                    statements=model.statements + ytargettot, dependent_variables=dvs
                )
        elif uptype == 'MMAPP':
            if 'target' in dv_types.keys():
                y_target = Expr.symbol("Y_TARGET")
                ytarget = Assignment.create(y_target, target_comp.amount / vc)
                dvs = model.dependent_variables.replace(y_target, dv_types['target'])
                model = model.replace(
                    statements=model.statements + ytarget, dependent_variables=dvs
                )
            if 'target_tot' in dv_types.keys():
                y_target_tot = Expr.symbol("Y_TOTTARGET")
                ytargettot = Assignment.create(y_target_tot, target_comp.amount / vc)
                dvs = model.dependent_variables.replace(y_target_tot, dv_types['target_tot'])
                model = model.replace(
                    statements=model.statements + ytargettot, dependent_variables=dvs
                )
        elif uptype in ('CR', 'CRIB'):
            if 'drug_tot' in dv_types.keys():
                new_y = (central.amount + complex_comp.amount) / vc
                after = model.statements.after_odes
                after = after.reassign(y_symbol, new_y)
                model = model.replace(
                    statements=model.statements.before_odes + get_and_check_odes(model) + after
                )
            if 'complex' in dv_types.keys():
                y_complex = Expr.symbol("Y_COMPLEX")
                ycomplex = Assignment.create(y_complex, complex_comp.amount / vc)
                dvs = model.dependent_variables.replace(y_complex, dv_types['complex'])
                model = model.replace(
                    statements=model.statements + ycomplex, dependent_variables=dvs
                )
        elif uptype == 'WAGNER':
            if 'drug_tot' in dv_types.keys():
                new_y = central.amount / vc
                after = model.statements.after_odes
                after = after.reassign(y_symbol, new_y)
                model = model.replace(
                    statements=model.statements.before_odes + get_and_check_odes(model) + after
                )
            if 'complex' in dv_types.keys():
                y_complex = Expr.symbol("Y_COMPLEX")
                ycomplex = Assignment.create(y_complex, (central.amount - lafreef) / vc)
                dvs = model.dependent_variables.replace(y_complex, dv_types['complex'])
                model = model.replace(
                    statements=model.statements + ycomplex, dependent_variables=dvs
                )

        # Add proportional error model
        if Expr.symbol('Y_TARGET') in list(model.dependent_variables):
            model = set_proportional_error_model(model, dv=dv_types['target'])
        if Expr.symbol('Y_COMPLEX') in list(model.dependent_variables):
            model = set_proportional_error_model(model, dv=dv_types['complex'])
        if Expr.symbol('Y') in list(model.dependent_variables) and 'drug_tot' in dv_types.keys():
            model = set_proportional_error_model(model, dv=dv_types['drug_tot'])
        if Expr.symbol('Y_TOTTARGET') in list(model.dependent_variables):
            model = set_proportional_error_model(model, dv=dv_types['target_tot'])

        if model.dataset is not None:
            dvid = model.datainfo.find_single_column_name('dvid', default='DVID')
            dvs = [dv for dv in model.dependent_variables.values()]
            model = model.replace(dataset=model.dataset.query(f'{dvid} in @dvs'))
    return model.update_source()


def _create_parameters(model, names):
    symbs = []
    for name in names:
        symb = Expr.symbol(name)
        symbs.append(symb)
        model = add_individual_parameter(model, symb.name)
    return model, *symbs


def _create_compartment(cb, name: str):
    comp = Compartment.create(name=name)
    cb.add_compartment(comp)
    return comp


def _create_ksyn():
    ksyn = Expr.symbol('KSYN')
    ksyn_ass = Assignment.create(ksyn, Expr.symbol("R_0") * Expr.symbol("KDEG"))
    return ksyn, ksyn_ass


def _validate_dv_types(dv_types):
    # Make sure that values are unique
    assert len(dv_types.values()) == len(set(dv_types.values()))
    # Validate keys
    for key, value in dv_types.items():
        if key not in ['drug', 'target', 'complex', 'drug_tot', 'target_tot']:
            raise ValueError(
                f'Invalid dv_types key "{key}". Allowed keys are:'
                f' "drug", "target", "complex", "drug_tot" and "target_tot".'
            )
        if key not in ['drug', 'drug_tot'] and value == 1:
            raise ValueError('Only drug can have DVID = 1. Please choose another DVID.')


def _get_y_symbol(model):
    t = Expr.symbol("t")
    y_statement = model.statements.find_assignment("Y")
    if t in y_statement.free_symbols:
        return y_statement.symbol
    else:
        sset = model.statements.direct_dependencies(y_statement)
        for s in sset:
            if Expr.symbol("t") in s.free_symbols:
                return s.symbol
