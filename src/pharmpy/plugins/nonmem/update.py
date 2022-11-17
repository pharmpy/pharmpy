from __future__ import annotations

import re
import warnings
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, List, Mapping, Optional

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy, sympy_printing
from pharmpy.internals.parse import AttrTree
from pharmpy.internals.sequence.lcs import diff
from pharmpy.model import (
    Assignment,
    Bolus,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    Distribution,
    ExplicitODESystem,
    Infusion,
    ODESystem,
    Parameter,
    Parameters,
    RandomVariables,
    Statements,
    data,
)
from pharmpy.modeling import get_ids, simplify_expression

if TYPE_CHECKING:
    from .model import Model

from .nmtran_parser import NMTranControlStream
from .parameters import parameter_translation
from .parsing import get_zero_fix_rvs, parse_column_info
from .records import code_record
from .records.code_record import CodeRecord
from .records.etas_record import EtasRecord
from .records.factory import create_record
from .records.model_record import ModelRecord
from .records.omega_record import OmegaRecord
from .records.sizes_record import SizesRecord
from .records.theta_record import ThetaRecord
from .table import NONMEMTableFile, PhiTable


def update_description(model: Model):
    if model.description != model.internals._old_description:
        probrec = model.internals.control_stream.get_records('PROBLEM')[0]
        probrec.title = model.description


def update_parameters(model: Model, old: Parameters, new: Parameters):
    new_names = {p.name for p in new}
    old_names = {p.name for p in old}
    removed = old_names - new_names
    if removed:
        remove_records = []
        next_theta = 1
        for theta_record in model.internals.control_stream.get_records('THETA'):
            current_names = theta_record.name_map.keys()
            if removed >= current_names:
                remove_records.append(theta_record)
            elif not removed.isdisjoint(current_names):
                # one or more in the record
                theta_record.remove(removed & current_names)
                theta_record.renumber(next_theta)
                next_theta += len(theta_record)
            else:
                # keep all
                theta_record.renumber(next_theta)
                next_theta += len(theta_record)
        for sigma_record in model.internals.control_stream.get_records('SIGMA'):
            current_names = sigma_record.name_map.keys()
            if removed >= current_names:
                remove_records.append(sigma_record)
        model.internals.control_stream.remove_records(remove_records)

    renamed_params_list = []

    for p in new:
        name = p.name
        p_renamed = p
        if name not in old and name not in model.random_variables.parameter_names:
            # This is a new theta
            theta_number = get_next_theta(model)
            if re.match(r'THETA\(\d+\)', name):
                p_renamed = p.replace(name=f'THETA({theta_number})')
            record = create_theta_record(model, p_renamed)
            assert isinstance(record, ThetaRecord)
            record.add_nonmem_name(p_renamed.name, theta_number)

        renamed_params_list.append(p_renamed)

    renamed_params = Parameters(renamed_params_list)

    if renamed_params != new:
        model.parameters = renamed_params

    next_theta = 1
    for theta_record in model.internals.control_stream.get_records('THETA'):
        theta_record.update(new, next_theta)
        next_theta += len(theta_record)

    next_omega = 1
    previous_size = None
    for omega_record in model.internals.control_stream.get_records('OMEGA'):
        next_omega, previous_size = omega_record.update(new, next_omega, previous_size)

    next_sigma = 1
    previous_size = None
    for sigma_record in model.internals.control_stream.get_records('SIGMA'):
        next_sigma, previous_size = sigma_record.update(new, next_sigma, previous_size)


def update_random_variables(model: Model, old: RandomVariables, new: RandomVariables):
    if not hasattr(model, '_parameters'):
        model.parameters

    rec_dict = {}
    comment_dict = {}

    for omega_record in model.internals.control_stream.get_records(
        'OMEGA'
    ) + model.internals.control_stream.get_records('SIGMA'):
        comment_dict.update(omega_record.comment_map)
        current_names = omega_record.eta_map.keys()
        for name in current_names:
            rec_dict[name] = omega_record

    rvs_diff_eta = diff(old.etas, new.etas)
    rvs_diff_eps = diff(old.epsilons, new.epsilons)

    update_random_variable_records(model, rvs_diff_eta, rec_dict, comment_dict)
    update_random_variable_records(model, rvs_diff_eps, rec_dict, comment_dict)

    next_eta = 1
    for omega_record in model.internals.control_stream.get_records('OMEGA'):
        omega_record.renumber(next_eta)
        next_eta += len(omega_record)

    next_eps = 1
    for sigma_record in model.internals.control_stream.get_records('SIGMA'):
        sigma_record.renumber(next_eps)
        next_eps += len(sigma_record)


def update_random_variable_records(model: Model, rvs_diff, rec_dict, comment_dict):
    removed = []
    eta_number = 1
    number_of_records = 0

    rvs_diff = list(rvs_diff)

    rvs_removed = [RandomVariables.create(rvs).names for (op, rvs) in rvs_diff if op == -1]
    rvs_removed = [rv for sublist in rvs_removed for rv in sublist]

    for op, rvs in rvs_diff:
        if op == 1:
            if len(rvs) == 1:
                create_omega_single(model, rvs, eta_number, number_of_records, comment_dict)
            else:
                create_omega_block(model, rvs, eta_number, number_of_records, comment_dict)
            eta_number += len(rvs)
            number_of_records += 1
        elif op == -1:
            rvs_rec = list({rec_dict[name] for name in rvs.names})
            recs_to_remove = [rec for rec in rvs_rec if rec not in removed]
            if recs_to_remove:
                model.internals.control_stream.remove_records(recs_to_remove)
                removed += list(recs_to_remove)
        else:
            diag_rvs = get_diagonal(rvs, rec_dict)
            # Account for etas in diagonal
            if diag_rvs:
                # Create new diagonal record if any in record has been removed
                if any(rv in rvs_removed for rv in diag_rvs):
                    create_omega_single(model, rvs, eta_number, number_of_records, comment_dict)
                # If none has been removed and this rv is not the first one,
                #   the record index should not increase
                elif len(diag_rvs) > 1 and rvs != model.random_variables[diag_rvs[0]]:
                    continue
            eta_number += len(rvs)
            number_of_records += 1


def get_diagonal(rv: RandomVariables, rec_dict):
    rv_rec = rec_dict[rv.names[0]]
    etas_from_same_record = [eta for eta, rec in rec_dict.items() if rec == rv_rec]
    return etas_from_same_record if len(etas_from_same_record) >= 2 else None


def get_next_theta(model: Model):
    """Find the next available theta number"""
    next_theta = 1

    for theta_record in model.internals.control_stream.get_records('THETA'):
        thetas = theta_record.parameters(next_theta)
        next_theta += len(thetas)

    return next_theta


def create_theta_record(model: Model, param: Parameter):
    param_str = '$THETA  '

    if param.init == 0.0:
        init = 0
    else:
        init = param.init

    if param.lower == 0.0:
        lower = 0
    else:
        lower = param.lower

    if param.upper == 0.0:
        upper = 0
    else:
        upper = param.upper

    if upper < 1000000:
        if lower <= -1000000:
            param_str += f'(-INF,{init},{upper})'
        else:
            param_str += f'({lower},{init},{upper})'
    else:
        if lower <= -1000000:
            param_str += f'{init}'
        else:
            param_str += f'({lower},{init})'
    if param.fix:
        param_str += ' FIX'
    param_str += '\n'
    record = model.internals.control_stream.insert_record(param_str)
    return record


def create_omega_single(
    model: Model, rv: Distribution, eta_number: int, record_number: int, comment_dict
):
    rvs, pset = model.random_variables, model.parameters

    if rv.level == 'RUV':
        record_type = 'SIGMA'
    else:
        record_type = 'OMEGA'

    variance_param = pset[rv.parameter_names[0]]

    if rv.level == 'IOV':
        param_str = f'${record_type}  BLOCK(1)'
        first_iov = next(filter(lambda iov: iov.parameter_names == rv.parameter_names, rvs.iov))
        if rv == first_iov:
            param_str += f'\n{variance_param.init}'
        else:
            param_str += ' SAME'
    else:
        param_str = f'${record_type}  {variance_param.init}'

    if not re.match(r'(OMEGA|SIGMA)\(\d+,\d+\)', variance_param.name):
        param_str += f' ; {variance_param.name}'
    elif comment_dict and variance_param.name in comment_dict.keys():
        param_str += f' ; {comment_dict[variance_param.name]}'

    record = insert_omega_record(model, f'{param_str}\n', record_number, record_type)

    record.comment_map = comment_dict
    record.eta_map = {rv.names[0]: eta_number}
    record.name_map = {variance_param.name: (eta_number, eta_number)}


def create_omega_block(
    model: Model, distribution: Distribution, eta_number: int, record_number: int, comment_dict
):
    rvs = RandomVariables.create([distribution])
    cm = rvs.covariance_matrix
    param_str = f'$OMEGA BLOCK({cm.shape[0]})'

    rv = rvs[0]

    if rv.level == 'IOV' and rv != next(
        filter(lambda iov: iov.parameter_names == rv.parameter_names, model.random_variables.iov)
    ):
        param_str += ' SAME\n'

    else:
        param_str += '\n'
        for row in range(cm.shape[0]):
            for col in range(row + 1):
                elem = cm.row(row).col(col)
                name = str(elem[0])
                omega = model.parameters[name]
                param_str += f'{omega.init}'.upper()

                if not re.match(r'OMEGA\(\d+,\d+\)', omega.name):
                    param_str += f'\t; {omega.name}'
                elif comment_dict and omega.name in comment_dict:
                    param_str += f'\t; {comment_dict[omega.name]}'

                param_str += '\n'

            param_str = f'{param_str.rstrip()}\n'

    eta_map, name_variance = {}, {}

    for rv_name in rvs.names:
        variance_param = rvs[rv_name].get_variance(rv_name)
        eta_map[rv_name] = eta_number
        name_variance[variance_param.name] = (eta_number, eta_number)
        eta_number += 1

    rv_combinations = [
        (rv1, rv2) for idx, rv1 in enumerate(rvs.names) for rv2 in rvs.names[idx + 1 :]
    ]
    name_covariance = {
        rvs.get_covariance(rv1, rv2).name: (eta_map[rv2], eta_map[rv1])
        for rv1, rv2 in rv_combinations
    }

    record = insert_omega_record(model, param_str, record_number, 'OMEGA')

    record.comment_map = comment_dict
    record.eta_map = eta_map
    record.name_map = {**name_variance, **name_covariance}


def insert_omega_record(model: Model, param_str: str, record_number: int, record_type: str):
    records = model.internals.control_stream.records
    tprecs = model.internals.control_stream.get_records(record_type)
    if tprecs:
        index = records.index(tprecs[0])
        record = model.internals.control_stream.insert_record(param_str, index + record_number)
    else:
        record = model.internals.control_stream.insert_record(param_str)
    assert isinstance(record, OmegaRecord)
    return record


def update_ode_system(model: Model, old: Optional[ODESystem], new: ODESystem):
    """Update ODE system

    Handle changes from CompartmentSystem to ExplicitODESystem
    """
    if old is None:
        old = CompartmentalSystem(CompartmentalSystemBuilder())

    update_lag_time(model, old, new)

    if isinstance(old, CompartmentalSystem) and isinstance(new, ExplicitODESystem):
        to_des(model, new)
    elif isinstance(old, ExplicitODESystem) and isinstance(new, CompartmentalSystem):
        # Stay with $DES for now
        update_des(model, old, new)
    elif isinstance(old, CompartmentalSystem) and isinstance(new, CompartmentalSystem):
        if isinstance(new.dosing_compartment.dose, Bolus) and 'RATE' in model.datainfo.names:
            df = model.dataset.drop(columns=['RATE'])
            model.dataset = df

        advan, trans = new_advan_trans(model)
        pk_param_conversion(model, advan=advan, trans=trans)
        add_needed_pk_parameters(model, advan, trans)
        update_subroutines_record(model, advan, trans)
        update_model_record(model, advan)

    update_infusion(model, old)

    force_des(model, new)


def is_nonlinear_odes(model: Model):
    """Check if ode system is nonlinear"""
    odes = model.statements.ode_system
    assert isinstance(odes, CompartmentalSystem)
    M = odes.compartmental_matrix
    return odes.t in M.free_symbols


def update_infusion(model: Model, old: ODESystem):
    statements = model.statements
    new = statements.ode_system
    assert new is not None
    old = old.to_compartmental_system()
    new = new.to_compartmental_system()
    if isinstance(new.dosing_compartment.dose, Infusion) and not statements.find_assignment('D1'):
        # Handle direct moving of Infusion dose
        statements.subs({'D2': 'D1'})

    if isinstance(new.dosing_compartment.dose, Infusion) and isinstance(
        old.dosing_compartment.dose, Bolus
    ):
        dose = new.dosing_compartment.dose
        if dose.rate is None:
            # FIXME: Not always D1 here!
            ass = Assignment(sympy.Symbol('D1'), dose.duration)
            cb = CompartmentalSystemBuilder(new)
            cb.set_dose(new.dosing_compartment, Infusion(dose.amount, ass.symbol))
            model.statements = (
                model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
            )
        else:
            raise NotImplementedError("First order infusion rate is not yet supported")
        model.statements = (
            model.statements.before_odes
            + ass
            + model.statements.ode_system
            + model.statements.after_odes
        )
        df = model.dataset.copy()
        rate = np.where(df['AMT'] == 0, 0.0, -2.0)
        df['RATE'] = rate
        # FIXME: Adding at end for now. Update $INPUT cannot yet handle adding in middle
        # df.insert(list(df.columns).index('AMT') + 1, 'RATE', rate)
        model.dataset = df


def update_des(model: Model, old: ExplicitODESystem, new: ODESystem):
    """Update where both old and new should be explicit ODE systems"""
    pass


def force_des(model: Model, odes: ODESystem):
    """Switch to $DES if necessary"""
    if isinstance(odes, ExplicitODESystem):
        return

    amounts = {sympy.Function(amt.name)(sympy.Symbol('t')) for amt in odes.amounts}
    if odes.atoms(sympy.Function) & amounts:
        explicit_odes(model)
        new = model.statements.ode_system
        assert new is not None
        to_des(model, new)


def explicit_odes(model: Model):
    """Convert model from compartmental system to explicit ODE system
    or do nothing if it already has an explicit ODE system

    Parameters
    ----------
    model : Model
        Pharmpy model

    Return
    ------
    Model
        Reference to same model
    """
    statements = model.statements
    odes = statements.ode_system
    if isinstance(odes, CompartmentalSystem):
        new = odes.to_explicit_system()
        model.statements = statements.before_odes + new + statements.after_odes
    return model


def to_des(model: Model, new: ODESystem):
    old_des = model.internals.control_stream.get_records('DES')
    model.internals.control_stream.remove_records(old_des)
    subs = model.internals.control_stream.get_records('SUBROUTINES')[0]
    subs.remove_option_startswith('TRANS')
    subs.remove_option_startswith('ADVAN')
    subs.remove_option('TOL')
    step = model.estimation_steps[0]
    solver = step.solver
    if solver:
        advan = solver_to_advan(solver)
        if not isinstance(new, ExplicitODESystem):
            new = new.to_explicit_system()
        subs.append_option(advan)
    else:
        subs.append_option('ADVAN13')
    if not subs.has_option('TOL'):
        subs.append_option('TOL', '9')
    des = model.internals.control_stream.insert_record('$DES\nDUMMY=0\n')
    assert isinstance(des, CodeRecord)
    des.from_odes(new)
    model.internals.control_stream.remove_records(
        model.internals.control_stream.get_records('MODEL')
    )
    mod = model.internals.control_stream.insert_record('$MODEL\n')
    assert isinstance(mod, ModelRecord)
    for eq, ic in zip(new.odes[:-1], list(new.ics.keys())[:-1]):
        name = eq.lhs.args[0].name[2:]
        if new.ics[ic] != 0:
            dose = True
        else:
            dose = False
        mod.add_compartment(name, dosing=dose)


def update_statements(model: Model, old: Statements, new: Statements, trans):
    trans['NaN'] = int(data.conf.na_rep)
    main_statements = Statements()
    error_statements = Statements()

    new_odes = new.ode_system
    if new_odes is not None:
        old_odes = old.ode_system
        if new_odes != old_odes:
            colnames, drop, _, _ = parse_column_info(model.internals.control_stream)
            col_dropped = dict(zip(colnames, drop))
            if 'CMT' in col_dropped.keys() and not col_dropped['CMT']:
                warnings.warn(
                    'Compartment structure has been updated, CMT-column '
                    'in dataset might not be relevant anymore. Check '
                    'CMT-column or drop column'
                )
            update_ode_system(model, old_odes, new_odes)
        else:
            if len(model.estimation_steps) > 0:
                new_solver = model.estimation_steps[0].solver
            else:
                new_solver = None
            if isinstance(new_odes, ExplicitODESystem) or new_solver:
                old_solver = model.internals._old_estimation_steps[0].solver
                if new_solver != old_solver:
                    advan = solver_to_advan(new_solver)
                    subs = model.internals.control_stream.get_records('SUBROUTINES')[0]
                    subs.advan = advan

    main_statements = model.statements.before_odes
    error_statements = model.statements.after_odes

    rec = model.internals.control_stream.get_pred_pk_record()
    rec.rvs, rec.trans = model.random_variables, trans
    rec.statements = main_statements.subs(trans)

    error = model.internals.control_stream.get_error_record()
    if not error and len(error_statements) > 0:
        model.internals.control_stream.insert_record('$ERROR\n')
    if error:
        if (
            len(error_statements) > 0
            and isinstance((s := error_statements[0]), Assignment)
            and s.symbol.name == 'F'
        ):
            error_statements = error_statements[1:]  # Remove the link statement
        error.rvs, error.trans = model.random_variables, trans
        new_ode_system = new.ode_system
        if new_ode_system is not None:
            amounts = list(new_ode_system.amounts)
            for i, amount in enumerate(amounts, start=1):
                trans[amount] = sympy.Symbol(f"A({i})")
        error.statements = error_statements.subs(trans)
        error.is_updated = True

    rec.is_updated = True


def update_lag_time(model: Model, old: ODESystem, new: ODESystem):
    old = old.to_compartmental_system()
    new = new.to_compartmental_system()
    new_dosing = new.dosing_compartment
    new_lag_time = new_dosing.lag_time
    try:
        old_lag_time = old.dosing_compartment.lag_time
    except ValueError:
        old_lag_time = 0
    if new_lag_time != old_lag_time and new_lag_time != 0:
        ass = Assignment(sympy.Symbol('ALAG1'), new_lag_time)
        cb = CompartmentalSystemBuilder(new)
        cb.set_lag_time(new_dosing, ass.symbol)
        model.statements = (
            model.statements.before_odes
            + ass
            + CompartmentalSystem(cb)
            + model.statements.after_odes
        )


def new_compartmental_map(cs: CompartmentalSystem, oldmap: Mapping[str, int]):
    """Create compartmental map for updated model
    cs - new compartmental system
    old - old compartmental map

    Can handle compartments from dosing to central, peripherals and output
    """
    comp = cs.dosing_compartment
    central = cs.central_compartment
    i = 1
    compmap = {}
    while True:
        compmap[comp.name] = i
        i += 1
        if comp == central:
            break
        comp, _ = cs.get_compartment_outflows(comp)[0]

    peripherals = cs.peripheral_compartments
    for p in peripherals:
        compmap[p.name] = i
        i += 1

    diff = len(cs) - len(oldmap)
    for name in cs.compartment_names:
        if name not in compmap.keys():
            compmap[name] = oldmap[name] + diff
    return compmap


def create_compartment_remap(oldmap, newmap):
    """Creates a map from old compartment number to new compartment number

    For all compartments where remapping is needed
    Assume that compartments with same name in new and old are the same compartments
    """
    remap = {}
    for name, number in oldmap.items():
        if name in newmap:
            remap[number] = newmap[name]
    return remap


def pk_param_conversion(model: Model, advan, trans):
    """Conversion map for pk parameters for removed or added compartment"""
    all_subs = model.internals.control_stream.get_records('SUBROUTINES')
    if not all_subs:
        return
    subs = all_subs[0]
    from_advan = subs.advan
    statements = model.statements
    cs = statements.ode_system
    assert isinstance(cs, CompartmentalSystem)
    oldmap = model.internals._compartment_map
    assert oldmap is not None
    newmap = new_compartmental_map(cs, oldmap)
    remap = create_compartment_remap(oldmap, newmap)
    d = {}
    for old, new in remap.items():
        d[sympy.Symbol(f'S{old}')] = sympy.Symbol(f'S{new}')
        d[sympy.Symbol(f'F{old}')] = sympy.Symbol(f'F{new}')
        # FIXME: R, D and ALAG should be moved with dose compartment
        # d[sympy.Symbol(f'R{old}')] = sympy.Symbol(f'R{new}')
        # d[sympy.Symbol(f'D{old}')] = sympy.Symbol(f'D{new}')
        # d[sympy.Symbol(f'ALAG{old}')] = sympy.Symbol(f'ALAG{new}')
        d[sympy.Symbol(f'A({old})')] = sympy.Symbol(f'A({new})')
    if from_advan == 'ADVAN5' or from_advan == 'ADVAN7':
        reverse_map = {v: k for k, v in newmap.items()}
        for i, j in product(range(1, len(oldmap)), range(0, len(oldmap))):
            if i != j and (i in remap and (j in remap or j == 0)):
                if i in remap:
                    to_i = remap[i]
                else:
                    to_i = i
                if j in remap:
                    to_j = remap[j]
                else:
                    to_j = j
                outind = to_j if to_j != 0 else len(cs)
                from_comp = cs.find_compartment(reverse_map[to_i])
                to_comp = cs.find_compartment(reverse_map[outind])
                if cs.get_flow(from_comp, to_comp) is not None:
                    d[sympy.Symbol(f'K{i}{j}')] = sympy.Symbol(f'K{to_i}{to_j}')
                    d[sympy.Symbol(f'K{i}T{j}')] = sympy.Symbol(f'K{to_i}T{to_j}')
        if advan == 'ADVAN3':
            n = len(oldmap)
            for i in range(1, n):
                d[sympy.Symbol(f'K{i}0')] = sympy.Symbol('K')
                d[sympy.Symbol(f'K{i}T0')] = sympy.Symbol('K')
                d[sympy.Symbol(f'K{i}{n}')] = sympy.Symbol('K')
                d[sympy.Symbol(f'K{i}T{n}')] = sympy.Symbol('K')
    elif from_advan == 'ADVAN1':
        if advan == 'ADVAN3' or advan == 'ADVAN11':
            d[sympy.Symbol('V')] = sympy.Symbol('V1')
        elif advan == 'ADVAN4' or advan == 'ADVAN12':
            d[sympy.Symbol('V')] = sympy.Symbol('V2')
    elif from_advan == 'ADVAN2':
        if advan == 'ADVAN3' and trans != 'TRANS1':
            d[sympy.Symbol('V')] = sympy.Symbol('V1')
        elif advan == 'ADVAN4' and trans != 'TRANS1':
            d[sympy.Symbol('V')] = sympy.Symbol('V2')
    elif from_advan == 'ADVAN3':
        if advan == 'ADVAN1':
            if trans == 'TRANS2':
                d[sympy.Symbol('V1')] = sympy.Symbol('V')
        elif advan == 'ADVAN4':
            if trans == 'TRANS4':
                d[sympy.Symbol('V1')] = sympy.Symbol('V2')
                d[sympy.Symbol('V2')] = sympy.Symbol('V3')
            elif trans == 'TRANS6':
                d[sympy.Symbol('K21')] = sympy.Symbol('K32')
            else:  # TRANS1
                d[sympy.Symbol('K12')] = sympy.Symbol('K23')
                d[sympy.Symbol('K21')] = sympy.Symbol('K32')
        elif advan == 'ADVAN11':
            if trans == 'TRANS4':
                d.update({sympy.Symbol('Q'): sympy.Symbol('Q2')})
    elif from_advan == 'ADVAN4':
        if advan == 'ADVAN2':
            if trans == 'TRANS2':
                d[sympy.Symbol('V2')] = sympy.Symbol('V')
        if advan == 'ADVAN3':
            if trans == 'TRANS4':
                d.update(
                    {sympy.Symbol('V2'): sympy.Symbol('V1'), sympy.Symbol('V3'): sympy.Symbol('V2')}
                )
            elif trans == 'TRANS6':
                d.update({sympy.Symbol('K32'): sympy.Symbol('K21')})
            else:  # TRANS1
                d.update(
                    {
                        sympy.Symbol('K23'): sympy.Symbol('K12'),
                        sympy.Symbol('K32'): sympy.Symbol('K21'),
                    }
                )
        elif advan == 'ADVAN12':
            if trans == 'TRANS4':
                d.update({sympy.Symbol('Q'): sympy.Symbol('Q3')})
    elif from_advan == 'ADVAN11':
        if advan == 'ADVAN1':
            if trans == 'TRANS2':
                d[sympy.Symbol('V1')] = sympy.Symbol('V')
        elif advan == 'ADVAN3':
            if trans == 'TRANS4':
                d[sympy.Symbol('Q2')] = sympy.Symbol('Q')
        elif advan == 'ADVAN12':
            if trans == 'TRANS4':
                d.update(
                    {
                        sympy.Symbol('V1'): sympy.Symbol('V2'),
                        sympy.Symbol('Q2'): sympy.Symbol('Q3'),
                        sympy.Symbol('V2'): sympy.Symbol('V3'),
                        sympy.Symbol('Q3'): sympy.Symbol('Q4'),
                        sympy.Symbol('V3'): sympy.Symbol('V4'),
                    }
                )
            elif trans == 'TRANS6':
                d.update(
                    {
                        sympy.Symbol('K31'): sympy.Symbol('K42'),
                        sympy.Symbol('K21'): sympy.Symbol('K32'),
                    }
                )
            else:  # TRANS1
                d.update(
                    {
                        sympy.Symbol('K12'): sympy.Symbol('K23'),
                        sympy.Symbol('K21'): sympy.Symbol('K32'),
                        sympy.Symbol('K13'): sympy.Symbol('K24'),
                        sympy.Symbol('K31'): sympy.Symbol('K42'),
                    }
                )
    elif from_advan == 'ADVAN12':
        if advan == 'ADVAN2':
            if trans == 'TRANS2':
                d[sympy.Symbol('V2')] = sympy.Symbol('V')
        elif advan == 'ADVAN4':
            if trans == 'TRANS4':
                d[sympy.Symbol('Q3')] = sympy.Symbol('Q')
        elif advan == 'ADVAN11':
            if trans == 'TRANS4':
                d.update(
                    {
                        sympy.Symbol('V2'): sympy.Symbol('V1'),
                        sympy.Symbol('Q3'): sympy.Symbol('Q2'),
                        sympy.Symbol('V3'): sympy.Symbol('V2'),
                        sympy.Symbol('Q4'): sympy.Symbol('Q3'),
                        sympy.Symbol('V4'): sympy.Symbol('V3'),
                    }
                )
            elif trans == 'TRANS6':
                d.update(
                    {
                        sympy.Symbol('K42'): sympy.Symbol('K31'),
                        sympy.Symbol('K32'): sympy.Symbol('K21'),
                    }
                )
            else:  # TRANS1
                d.update(
                    {
                        sympy.Symbol('K23'): sympy.Symbol('K12'),
                        sympy.Symbol('K32'): sympy.Symbol('K21'),
                        sympy.Symbol('K24'): sympy.Symbol('K13'),
                        sympy.Symbol('K42'): sympy.Symbol('K31'),
                    }
                )
    if advan == 'ADVAN5' or advan == 'ADVAN7' and from_advan not in ('ADVAN5', 'ADVAN7'):
        n = len(newmap)
        d[sympy.Symbol('K')] = sympy.Symbol(f'K{n-1}0')
    model.statements = statements.subs(d)


def new_advan_trans(model: Model):
    """Decide which new advan and trans to be used"""
    all_subs = model.internals.control_stream.get_records('SUBROUTINES')
    if all_subs:
        subs = all_subs[0]
        oldtrans = subs.get_option_startswith('TRANS')
    else:
        oldtrans = None
    statements = model.statements
    odes = model.statements.ode_system
    assert isinstance(odes, CompartmentalSystem)
    nonlin = is_nonlinear_odes(model)
    if nonlin:
        advan = 'ADVAN13'
    elif len(odes) > 5 or odes.get_n_connected(odes.central_compartment) != len(odes) - 1:
        advan = 'ADVAN5'
    elif len(odes) == 2:
        advan = 'ADVAN1'
    elif len(odes) == 3 and odes.find_depot(statements):
        advan = 'ADVAN2'
    elif len(odes) == 3:
        advan = 'ADVAN3'
    elif len(odes) == 4 and odes.find_depot(statements):
        advan = 'ADVAN4'
    elif len(odes) == 4:
        advan = 'ADVAN11'
    else:  # len(odes) == 5
        advan = 'ADVAN12'

    if nonlin:
        trans = None
    elif oldtrans == 'TRANS1':
        trans = oldtrans
    elif oldtrans == 'TRANS2':
        if advan in ['ADVAN1', 'ADVAN2']:
            trans = oldtrans
        elif advan in ['ADVAN3', 'ADVAN4', 'ADVAN11', 'ADVAN12']:
            trans = 'TRANS4'
        else:
            trans = 'TRANS1'
    elif oldtrans == 'TRANS3':
        if advan in ['ADVAN3', 'ADVAN4']:
            trans = oldtrans
        elif advan in ['ADVAN11', 'ADVAN12']:
            trans = 'TRANS4'
        elif advan in ['ADVAN1', 'ADVAN2']:
            trans = 'TRANS2'
        else:
            trans = 'TRANS1'
    elif oldtrans == 'TRANS4':
        if advan in ['ADVAN3', 'ADVAN4', 'ADVAN11', 'ADVAN12']:
            trans = oldtrans
        elif advan in ['ADVAN1', 'ADVAN2']:
            trans = 'TRANS2'
        else:
            trans = 'TRANS1'
    elif oldtrans is None:
        output = odes.output_compartment
        central = odes.central_compartment
        elimination_rate = odes.get_flow(central, output)
        assert elimination_rate is not None
        num, den = elimination_rate.as_numer_denom()
        if num.is_Symbol and den.is_Symbol:
            if advan in ['ADVAN1', 'ADVAN2']:
                trans = 'TRANS2'
            else:
                trans = 'TRANS4'
        else:
            trans = 'TRANS1'
    else:
        trans = 'TRANS1'

    return advan, trans


def update_subroutines_record(model: Model, advan, trans):
    """Update $SUBROUTINES with new advan and trans"""
    all_subs = model.internals.control_stream.get_records('SUBROUTINES')
    if not all_subs:
        content = f'$SUBROUTINES {advan} {trans}\n'
        model.internals.control_stream.insert_record(content)
        return
    subs = all_subs[0]
    oldadvan = subs.advan
    oldtrans = subs.trans

    if advan != oldadvan:
        subs.replace_option(oldadvan, advan)
    if trans != oldtrans:
        if trans is None:
            subs.remove_option_startswith('TRANS')
        else:
            subs.replace_option(oldtrans, trans)


def update_model_record(model: Model, advan):
    """Update $MODEL"""
    odes = model.statements.ode_system
    if not isinstance(odes, CompartmentalSystem):
        return

    oldmap = model.internals._compartment_map
    if oldmap is None:
        return
    newmap = new_compartmental_map(odes, oldmap)

    if advan in ['ADVAN1', 'ADVAN2', 'ADVAN3', 'ADVAN4', 'ADVAN10', 'ADVAN11', 'ADVAN12']:
        model.internals.control_stream.remove_records(
            model.internals.control_stream.get_records('MODEL')
        )
    else:
        if oldmap != newmap or model.estimation_steps[0].solver:
            model.internals.control_stream.remove_records(
                model.internals.control_stream.get_records('MODEL')
            )
            mod = model.internals.control_stream.insert_record('$MODEL\n')
            assert isinstance(mod, ModelRecord)
            output_name = odes.output_compartment.name
            comps = {v: k for k, v in newmap.items() if k != output_name}
            i = 1
            while True:
                if i not in comps:
                    break
                if i == 1:
                    mod.add_compartment(comps[i], dosing=True)
                else:
                    mod.add_compartment(comps[i], dosing=False)
                i += 1
    model.internals._compartment_map = newmap


def add_needed_pk_parameters(model: Model, advan, trans):
    """Add missing pk parameters that NONMEM needs"""
    statements = model.statements
    odes = statements.ode_system
    assert isinstance(odes, CompartmentalSystem)
    if advan == 'ADVAN2' or advan == 'ADVAN4' or advan == 'ADVAN12':
        if not statements.find_assignment('KA'):
            comp, rate = odes.get_compartment_outflows(odes.find_depot(statements))[0]
            ass = Assignment(sympy.Symbol('KA'), rate)
            if rate != ass.symbol:
                cb = CompartmentalSystemBuilder(odes)
                cb.add_flow(odes.find_depot(statements), comp, ass.symbol)
                model.statements = (
                    statements.before_odes + ass + CompartmentalSystem(cb) + statements.after_odes
                )
    if advan in ['ADVAN1', 'ADVAN2'] and trans == 'TRANS2':
        central = odes.central_compartment
        output = odes.output_compartment
        add_parameters_ratio(model, 'CL', 'V', central, output)
    elif advan == 'ADVAN3' and trans == 'TRANS4':
        central = odes.central_compartment
        output = odes.output_compartment
        peripheral = odes.peripheral_compartments[0]
        add_parameters_ratio(model, 'CL', 'V1', central, output)
        add_parameters_ratio(model, 'Q', 'V2', peripheral, central)
        add_parameters_ratio(model, 'Q', 'V1', central, peripheral)
    elif advan == 'ADVAN4':
        central = odes.central_compartment
        output = odes.output_compartment
        peripheral = odes.peripheral_compartments[0]
        if trans == 'TRANS1':
            rate1 = odes.get_flow(central, peripheral)
            add_rate_assignment_if_missing(model, 'K23', rate1, central, peripheral)
            rate2 = odes.get_flow(peripheral, central)
            add_rate_assignment_if_missing(model, 'K32', rate2, peripheral, central)
        if trans == 'TRANS4':
            add_parameters_ratio(model, 'CL', 'V2', central, output)
            add_parameters_ratio(model, 'Q', 'V3', peripheral, central)
    elif advan == 'ADVAN12' and trans == 'TRANS4':
        central = odes.central_compartment
        output = odes.output_compartment
        peripheral1 = odes.peripheral_compartments[0]
        peripheral2 = odes.peripheral_compartments[1]
        add_parameters_ratio(model, 'CL', 'V2', central, output)
        add_parameters_ratio(model, 'Q3', 'V3', peripheral1, central)
        add_parameters_ratio(model, 'Q4', 'V4', peripheral2, central)
    elif advan == 'ADVAN11' and trans == 'TRANS4':
        central = odes.central_compartment
        output = odes.output_compartment
        peripheral1 = odes.peripheral_compartments[0]
        peripheral2 = odes.peripheral_compartments[1]
        add_parameters_ratio(model, 'CL', 'V1', central, output)
        add_parameters_ratio(model, 'Q2', 'V2', peripheral1, central)
        add_parameters_ratio(model, 'Q3', 'V3', peripheral2, central)
    elif advan == 'ADVAN5' or advan == 'ADVAN7':
        oldmap = model.internals._compartment_map
        assert oldmap is not None
        newmap = new_compartmental_map(odes, oldmap)
        for source in newmap.keys():
            if source == len(newmap):  # NOTE Skip last
                continue
            for dest in newmap.keys():
                if source != dest:  # NOTE Skip same
                    source_comp = odes.find_compartment(source)
                    dest_comp = odes.find_compartment(dest)
                    rate = odes.get_flow(source_comp, dest_comp)
                    if rate is not None:
                        assert isinstance(source_comp, Compartment)
                        assert isinstance(dest_comp, Compartment)
                        sn = newmap[source]
                        dn = newmap[dest]
                        if len(str(sn)) > 1 or len(str(dn)) > 1:
                            t = 'T'
                        else:
                            t = ''
                        names = [f'K{sn}{dn}', f'K{sn}T{dn}']
                        if dn == len(newmap):
                            names += [f'K{sn}0', f'K{sn}T0']
                            param = f'K{sn}{t}{0}'
                        else:
                            param = f'K{sn}{t}{dn}'
                        add_rate_assignment_if_missing(
                            model, param, rate, source_comp, dest_comp, synonyms=names
                        )


def add_parameters_ratio(model: Model, numpar, denompar, source, dest):
    statements = model.statements
    if not statements.find_assignment(numpar) or not statements.find_assignment(denompar):
        odes = statements.ode_system
        assert isinstance(odes, CompartmentalSystem)
        rate = odes.get_flow(source, dest)
        assert rate is not None
        numer, denom = rate.as_numer_denom()
        par1 = Assignment(sympy.Symbol(numpar), numer)
        par2 = Assignment(sympy.Symbol(denompar), denom)
        new_statement1 = Statements()
        new_statement2 = Statements()
        if rate != par1.symbol / par2.symbol:
            if not statements.find_assignment(numpar):
                odes = odes.subs({numer: sympy.Symbol(numpar)})
                new_statement1 = par1
            if not statements.find_assignment(denompar):
                odes = odes.subs({denom: sympy.Symbol(denompar)})
                new_statement2 = par2
        cb = CompartmentalSystemBuilder(odes)
        cb.add_flow(source, dest, par1.symbol / par2.symbol)
        model.statements = (
            statements.before_odes
            + new_statement1
            + new_statement2
            + CompartmentalSystem(cb)
            + statements.after_odes
        )


def define_parameter(
    model: Model, name: str, value: sympy.Expr, synonyms: Optional[List[str]] = None
):
    """Define a parameter in statments if not defined
    Update if already defined as other value
    return True if new assignment was added
    """
    if synonyms is None:
        synonyms = [name]
    for syn in synonyms:
        i = model.statements.find_assignment_index(syn)
        if i is not None:
            ass = model.statements[i]
            assert isinstance(ass, Assignment)
            if value != ass.expression and value != sympy.Symbol(name):
                replacement_ass = Assignment(ass.symbol, value)
                model.statements = (
                    model.statements[:i] + replacement_ass + model.statements[i + 1 :]
                )
            return False
    new_ass = Assignment(sympy.Symbol(name), value)
    model.statements = (
        model.statements.before_odes
        + new_ass
        + model.statements.ode_system
        + model.statements.after_odes
    )
    return True


def add_rate_assignment_if_missing(
    model: Model,
    name: str,
    value: sympy.Expr,
    source: Compartment,
    dest: Compartment,
    synonyms: Optional[List[str]] = None,
):
    added = define_parameter(model, name, value, synonyms=synonyms)
    if added:
        cb = CompartmentalSystemBuilder(model.statements.ode_system)
        cb.add_flow(source, dest, sympy.Symbol(name))
        model.statements = (
            model.statements.before_odes + CompartmentalSystem(cb) + model.statements.after_odes
        )


def update_abbr_record(model: Model, rv_trans):
    trans = {}
    if not rv_trans:
        return trans
    # Remove already abbreviated symbols
    # FIXME: Doesn't update if name has changed
    abbr_recs = model.internals.control_stream.get_records('ABBREVIATED')
    to_delete = set().union(*(rec.replace.values() for rec in abbr_recs))
    rv_trans = {tk: tv for tk, tv in rv_trans.items() if tv.name not in to_delete}
    if not rv_trans:
        return trans

    for rv in model.random_variables.names:
        rv_symb = sympy.Symbol(rv)
        abbr_pattern = re.match(r'ETA_(\w+)', rv)
        if abbr_pattern and '_' not in abbr_pattern.group(1):
            parameter = abbr_pattern.group(1)
            nonmem_name = rv_trans[rv_symb]
            abbr_name = f'ETA({parameter})'
            trans[rv_symb] = sympy.Symbol(abbr_name)
            abbr_record = f'$ABBR REPLACE {abbr_name}={nonmem_name}\n'
            model.internals.control_stream.insert_record(abbr_record)
        elif not re.match(r'(ETA|EPS)\([0-9]\)', rv):
            warnings.warn(
                f'Not valid format of name {rv}, falling back to NONMEM name. If custom name, '
                f'follow the format "ETA_X" to get "ETA(X)" in $ABBR.'
            )
    return trans


def update_estimation(model: Model):
    new = model.estimation_steps
    try:
        old = model.internals._old_estimation_steps
    except AttributeError:
        old = []
        # Add SADDLE_RESET=1 if model did not have $EST before
        if 'SADDLE_RESET' not in new[-1].tool_options:
            new[-1].tool_options['SADDLE_RESET'] = 1
    if old == new:
        return

    delta = code_record.diff(old, new)
    old_records = model.internals.control_stream.get_records('ESTIMATION')
    i = 0
    new_records = []

    prev = (None, None)
    for op, est in delta:
        if op == 1:
            est_code = '$ESTIMATION'
            protected_attributes = []
            if est.method == 'FO':
                method = 'ZERO'
            elif est.method == 'FOCE':
                method = 'COND'
            else:
                method = est.method
            est_code += f' METHOD={method}'
            if est.laplace:
                est_code += ' LAPLACE'
                protected_attributes += ['LAPLACE']
            if est.interaction:
                est_code += ' INTER'
                protected_attributes += ['INTERACTION', 'INTER']
            if est.evaluation:
                if est.method == 'FO' or est.method == 'FOCE':
                    est_code += ' MAXEVAL=0'
                    protected_attributes += ['MAXEVALS', 'MAXEVAL']
                else:
                    est_code += ' EONLY=1'
                    protected_attributes += ['EONLY']
            if est.maximum_evaluations is not None:
                op_prev, est_prev = prev
                if not (
                    est.method.startswith('FO')
                    and op_prev == -1
                    and est.evaluation
                    and est_prev is not None
                    and not est_prev.evaluation
                    and est_prev.maximum_evaluations == est.maximum_evaluations
                ):
                    if set(protected_attributes).intersection({'MAXEVALS', 'MAXEVAL'}):
                        raise ValueError('MAXEVAL already set by evaluation=True')
                    est_code += f' MAXEVAL={est.maximum_evaluations}'
                protected_attributes += ['MAXEVALS', 'MAXEVAL']
            if est.isample is not None:
                est_code += f' ISAMPLE={est.isample}'
            if est.niter is not None:
                est_code += f' NITER={est.niter}'
            if est.auto is not None:
                est_code += f' AUTO={int(est.auto)}'
            if est.keep_every_nth_iter is not None:
                est_code += f' PRINT={est.keep_every_nth_iter}'
            if est.tool_options:
                option_names = set(est.tool_options.keys())
                overlapping_attributes = set(protected_attributes).intersection(option_names)
                if overlapping_attributes:
                    overlapping_attributes_str = ', '.join(list(overlapping_attributes))
                    raise ValueError(
                        f'{overlapping_attributes_str} already set as attribute in '
                        f'estimation method object'
                    )
                options_code = ' '.join(
                    [
                        f'{key}={value}'.upper() if value else str(key).upper()
                        for key, value in est.tool_options.items()
                    ]
                )
                est_code += f' {options_code}'
            est_code += '\n'
            newrec = create_record(est_code)
            new_records.append(newrec)
        elif op == -1:
            i += 1
        else:
            new_records.append(old_records[i])
            i += 1
        prev = (op, est)

    if old_records:
        model.internals.control_stream.replace_records(old_records, new_records)
    else:
        for rec in new_records:
            model.internals.control_stream.insert_record(str(rec))

    old_cov = False
    for est in old:
        old_cov |= est.cov
    new_cov = False
    for est in new:
        new_cov |= est.cov
    if not old_cov and new_cov:
        # Add $COV
        last_est_rec = model.internals.control_stream.get_records('ESTIMATION')[-1]
        idx_cov = model.internals.control_stream.records.index(last_est_rec)
        model.internals.control_stream.insert_record('$COVARIANCE\n', at_index=idx_cov + 1)
    elif old_cov and not new_cov:
        # Remove $COV
        covrecs = model.internals.control_stream.get_records('COVARIANCE')
        model.internals.control_stream.remove_records(covrecs)

    # Update $TABLE
    # Currently only adds if did not exist before
    cols = set()
    for estep in new:
        cols.update(estep.predictions)
        cols.update(estep.residuals)
    tables = model.internals.control_stream.get_records('TABLE')
    if not tables and cols:
        s = f'$TABLE {model.datainfo.id_column.name} {model.datainfo.idv_column.name} '
        s += f'{model.datainfo.dv_column.name} '
        s += f'{" ".join(cols)} FILE=mytab NOAPPEND NOPRINT'
        if any(id_val > 99999 for id_val in get_ids(model)):
            s += ' FORMAT=s1PE16.8'
        model.internals.control_stream.insert_record(s)
    model.internals._old_estimation_steps = new


def solver_to_advan(solver):
    if solver == 'LSODA':
        return 'ADVAN13'
    elif solver == 'CVODES':
        return 'ADVAN14'
    elif solver == 'DGEAR':
        return 'ADVAN8'
    elif solver == 'DVERK':
        return 'ADVAN6'
    elif solver == 'IDA':
        return 'ADVAN15'
    elif solver == 'LSODI':
        return 'ADVAN9'

    raise ValueError(solver)


def update_ccontra(model: Model, path=None, force=False):
    h = model.observation_transformation
    y = model.dependent_variable
    dhdy = sympy.diff(h, y)
    ll = -2 * sympy.log(dhdy)
    ll = ll.subs(y, sympy.Symbol('y', real=True, positive=True))
    ll = simplify_expression(model, ll)
    ll = ll.subs(sympy.Symbol('y', real=True, positive=True), y)

    tr = parameter_translation(
        model.internals.control_stream, reverse=True, remove_idempotent=True, as_symbols=True
    )
    ll = ll.subs(tr)
    h = h.subs(tr)

    # FIXME: break out into method to get path
    if path is None:
        path = Path('.')
    else:
        path = path.parent
    contr_path = path / f'{model.name}_contr.f90'
    ccontr_path = path / f'{model.name}_ccontra.f90'

    contr = """      subroutine contr (icall,cnt,ier1,ier2)
      double precision cnt
      call ncontr (cnt,ier1,ier2,l2r)
      return
      end
"""
    with open(contr_path, 'w') as fh:
        fh.write(contr)

    ccontr1 = """      subroutine ccontr (icall,c1,c2,c3,ier1,ier2)
      USE ROCM_REAL,   ONLY: theta=>THETAC,y=>DV_ITM2
      USE NM_INTERFACE,ONLY: CELS
      double precision c1,c2,c3,w,one,two
      dimension c2(:),c3(:,:)
      if (icall.le.1) return
      w=y(1)

"""

    ccontr2 = """
      call cels (c1,c2,c3,ier1,ier2)
      y(1)=w
"""

    ccontr3 = """
      return
      end
"""

    with open(ccontr_path, 'w') as fh:
        fh.write(ccontr1)
        e1 = sympy_printing.fortran.fcode(h.subs(y, sympy.Symbol('y(1)')), assign_to='y(1)')
        fh.write(e1)
        fh.write(ccontr2)
        e2 = sympy_printing.fortran.fcode(
            sympy.Symbol('c1') + ll.subs(y, sympy.Symbol('y(1)')), assign_to='c1'
        )
        fh.write(e2)
        fh.write(ccontr3)


def update_name_of_tables(control_stream: NMTranControlStream, new_name: str):
    m = re.search(r'.*?(\d+)$', new_name)
    if m:
        n = int(m.group(1))
        for table in control_stream.get_records('TABLE'):
            table_path = table.path
            table_name = table_path.stem
            m = re.search(r'(.*?)(\d+)$', table_name)
            if m:
                table_stem = m.group(1)
                new_table_name = f'{table_stem}{n}'
                table.path = table_path.parent / new_table_name


def update_sizes(model: Model):
    """Update $SIZES if needed"""
    all_sizes = model.internals.control_stream.get_records('SIZES')
    sizes = all_sizes[0] if all_sizes else create_record('$SIZES ')
    assert isinstance(sizes, SizesRecord)
    odes = model.statements.ode_system

    if odes is not None and isinstance(odes, CompartmentalSystem):
        n_compartments = len(odes)
        sizes.PC = n_compartments
    thetas = [p for p in model.parameters if p.symbol not in model.random_variables.free_symbols]
    sizes.LTH = len(thetas)

    if len(all_sizes) == 0 and len(str(sizes)) > 7:
        model.internals.control_stream.insert_record(str(sizes))


def update_input(model: Model):
    """Update $INPUT"""
    input_records = model.internals.control_stream.get_records("INPUT")
    _, drop, _, colnames = parse_column_info(model.internals.control_stream)
    keep = []
    i = 0
    for child in input_records[0].root.children:
        if child.rule != 'option':
            keep.append(child)
            continue

        if (colnames[i] is not None and (colnames[i] != model.datainfo[i].name)) or (
            not drop[i] and (model.datainfo[i].drop or model.datainfo[i].datatype == 'nmtran-date')
        ):
            dropped = model.datainfo[i].drop or model.datainfo[i].datatype == 'nmtran-date'
            anonymous = colnames[i] is None
            key = 'DROP' if anonymous and dropped else model.datainfo[i].name
            value = 'DROP' if not anonymous and dropped else None
            new = input_records[0]._create_option(key, value)
            keep.append(new)
        else:
            keep.append(child)

        i += 1

        if i >= len(model.datainfo):
            last_child = input_records[0].root.children[-1]
            if last_child.rule == 'NEWLINE':
                keep.append(last_child)
            break

    input_records[0].root = AttrTree(input_records[0].root.rule, tuple(keep))

    last_input_record = input_records[-1]
    for ci in model.datainfo[len(colnames) :]:
        last_input_record.append_option(ci.name, 'DROP' if ci.drop else None)


def update_initial_individual_estimates(model: Model, path, nofiles=False):
    """Update $ETAS

    Could have 0 FIX in model. Need to read these
    """
    if path is None:  # What to do here?
        phi_path = Path('.')
    else:
        phi_path = path.parent
    phi_path /= f'{model.name}_input.phi'

    estimates = model.initial_individual_estimates
    if estimates is not model.internals._old_initial_individual_estimates:
        assert estimates is not None
        rv_names = {rv for rv in model.random_variables.names if rv.startswith('ETA')}
        columns = set(estimates.columns)
        if columns < rv_names:
            raise ValueError(
                f'Cannot set initial estimate for random variable not in the model:'
                f' {rv_names - columns}'
            )
        diff = columns - rv_names
        # If not setting all etas automatically set remaining to 0 for all individuals
        if len(diff) > 0:
            for name in diff:
                estimates = estimates.copy(deep=True)
                estimates[name] = 0
            estimates = _sort_eta_columns(estimates)

        etas = estimates
        zero_fix = get_zero_fix_rvs(model.internals.control_stream, eta=True)
        if zero_fix:
            for eta in zero_fix:
                etas[eta] = 0
        etas = _sort_eta_columns(etas)
        if not nofiles:
            phi = PhiTable(df=etas)
            table_file = NONMEMTableFile(tables=[phi])
            table_file.write(phi_path)
        # FIXME: This is a common operation
        eta_records = model.internals.control_stream.get_records('ETAS')
        if eta_records:
            record = eta_records[0]
        else:
            record = model.internals.control_stream.append_record('$ETAS ')
        assert isinstance(record, EtasRecord)
        record.path = phi_path

        first_est_record = model.internals.control_stream.get_records('ESTIMATION')[0]
        try:
            first_est_record.option_pairs['MCETA']
        except KeyError:
            first_est_record.set_option('MCETA', '1')


def _sort_eta_columns(df: pd.DataFrame):
    return df.reindex(sorted(df.columns), axis=1)


def abbr_translation(model: Model, rv_trans):
    abbr_pharmpy = model.internals.control_stream.abbreviated.translate_to_pharmpy_names()
    abbr_replace = model.internals.control_stream.abbreviated.replace
    abbr_trans = update_abbr_record(model, rv_trans)
    abbr_recs = {
        sympy.Symbol(abbr_pharmpy[value]): sympy.Symbol(key)
        for key, value in abbr_replace.items()
        if value in abbr_pharmpy.keys()
    }
    abbr_trans.update(abbr_recs)
    return abbr_trans
