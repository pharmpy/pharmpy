import copy
import itertools
import re
import warnings
from pathlib import Path

import numpy as np
import sympy

from pharmpy import data
from pharmpy.plugins.nonmem.records import code_record
from pharmpy.random_variables import RandomVariables
from pharmpy.statements import (
    Assignment,
    Bolus,
    CompartmentalSystem,
    ExplicitODESystem,
    Infusion,
    ModelStatements,
    ODESystem,
)
from pharmpy.symbols import symbol

from .records.factory import create_record


def update_parameters(model, old, new):
    new_names = {p.name for p in new}
    old_names = {p.name for p in old}
    removed = old_names - new_names
    if removed:
        remove_records = []
        next_theta = 1
        for theta_record in model.control_stream.get_records('THETA'):
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
        for sigma_record in model.control_stream.get_records('SIGMA'):
            current_names = sigma_record.name_map.keys()
            if removed >= current_names:
                remove_records.append(sigma_record)
        model.control_stream.remove_records(remove_records)

    for p in new:
        name = p.name
        if name not in old and name not in model.random_variables.parameter_names:
            # This is a new theta
            theta_number = get_next_theta(model)
            record = create_theta_record(model, p)
            if re.match(r'THETA\(\d+\)', name):
                p.name = f'THETA({theta_number})'
            record.add_nonmem_name(p.name, theta_number)

    next_theta = 1
    for theta_record in model.control_stream.get_records('THETA'):
        theta_record.update(new, next_theta)
        next_theta += len(theta_record)
    next_omega = 1
    previous_size = None
    for omega_record in model.control_stream.get_records('OMEGA'):
        next_omega, previous_size = omega_record.update(new, next_omega, previous_size)
    next_sigma = 1
    previous_size = None
    for sigma_record in model.control_stream.get_records('SIGMA'):
        next_sigma, previous_size = sigma_record.update(new, next_sigma, previous_size)


def update_random_variables(model, old, new):
    from pharmpy.plugins.nonmem.records.code_record import diff

    if not hasattr(model, '_parameters'):
        model.parameters

    rec_dict = dict()
    comment_dict = dict()

    for omega_record in model.control_stream.get_records(
        'OMEGA'
    ) + model.control_stream.get_records('SIGMA'):
        comment_dict = {**comment_dict, **omega_record.comment_map}
        current_names = list(omega_record.eta_map.keys())
        for name in current_names:
            rec_dict[name] = omega_record

    rvs_diff_eta = diff(old.etas.distributions(), new.etas.distributions())
    rvs_diff_eps = diff(old.epsilons.distributions(), new.epsilons.distributions())

    update_random_variable_records(model, rvs_diff_eta, rec_dict, comment_dict)
    update_random_variable_records(model, rvs_diff_eps, rec_dict, comment_dict)

    next_eta = 1
    for omega_record in model.control_stream.get_records('OMEGA'):
        omega_record.renumber(next_eta)
        next_eta += len(omega_record)

    next_eps = 1
    for sigma_record in model.control_stream.get_records('SIGMA'):
        sigma_record.renumber(next_eps)
        next_eps += len(sigma_record)


def update_random_variable_records(model, rvs_diff, rec_dict, comment_dict):
    removed = []
    eta_number = 1
    number_of_records = 0

    rvs_diff = list(rvs_diff)

    rvs_removed = [RandomVariables(rvs).names for (op, (rvs, _)) in rvs_diff if op == '-']
    rvs_removed = [rv for sublist in rvs_removed for rv in sublist]

    for i, (op, (rvs, _)) in enumerate(rvs_diff):
        if op == '+':
            if len(rvs) == 1:
                create_omega_single(model, rvs[0], eta_number, number_of_records, comment_dict)
            else:
                create_omega_block(model, rvs, eta_number, number_of_records, comment_dict)
            eta_number += len(rvs)
            number_of_records += 1
        elif op == '-':
            rvs_rec = list({rec_dict[rv.name] for rv in rvs})
            recs_to_remove = [rec for rec in rvs_rec if rec not in removed]
            if recs_to_remove:
                model.control_stream.remove_records(recs_to_remove)
                removed += [rec for rec in recs_to_remove]
        else:
            diag_rvs = get_diagonal(rvs[0], rec_dict)
            # Account for etas in diagonal
            if len(rvs) == 1 and diag_rvs and any(rv in rvs_removed for rv in diag_rvs):
                create_omega_single(model, rvs[0], eta_number, number_of_records, comment_dict)
            eta_number += len(rvs)
            number_of_records += 1


def get_diagonal(rv, rec_dict):
    rv_rec = rec_dict[rv.name]
    etas_from_same_record = [eta for eta, rec in rec_dict.items() if rec == rv_rec]
    if len(etas_from_same_record) > 1:
        return etas_from_same_record
    return None


def get_next_theta(model):
    """Find the next available theta number"""
    next_theta = 1

    for theta_record in model.control_stream.get_records('THETA'):
        thetas = theta_record.parameters(next_theta)
        next_theta += len(thetas)

    return next_theta


def create_theta_record(model, param):
    param_str = '$THETA  '

    if param.upper < 1000000:
        if param.lower <= -1000000:
            param_str += f'(-INF,{param.init},{param.upper})'
        else:
            param_str += f'({param.lower},{param.init},{param.upper})'
    else:
        if param.lower <= -1000000:
            param_str += f'{param.init}'
        else:
            param_str += f'({param.lower},{param.init})'
    if param.fix:
        param_str += ' FIX'
    param_str += '\n'
    record = model.control_stream.insert_record(param_str)
    return record


def create_omega_single(model, rv, eta_number, record_number, comment_dict):
    rvs, pset = model.random_variables, model.parameters

    if rv.level == 'RUV':
        record_type = 'SIGMA'
    else:
        record_type = 'OMEGA'

    variance_param = pset[rv.parameter_names[0]]

    if rv.level == 'IOV':
        param_str = f'${record_type}  BLOCK(1)'
        first_iov = [iov for iov in rvs.iov if iov.parameter_names == rv.parameter_names][0]
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
    record.eta_map = {rv.name: eta_number}
    record.name_map = {variance_param.name: (eta_number, eta_number)}


def create_omega_block(model, rvs, eta_number, record_number, comment_dict):
    rvs = RandomVariables(rvs)
    param_str = f'$OMEGA BLOCK({rvs.covariance_matrix.shape[0]})\n'

    for row in range(rvs.covariance_matrix.shape[0]):
        for col in range(row + 1):
            elem = rvs.covariance_matrix.row(row).col(col)
            name = str(elem[0])
            omega = model.parameters[name]
            param_str += f'{omega.init}'

            if not re.match(r'OMEGA\(\d+,\d+\)', omega.name):
                param_str += f'\t; {omega.name}'
            elif comment_dict and omega.name in comment_dict:
                param_str += f'\t; {comment_dict[omega.name]}'

            param_str += '\n'

        param_str = f'{param_str.rstrip()}\n'

    eta_map, name_variance = dict(), dict()

    for rv in rvs:
        variance_param = rvs.get_variance(rv)
        eta_map[rv.name] = eta_number
        name_variance[variance_param.name] = (eta_number, eta_number)
        eta_number += 1

    rv_combinations = [
        (rv1.name, rv2.name) for idx, rv1 in enumerate(rvs) for rv2 in rvs[idx + 1 :]
    ]
    name_covariance = {
        rvs.get_covariance(rv1, rv2).name: (eta_map[rv2], eta_map[rv1])
        for rv1, rv2 in rv_combinations
    }

    record = insert_omega_record(model, param_str, record_number, 'OMEGA')

    record.comment_map = comment_dict
    record.eta_map = eta_map
    record.name_map = {**name_variance, **name_covariance}


def insert_omega_record(model, param_str, record_number, record_type):
    records = model.control_stream.records
    tprecs = model.control_stream.get_records(record_type)
    if tprecs:
        index = records.index(tprecs[0])
        record = model.control_stream.insert_record(param_str, index + record_number)
    else:
        record = model.control_stream.insert_record(param_str)
    return record


def update_ode_system(model, old, new):
    """Update ODE system

    Handle changes from CompartmentSystem to ExplicitODESystem
    """
    if type(old) == CompartmentalSystem and type(new) == ExplicitODESystem:
        to_des(model, new)
    elif type(old) == ExplicitODESystem and type(new) == CompartmentalSystem:
        # Stay with $DES for now
        update_des(model, old, new)
    elif type(old) == CompartmentalSystem and type(new) == CompartmentalSystem:
        # subs = model.control_stream.get_records('SUBROUTINES')[0]
        # old_trans = subs.get_option_startswith('TRANS')
        # conv_advan, new_advan, new_trans = change_advan(model)
        update_lag_time(model, old, new)
        if isinstance(new.dosing_compartment.dose, Bolus) and 'RATE' in model.dataset.columns:
            df = model.dataset
            df.drop(columns=['RATE'], inplace=True)
            model.dataset = df

        advan, trans = new_advan_trans(model)
        pk_param_conversion(model, advan=advan, trans=trans)
        add_needed_pk_parameters(model, advan, trans)
        update_subroutines_record(model, advan, trans)
        update_model_record(model, advan)

        statements = model.statements
        if isinstance(new.dosing_compartment.dose, Infusion) and not statements.find_assignment(
            'D1'
        ):
            # Handle direct moving of Infusion dose
            statements.subs({'D2': 'D1'})

        if isinstance(new.dosing_compartment.dose, Infusion) and isinstance(
            old.dosing_compartment.dose, Bolus
        ):
            dose = new.dosing_compartment.dose
            if dose.rate is None:
                # FIXME: Not always D1 here!
                ass = Assignment('D1', dose.duration)
                dose.duration = ass.symbol
            else:
                raise NotImplementedError("First order infusion rate is not yet supported")
            statements = model.statements
            statements.insert_before_odes(ass)
            df = model.dataset
            rate = np.where(df['AMT'] == 0, 0, -2)
            df['RATE'] = rate
            # FIXME: Adding at end for now. Update $INPUT cannot yet handle adding in middle
            # df.insert(list(df.columns).index('AMT') + 1, 'RATE', rate)
            model.dataset = df

    force_des(model, new)


def update_des(model, old, new):
    """Update where both old and new should be explicit ODE systems"""
    pass


def force_des(model, odes):
    """Switch to $DES if necessary"""
    if isinstance(odes, ExplicitODESystem):
        return

    amounts = {sympy.Function(amt.name)(symbol('t')) for amt in odes.amounts}
    if odes.atoms(sympy.Function) & amounts:
        explicit_odes(model)
        new = model.statements.ode_system
        to_des(model, new)


def explicit_odes(model):
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
        statements[model.statements.index(odes)] = new
        model.statements = statements
        new.solver = odes.solver
    return model


def to_des(model, new):
    subs = model.control_stream.get_records('SUBROUTINES')[0]
    subs.remove_option_startswith('TRANS')
    subs.remove_option_startswith('ADVAN')
    if new.solver:
        if new.solver == 'LSODA':
            advan = 'ADVAN13'
        elif new.solver == 'CVODES':
            advan = 'ADVAN14'
        elif new.solver == 'DGEAR':
            advan = 'ADVAN8'
        elif new.solver == 'DVERK':
            advan = 'ADVAN6'
        elif new.solver == 'IDA':
            advan = 'ADVAN15'
        elif new.solver == 'LSODA':
            advan = 'ADVAN13'
        elif new.solver == 'LSODI':
            advan = 'ADVAN9'
        subs.append_option(advan)
    else:
        subs.append_option('ADVAN6')
    subs.append_option('TOL', 3)
    des = model.control_stream.insert_record('$DES\nDUMMY=0')
    des.from_odes(new)
    model.control_stream.remove_records(model.control_stream.get_records('MODEL'))
    mod = model.control_stream.insert_record('$MODEL\n')
    for eq, ic in zip(new.odes[:-1], list(new.ics.keys())[:-1]):
        name = eq.lhs.args[0].name[2:]
        if new.ics[ic] != 0:
            dose = True
        else:
            dose = False
        mod.add_compartment(name, dosing=dose)


def update_statements(model, old, new, trans):
    trans['NaN'] = int(data.conf.na_rep)
    main_statements = ModelStatements()
    error_statements = ModelStatements()

    new_odes = new.ode_system
    if new_odes is not None:
        old_odes = old.ode_system
        if new_odes != old_odes:
            colnames, _, drop, _ = model._column_info()
            col_dropped = {key: value for key, value in zip(colnames, drop)}
            if 'CMT' in col_dropped.keys() and not col_dropped['CMT']:
                warnings.warn(
                    'Compartment structure has been updated, CMT-column '
                    'in dataset might not be relevant anymore. Check '
                    'CMT-column or drop column'
                )
            update_ode_system(model, old_odes, new_odes)

    after_odes = False
    for s in new:
        if isinstance(s, ODESystem):
            after_odes = True
        elif after_odes:
            error_statements.append(copy.deepcopy(s))
        else:
            main_statements.append(copy.deepcopy(s))

    main_statements.subs(trans)

    rec = model.get_pred_pk_record()
    rec.rvs, rec.trans = model.random_variables, trans
    rec.statements = main_statements

    error = model._get_error_record()
    if error:
        if len(error_statements) > 0:
            error_statements.pop(0)  # Remove the link statement
        error.rvs, error.trans = model.random_variables, trans
        error_statements.subs(trans)
        error.statements = error_statements
        error.is_updated = True

    rec.is_updated = True


def update_lag_time(model, old, new):
    new_dosing = new.dosing_compartment
    new_lag_time = new_dosing.lag_time
    old_lag_time = old.dosing_compartment.lag_time
    if new_lag_time != old_lag_time and new_lag_time != 0:
        ass = Assignment('ALAG1', new_lag_time)
        model.statements.insert_before_odes(ass)
        new_dosing.lag_time = ass.symbol


def new_compartmental_map(cs, oldmap):
    """Create compartmental map for updated model
    cs - new compartmental system
    old - old compartmental map

    Can handle compartments from dosing to central, peripherals and output
    """
    comp = cs.dosing_compartment
    central = cs.central_compartment
    i = 1
    compmap = dict()
    while True:
        compmap[comp.name] = i
        i += 1
        if comp is central:
            break
        comp, _ = cs.get_compartment_outflows(comp)[0]

    peripherals = cs.peripheral_compartments
    for p in peripherals:
        compmap[p.name] = i
        i += 1

    diff = len(cs) - len(oldmap)
    for name in cs.names:
        if name not in compmap.keys():
            compmap[name] = oldmap[name] + diff
    return compmap


def create_compartment_remap(oldmap, newmap):
    """Creates a map from old compartment number to new compartment number

    For all compartments where remapping is needed
    Assume that compartments with same name in new and old are the same compartments
    """
    remap = dict()
    for name, number in oldmap.items():
        if name in newmap:
            remap[number] = newmap[name]
    return remap


def pk_param_conversion(model, advan, trans):
    """Conversion map for pk parameters for removed or added compartment"""
    subs = model.control_stream.get_records('SUBROUTINES')[0]
    from_advan = subs.get_option_startswith('ADVAN')
    statements = model.statements
    cs = statements.ode_system
    oldmap = model._compartment_map
    newmap = new_compartmental_map(cs, oldmap)
    remap = create_compartment_remap(oldmap, newmap)
    d = dict()
    for old, new in remap.items():
        d[symbol(f'S{old}')] = symbol(f'S{new}')
        d[symbol(f'F{old}')] = symbol(f'F{new}')
        d[symbol(f'R{old}')] = symbol(f'R{new}')
        d[symbol(f'D{old}')] = symbol(f'D{new}')
        d[symbol(f'ALAG{old}')] = symbol(f'ALAG{new}')
        d[symbol(f'A({old})')] = symbol(f'A({new})')
    if from_advan == 'ADVAN5' or from_advan == 'ADVAN7':
        reverse_map = {v: k for k, v in newmap.items()}
        for i, j in itertools.product(range(1, len(oldmap)), range(0, len(oldmap))):
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
                    d[symbol(f'K{i}{j}')] = symbol(f'K{to_i}{to_j}')
                    d[symbol(f'K{i}T{j}')] = symbol(f'K{to_i}T{to_j}')
        if advan == 'ADVAN3':
            n = len(oldmap)
            for i in range(1, n):
                d[symbol(f'K{i}0')] = symbol('K')
                d[symbol(f'K{i}T0')] = symbol('K')
                d[symbol(f'K{i}{n}')] = symbol('K')
                d[symbol(f'K{i}T{n}')] = symbol('K')
    elif from_advan == 'ADVAN1':
        if advan == 'ADVAN3' or advan == 'ADVAN11':
            d[symbol('V')] = symbol('V1')
        elif advan == 'ADVAN4' or advan == 'ADVAN12':
            d[symbol('V')] = symbol('V2')
    elif from_advan == 'ADVAN2':
        if advan == 'ADVAN3' and trans != 'TRANS1':
            d[symbol('V')] = symbol('V1')
        elif advan == 'ADVAN4' and trans != 'TRANS1':
            d[symbol('V')] = symbol('V2')
    elif from_advan == 'ADVAN3':
        if advan == 'ADVAN1':
            if trans == 'TRANS2':
                d[symbol('V1')] = symbol('V')
        elif advan == 'ADVAN4':
            if trans == 'TRANS4':
                d[symbol('V1')] = symbol('V2')
                d[symbol('V2')] = symbol('V3')
            elif trans == 'TRANS6':
                d[symbol('K21')] = symbol('K32')
            else:  # TRANS1
                d[symbol('K12')] = symbol('K23')
                d[symbol('K21')] = symbol('K32')
        elif advan == 'ADVAN11':
            if trans == 'TRANS4':
                d.update({symbol('Q'): symbol('Q2')})
    elif from_advan == 'ADVAN4':
        if advan == 'ADVAN2':
            if trans == 'TRANS2':
                d[symbol('V2')] = symbol('V')
        if advan == 'ADVAN3':
            if trans == 'TRANS4':
                d.update({symbol('V2'): symbol('V1'), symbol('V3'): symbol('V2')})
            elif trans == 'TRANS6':
                d.update({symbol('K32'): symbol('K21')})
            else:  # TRANS1
                d.update({symbol('K23'): symbol('K12'), symbol('K32'): symbol('K21')})
        elif advan == 'ADVAN12':
            if trans == 'TRANS4':
                d.update({symbol('Q'): symbol('Q3')})
    elif from_advan == 'ADVAN11':
        if advan == 'ADVAN1':
            if trans == 'TRANS2':
                d[symbol('V1')] = symbol('V')
        elif advan == 'ADVAN3':
            if trans == 'TRANS4':
                d[symbol('Q2')] = symbol('Q')
        elif advan == 'ADVAN12':
            if trans == 'TRANS4':
                d.update(
                    {
                        symbol('V1'): symbol('V2'),
                        symbol('Q2'): symbol('Q3'),
                        symbol('V2'): symbol('V3'),
                        symbol('Q3'): symbol('Q4'),
                        symbol('V3'): symbol('V4'),
                    }
                )
            elif trans == 'TRANS6':
                d.update({symbol('K31'): symbol('K42'), symbol('K21'): symbol('K32')})
            else:  # TRANS1
                d.update(
                    {
                        symbol('K12'): symbol('K23'),
                        symbol('K21'): symbol('K32'),
                        symbol('K13'): symbol('K24'),
                        symbol('K31'): symbol('K42'),
                    }
                )
    elif from_advan == 'ADVAN12':
        if advan == 'ADVAN2':
            if trans == 'TRANS2':
                d[symbol('V2')] = symbol('V')
        elif advan == 'ADVAN4':
            if trans == 'TRANS4':
                d[symbol('Q3')] = symbol('Q')
        elif advan == 'ADVAN11':
            if trans == 'TRANS4':
                d.update(
                    {
                        symbol('V2'): symbol('V1'),
                        symbol('Q3'): symbol('Q2'),
                        symbol('V3'): symbol('V2'),
                        symbol('Q4'): symbol('Q3'),
                        symbol('V4'): symbol('V3'),
                    }
                )
            elif trans == 'TRANS6':
                d.update({symbol('K42'): symbol('K31'), symbol('K32'): symbol('K21')})
            else:  # TRANS1
                d.update(
                    {
                        symbol('K23'): symbol('K12'),
                        symbol('K32'): symbol('K21'),
                        symbol('K24'): symbol('K13'),
                        symbol('K42'): symbol('K31'),
                    }
                )
    if advan == 'ADVAN5' or advan == 'ADVAN7' and from_advan not in ('ADVAN5', 'ADVAN7'):
        n = len(newmap)
        d[symbol('K')] = symbol(f'K{n-1}0')
    statements.subs(d)


def new_advan_trans(model):
    """Decide which new advan and trans to be used"""
    subs = model.control_stream.get_records('SUBROUTINES')[0]
    oldtrans = subs.get_option_startswith('TRANS')
    statements = model.statements
    odes = model.statements.ode_system
    if len(odes) > 5 or odes.get_n_connected(odes.central_compartment) != len(odes) - 1:
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

    if oldtrans == 'TRANS1':
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
    else:
        trans = 'TRANS1'

    return advan, trans


def update_subroutines_record(model, advan, trans):
    """Update $SUBROUTINES with new advan and trans"""
    subs = model.control_stream.get_records('SUBROUTINES')[0]
    oldadvan = subs.get_option_startswith('ADVAN')
    oldtrans = subs.get_option_startswith('TRANS')

    if advan != oldadvan:
        subs.replace_option(oldadvan, advan)
    if trans != oldtrans:
        subs.replace_option(oldtrans, trans)


def update_model_record(model, advan):
    """Update $MODEL"""
    newmap = new_compartmental_map(model.statements.ode_system, model._compartment_map)
    if advan in ['ADVAN1', 'ADVAN2', 'ADVAN3', 'ADVAN4', 'ADVAN10', 'ADVAN11', 'ADVAN12']:
        model.control_stream.remove_records(model.control_stream.get_records('MODEL'))
    else:
        oldmap = model._compartment_map
        if oldmap != newmap:
            model.control_stream.remove_records(model.control_stream.get_records('MODEL'))
            mod = model.control_stream.insert_record('$MODEL\n')
            output_name = model.statements.ode_system.output_compartment.name
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
    model._compartment_map = newmap


def add_needed_pk_parameters(model, advan, trans):
    """Add missing pk parameters that NONMEM needs"""
    statements = model.statements
    odes = statements.ode_system
    if advan == 'ADVAN2' or advan == 'ADVAN4' or advan == 'ADVAN12':
        if not statements.find_assignment('KA'):
            comp, rate = odes.get_compartment_outflows(odes.find_depot(statements))[0]
            ass = Assignment('KA', rate)
            if rate != ass.symbol:
                statements.insert_before_odes(ass)
                odes.add_flow(odes.find_depot(statements), comp, ass.symbol)
    if advan == 'ADVAN3' and trans == 'TRANS4':
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
        newmap = new_compartmental_map(odes, model._compartment_map)
        for source in newmap.keys():
            for dest in newmap.keys():
                if source != dest and source != len(newmap):
                    source_comp = odes.find_compartment(source)
                    dest_comp = odes.find_compartment(dest)
                    rate = odes.get_flow(source_comp, dest_comp)
                    if rate is not None:
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


def add_parameters_ratio(model, numpar, denompar, source, dest):
    statements = model.statements
    if not statements.find_assignment(numpar) or not statements.find_assignment(denompar):
        odes = statements.ode_system
        rate = odes.get_flow(source, dest)
        numer, denom = rate.as_numer_denom()
        par1 = Assignment(numpar, numer)
        par2 = Assignment(denompar, denom)
        if rate != par1.symbol / par2.symbol:
            if not statements.find_assignment(numpar):
                statements.insert_before_odes(par1)
                odes.subs({numer: sympy.Symbol(numpar)})
            if not statements.find_assignment(denompar):
                statements.insert_before_odes(par2)
                odes.subs({denom: sympy.Symbol(denompar)})
        odes.add_flow(source, dest, par1.symbol / par2.symbol)


def define_parameter(model, name, value, synonyms=None):
    """Define a parameter in statments if not defined
    Update if already defined as other value
    return True if new assignment was added
    """
    if synonyms is None:
        synonyms = [name]
    for syn in synonyms:
        ass = model.statements.find_assignment(syn)
        if ass:
            if value != ass.expression and value != symbol(name):
                ass.expression = value
            return False
    new_ass = Assignment(name, value)
    model.statements.insert_before_odes(new_ass)
    return True


def add_rate_assignment_if_missing(model, name, value, source, dest, synonyms=None):
    added = define_parameter(model, name, value, synonyms=synonyms)
    if added:
        model.statements.ode_system.add_flow(source, dest, symbol(name))


def update_abbr_record(model, rv_trans):
    trans = dict()
    if not rv_trans:
        return trans
    # Remove already abbreviated symbols
    # FIXME: Doesn't update if name has changed
    kept = rv_trans.copy()
    abbr_recs = model.control_stream.get_records('ABBREVIATED')
    for rec in abbr_recs:
        for rk, rv in rec.replace.items():
            for tk, tv in rv_trans.items():
                if tv.name == rv:
                    del kept[tk]
    rv_trans = kept
    if not rv_trans:
        return trans

    for rv in model.random_variables:
        rv_symb = symbol(rv.name)
        abbr_pattern = re.match(r'ETA_(\w+)', rv.name)
        if abbr_pattern and '_' not in abbr_pattern.group(1):
            parameter = abbr_pattern.group(1)
            nonmem_name = rv_trans[rv_symb]
            abbr_name = f'ETA({parameter})'
            trans[rv_symb] = symbol(abbr_name)
            abbr_record = f'$ABBR REPLACE {abbr_name}={nonmem_name}\n'
            model.control_stream.insert_record(abbr_record)
        elif not re.match(r'(ETA|EPS)\([0-9]\)', rv.name):
            warnings.warn(
                f'Not valid format of name {rv.name}, falling back to NONMEM name. If custom name, '
                f'follow the format "ETA_X" to get "ETA(X)" in $ABBR.'
            )
    return trans


def update_estimation(model):
    new = model.estimation_steps
    try:
        old = model._old_estimation_steps
    except AttributeError:
        old = []
    if old == new:
        return

    delta = code_record.diff(old, new)
    old_records = model.control_stream.get_records('ESTIMATION')
    i = 0
    new_records = []
    for op, est in delta:
        if op == '+':
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
                est_code += f' MAXEVAL={est.maximum_evaluations}'
                if set(protected_attributes).intersection({'MAXEVALS', 'MAXEVAL'}):
                    raise ValueError('MAXEVAL already set by evaluation=True')
                protected_attributes += ['MAXEVALS', 'MAXEVAL']
            if est.tool_options:
                option_names = {key for key in est.tool_options.keys()}
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
        elif op == '-':
            i += 1
        else:
            new_records.append(old_records[i])
            i += 1
    if old_records:
        model.control_stream.replace_records(old_records, new_records)
    else:
        for rec in new_records:
            model.control_stream.insert_record(str(rec))

    old_cov = False
    for est in old:
        old_cov |= est.cov
    new_cov = False
    for est in new:
        new_cov |= est.cov
    if not old_cov and new_cov:
        # Add $COV
        model.control_stream.insert_record('$COVARIANCE\n')
    elif old_cov and not new_cov:
        # Remove $COV
        covrecs = model.control_stream.get_records('COVARIANCE')
        model.control_stream.remove_records(covrecs)

    model._old_estimation_steps = copy.deepcopy(new)


def update_ccontra(model, path=None, force=False):
    h = model.observation_transformation
    y = model.dependent_variable
    dhdy = sympy.diff(h, y)
    ll = -2 * sympy.log(dhdy)
    ll = ll.subs(y, sympy.Symbol('y', real=True, positive=True))
    ll = model.parameters.simplify(ll)
    ll = ll.subs(sympy.Symbol('y', real=True, positive=True), y)

    tr = model.parameter_translation(reverse=True, remove_idempotent=True, as_symbols=True)
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
        e1 = sympy.printing.fortran.fcode(h.subs(y, sympy.Symbol('y(1)')), assign_to='y(1)')
        fh.write(e1)
        fh.write(ccontr2)
        e2 = sympy.printing.fortran.fcode(
            sympy.Symbol('c1') + ll.subs(y, sympy.Symbol('y(1)')), assign_to='c1'
        )
        fh.write(e2)
        fh.write(ccontr3)
