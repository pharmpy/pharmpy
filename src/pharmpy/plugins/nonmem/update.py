import copy
import itertools
import re

import numpy as np
import sympy

from pharmpy import data
from pharmpy.random_variables import RandomVariables, VariabilityLevel
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
        if name not in old and name not in model.random_variables.all_parameters():
            # This is a new theta
            theta_number = get_next_theta(model)
            record = create_theta_record(model, p)
            if re.match(r'THETA\(\d+\)', name):
                p.name = f'THETA({theta_number})'
            else:
                record.add_nonmem_name(name, theta_number)

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
    model.control_stream.go_through_omega_rec()
    new_names = {rv.name for rv in new}
    old_names = {rv.name for rv in old}
    removed = old_names - new_names
    modified_names = []
    if removed:
        remove_records = []
        for omega_record in model.control_stream.get_records(
            'OMEGA'
        ) + model.control_stream.get_records('SIGMA'):
            current_names = omega_record.eta_map.keys()
            if removed >= current_names:
                remove_records.append(omega_record)
            elif not removed.isdisjoint(current_names):
                # one or more in the record
                omega_record.remove(removed & current_names)
                modified_names += current_names
        model.control_stream.remove_records(remove_records)

    new_maps = []
    for rv in new:
        if rv.name not in old_names:
            omega_name = (rv.pspace.distribution.std ** 2).name
            if omega_name not in old.all_parameters():
                rv_name = rv.name.upper()
                omega = model.parameters[omega_name]

                iov_rv = None
                if rv.variability_level == VariabilityLevel.RUV:
                    record_name = 'SIGMA'
                else:
                    record_name = 'OMEGA'
                    if rv.variability_level == VariabilityLevel.IOV:
                        iov_rv = rv

                record, eta_number = create_omega_single(model, omega, record_name, iov_rv=iov_rv)
                record.add_omega_name_comment(omega_name)

                new_maps.append(
                    (record, {omega_name: (eta_number, eta_number)}, {rv_name: eta_number})
                )
    # FIXME: Setting the maps needs to be done here and not in loop. Automatic renumbering is
    #        probably the culprit. There should be a difference between added parameters and
    #        original parameters when it comes to which naming scheme to use
    if new_maps:
        for record, name_map, eta_map in new_maps:
            record.name_map = name_map
            record.eta_map = eta_map

    rvs_old = [rvs[0] for rvs in old.distributions()]

    for rvs, dist in new.distributions():
        rv_names = [rv.name for rv in rvs]

        if modified_names and set(modified_names).issubset(rv_names):
            continue

        if rvs not in rvs_old and set(rv_names).issubset(old_names):
            records = get_omega_records(model, rv_names)

            comment_map_new = dict()
            for comment_map in [omega_record.comment_map for omega_record in records]:
                comment_map_new.update(comment_map)

            indices = [model.control_stream.records.index(rec) for rec in records]
            new_rec_index = None if not indices else indices[0]

            if len(indices) != len(rvs):
                new_rec_index = None
            elif new_rec_index and len(indices) == len(rvs):
                for i in range(1, len(indices)):
                    if indices[i] - indices[i - 1] != 1:
                        new_rec_index = None
                        break

            model.control_stream.remove_records(records)

            if len(rvs) == 1:
                omega_new, _ = create_omega_single(
                    model, model.parameters[str(dist.std ** 2)], index=new_rec_index
                )
            else:
                omega_new = create_omega_block(model, dist, comment_map_new, index=new_rec_index)

            omega_new.comment_map = comment_map_new

            omega_start = 1

            for omega_record in model.control_stream.get_records('OMEGA'):
                if omega_record != omega_new:
                    omega_start += len(omega_record)
                else:
                    etas, _, _, _ = omega_record.random_variables(omega_start)

                    create_record_maps(etas, rvs, omega_record.name_map, omega_record.eta_map)

    next_eta = 1
    for omega_record in model.control_stream.get_records('OMEGA'):
        omega_record.renumber(next_eta)
        next_eta += len(omega_record)

    next_eps = 1
    for sigma_record in model.control_stream.get_records('SIGMA'):
        sigma_record.renumber(next_eps)
        next_eps += len(sigma_record)


def get_omega_records(model, params):
    records = []

    for omega_record in model.control_stream.get_records('OMEGA'):
        _, eta_map = omega_record.name_map, omega_record.eta_map

        for eta in eta_map.keys():
            if str(eta) in params:
                records.append(omega_record)
                break
    return records


def get_next_theta(model):
    """Find the next available theta number"""
    next_theta = 1

    for theta_record in model.control_stream.get_records('THETA'):
        thetas = theta_record.parameters(next_theta)
        next_theta += len(thetas)

    return next_theta


def get_next_eta(model, record='OMEGA'):
    """Find the next available eta number"""
    next_omega = 1
    previous_size = None

    for omega_record in model.control_stream.get_records(record):
        _, next_omega, previous_size = omega_record.parameters(next_omega, previous_size)

    return next_omega, previous_size


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


def create_omega_single(model, param, record='OMEGA', index=None, iov_rv=None):
    eta_number, previous_size = get_next_eta(model, record)
    rvs = model.random_variables
    previous_cov = None

    if iov_rv:
        param_str = f'${record}  BLOCK(1)'
        first_iov = rvs.get_connected_iovs(iov_rv)[0]
        if iov_rv == first_iov:
            param_str += f'\n{param.init}'
        else:
            param_str += ' SAME\n'
        previous_cov = eta_number - 1
    else:
        param_str = f'${record}  {param.init}\n'

    record = model.control_stream.insert_record(param_str, index)

    record.parameters(eta_number, previous_size)
    record.random_variables(eta_number, previous_cov)

    return record, eta_number


def create_omega_block(model, dist, comment_map, index=None):
    m = dist.args[1]
    param_str = f'$OMEGA BLOCK({m.shape[0]})\n'

    for row in range(m.shape[0]):
        for col in range(row + 1):
            elem = m.row(row).col(col)
            name = str(elem[0])
            omega = model.parameters[name]
            param_str += f'{omega.init}'

            if not re.match(r'OMEGA\(\d+,\d+\)', omega.name):
                param_str += f'\t; {omega.name}'
            elif omega.name in comment_map:
                param_str += f'\t; {comment_map[omega.name]}'

            param_str += '\n'

        param_str = f'{param_str.rstrip()}\n'

    record = model.control_stream.insert_record(param_str, index)
    return record


def create_record_maps(etas, rvs, name_map, eta_map):
    cov_rvs = RandomVariables(rvs).covariance_matrix()
    cov_rec = RandomVariables(etas).covariance_matrix()

    for row in range(cov_rvs.shape[0]):
        for col in range(cov_rvs.shape[1]):
            if row >= col:
                elem_new = cov_rvs.row(row).col(col)[0]
                elem_rec = cov_rec.row(row).col(col)[0]
                name_map[str(elem_new)] = name_map.pop(str(elem_rec))
    for rv, eta in zip(rvs, etas):
        eta_map[str(rv)] = eta_map.pop(str(eta))


def update_ode_system(model, old, new):
    """Update ODE system

    Handle changes from CompartmentSystem to ExplicitODESystem
    """
    if type(old) == CompartmentalSystem and type(new) == ExplicitODESystem:
        to_des(model, new)
    elif type(old) == CompartmentalSystem and type(new) == CompartmentalSystem:
        # subs = model.control_stream.get_records('SUBROUTINES')[0]
        # old_trans = subs.get_option_startswith('TRANS')
        # conv_advan, new_advan, new_trans = change_advan(model)
        update_lag_time(model, old, new)
        if isinstance(new.find_dosing().dose, Bolus) and 'RATE' in model.dataset.columns:
            df = model.dataset
            df.drop(columns=['RATE'], inplace=True)
            model.dataset = df

        advan, trans = new_advan_trans(model)
        pk_param_conversion(model, advan=advan, trans=trans)
        add_needed_pk_parameters(model, advan, trans)
        update_subroutines_record(model, advan, trans)
        update_model_record(model, advan)

        statements = model.statements
        if isinstance(new.find_dosing().dose, Infusion) and not statements.find_assignment('D1'):
            # Handle direct moving of Infusion dose
            statements.subs({'D2': 'D1'})

        if isinstance(new.find_dosing().dose, Infusion) and isinstance(
            old.find_dosing().dose, Bolus
        ):
            dose = new.find_dosing().dose
            if dose.rate is None:
                # FIXME: Not always D1 here!
                ass = Assignment('D1', dose.duration)
                dose.duration = ass.symbol
            else:
                raise NotImplementedError("First order infusion rate is not yet supported")
            statements = model.statements
            statements.add_before_odes(ass)
            df = model.dataset
            rate = np.where(df['AMT'] == 0, 0, -2)
            df['RATE'] = rate
            # FIXME: Adding at end for now. Update $INPUT cannot yet handle adding in middle
            # df.insert(list(df.columns).index('AMT') + 1, 'RATE', rate)
            model.dataset = df

    force_des(model, new)


def force_des(model, odes):
    """Switch to $DES if necessary"""
    if isinstance(odes, ExplicitODESystem):
        return

    # Import put here to avoid circular import in Python 3.6
    import pharmpy.modeling as modeling

    amounts = {sympy.Function(amt.name)(symbol('t')) for amt in odes.amounts}
    if odes.atoms(sympy.Function) & amounts:
        modeling.explicit_odes(model)
        new = model.statements.ode_system
        to_des(model, new)


def to_des(model, new):
    subs = model.control_stream.get_records('SUBROUTINES')[0]
    subs.remove_option_startswith('TRANS')
    subs.remove_option_startswith('ADVAN')
    if new.solver:
        subs.append_option(new.solver)
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
    new_dosing = new.find_dosing()
    new_lag_time = new_dosing.lag_time
    old_lag_time = old.find_dosing().lag_time
    if new_lag_time != old_lag_time and new_lag_time != 0:
        ass = Assignment('ALAG1', new_lag_time)
        model.statements.add_before_odes(ass)
        new_dosing.lag_time = ass.symbol


def new_compartmental_map(cs, oldmap):
    """Create compartmental map for updated model
    cs - new compartmental system
    old - old compartmental map

    Can handle compartments from dosing to central, peripherals and output
    """
    comp = cs.find_dosing()
    central = cs.find_central()
    i = 1
    compmap = dict()
    while True:
        compmap[comp.name] = i
        i += 1
        if comp is central:
            break
        comp, _ = cs.get_compartment_outflows(comp)[0]

    peripherals = cs.find_peripherals()
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
        if advan == 'ADVAN3':
            d[symbol('V')] = symbol('V1')
        elif advan == 'ADVAN4':
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
    if trans == 'TRANS1' and len(oldmap) == 3 and len(newmap) > 3:
        n = len(newmap)
        d[symbol('K')] = symbol(f'K{n-1}0')
    statements.subs(d)


def new_advan_trans(model):
    """Decide which new advan and trans to be used"""
    subs = model.control_stream.get_records('SUBROUTINES')[0]
    oldtrans = subs.get_option_startswith('TRANS')
    statements = model.statements
    odes = model.statements.ode_system
    if len(odes) > 5 or odes.n_connected(odes.find_central()) != len(odes) - 1:
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
            output_name = model.statements.ode_system.find_output().name
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
                statements.add_before_odes(ass)
                odes.add_flow(odes.find_depot(statements), comp, ass.symbol)
    if advan == 'ADVAN3' and trans == 'TRANS4':
        central = odes.find_central()
        output = odes.find_output()
        peripheral = odes.find_peripherals()[0]
        add_parameters_ratio(model, 'CL', 'V1', central, output)
        add_parameters_ratio(model, 'Q', 'V2', peripheral, central)
    elif advan == 'ADVAN4' and trans == 'TRANS4':
        central = odes.find_central()
        output = odes.find_output()
        peripheral = odes.find_peripherals()[0]
        add_parameters_ratio(model, 'CL', 'V2', central, output)
        add_parameters_ratio(model, 'Q', 'V3', peripheral, central)
    elif advan == 'ADVAN12' and trans == 'TRANS4':
        central = odes.find_central()
        output = odes.find_output()
        peripheral1 = odes.find_peripherals()[0]
        peripheral2 = odes.find_peripherals()[1]
        add_parameters_ratio(model, 'CL', 'V2', central, output)
        add_parameters_ratio(model, 'Q3', 'V3', peripheral1, central)
        add_parameters_ratio(model, 'Q4', 'V4', peripheral2, central)
    elif advan == 'ADVAN11' and trans == 'TRANS4':
        central = odes.find_central()
        output = odes.find_output()
        peripheral1 = odes.find_peripherals()[0]
        peripheral2 = odes.find_peripherals()[1]
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
                statements.add_before_odes(par1)
            if not statements.find_assignment(denompar):
                statements.add_before_odes(par2)
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
    model.statements.add_before_odes(new_ass)
    return True


def add_rate_assignment_if_missing(model, name, value, source, dest, synonyms=None):
    added = define_parameter(model, name, value, synonyms=synonyms)
    if added:
        model.statements.ode_system.add_flow(source, dest, symbol(name))
