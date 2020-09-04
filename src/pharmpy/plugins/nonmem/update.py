import re

from pharmpy.statements import CompartmentalSystem, ExplicitODESystem
from pharmpy.symbols import real


def update_parameters(model, old, new):
    new_names = {p.name for p in new}
    old_names = {p.name for p in old}
    removed = old_names - new_names
    full_map = dict()
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
                full_map.update({key: f'THETA({value})'
                                 for key, value in theta_record.name_map.items()})
                next_theta += len(theta_record)
            else:
                # keep all
                theta_record.renumber(next_theta)
                full_map.update({key: f'THETA({value})'
                                 for key, value in theta_record.name_map.items()})
                next_theta += len(theta_record)
        model.control_stream.remove_records(remove_records)

    for p in new:
        name = p.name
        if name not in model._old_parameters and \
                name not in model.random_variables.all_parameters():
            # This is a new theta
            theta_number = get_next_theta(model)
            record = create_theta_record(model, p)
            if re.match(r'THETA\(\d+\)', name):
                p.name = f'THETA({theta_number})'
            else:
                record.add_nonmem_name(name, theta_number)
            full_map[p.name] = f'THETA({theta_number})'

    if full_map:
        update_code_symbols(model, full_map)

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
    new_names = {rv.name for rv in new}
    old_names = {rv.name for rv in old}
    removed = old_names - new_names
    if removed:
        remove_records = []
        next_eta = 1
        full_map = dict()
        for omega_record in model.control_stream.get_records('OMEGA'):
            current_names = omega_record.eta_map.keys()
            if removed >= current_names:
                remove_records.append(omega_record)
            elif not removed.isdisjoint(current_names):
                # one or more in the record
                omega_record.remove(removed & current_names)
                omega_record.renumber(next_eta)
                # FIXME: No handling of OMEGA(1,1) etc in code
                full_map.update({key: f'ETA({value})'
                                 for key, value in omega_record.eta_map.items()})
                next_eta += len(omega_record)
            else:
                # keep all
                omega_record.renumber(next_eta)
                full_map.update({key: f'ETA({value})'
                                 for key, value in omega_record.eta_map.items()})
                next_eta += len(omega_record)
        model.control_stream.remove_records(remove_records)
        update_code_symbols(model, full_map)


def get_next_theta(model):
    """ Find the next available theta number
    """
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
    record = model.control_stream.insert_record(param_str, 'THETA')
    return record


def update_code_symbols(model, name_map):
    """Update symbol names in code records.

        name_map - dict from old name to new name
    """
    code_record = model.get_pred_pk_record()
    code_record.update(name_map)
    error = model._get_error_record()
    if error:
        error.update(name_map)


def update_ode_system(model, old, new):
    """Update ODE system

       Handle changes from CompartmentSystem to ExplicitODESystem
    """
    if type(old) == CompartmentalSystem and type(new) == ExplicitODESystem:
        subs = model.control_stream.get_records('SUBROUTINES')[0]
        subs.remove_option_startswith('TRANS')
        subs.remove_option_startswith('ADVAN')
        subs.append_option('ADVAN6')
        des = model.control_stream.insert_record('$DES\nDUMMY=0', 'PK')
        des.from_odes(new)
        mod = model.control_stream.insert_record('$MODEL TOL=3\n', 'SUBROUTINES')
        for eq, ic in zip(new.odes[:-1], list(new.ics.keys())[:-1]):
            name = eq.lhs.args[0].name[2:]
            if new.ics[ic] != 0:
                dose = True
            else:
                dose = False
            mod.add_compartment(name, dosing=dose)
    elif type(old) == CompartmentalSystem and type(new) == CompartmentalSystem:
        if old.find_depot() and not new.find_depot():
            subs = model.control_stream.get_records('SUBROUTINES')[0]
            advan = subs.get_option_startswith('ADVAN')
            statements = model.statements
            if advan == 'ADVAN2':
                subs.replace_option('ADVAN2', 'ADVAN1')
            elif advan == 'ADVAN4':
                subs.replace_option('ADVAN4', 'ADVAN3')
                statements.subs({real('K23'): real('K12'), real('K32'): real('K32')})
            elif advan == 'ADVAN12':
                subs.replace_option('ADVAN12', 'ADVAN11')
                statements.subs({real('K23'): real('K12'), real('K32'): real('K32'),
                                 real('K24'): real('K13'), real('K42'): real('K31')})
            # FIXME: It could possibly be other than the first below
            # also assumes that only one compartment has been removed
            secondary = secondary_pk_param_conversion_map(len(old), 1)
            statements.subs(secondary)
            model.statements = statements


def primary_pk_param_conversion_map(ncomp, trans, removed):
    """Conversion map for pk parameters for one removed compartment
    """
    if trans == 'TRANS1':
        pass


def secondary_pk_param_conversion_map(ncomp, removed):
    """Conversion map for pk parameters for one removed compartment

        ncomp - total number of compartments before removing (including output)
        removed - number of removed compartment
    """
    d = dict()
    for i in range(removed + 1, ncomp + 1):
        d.update({real(f'S{i})'): real(f'S{i - 1}'),
                  real(f'F{i}'): real(f'F{i - 1}'),
                  real(f'R{i}'): real(f'R{i - 1}'),
                  real(f'D{i}'): real(f'D{i - 1}'),
                  real(f'ALAG{i}'): real(f'ALAG{i - 1}')})
    return d
