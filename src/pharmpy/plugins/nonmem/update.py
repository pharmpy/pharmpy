import re


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
            current_names = omega_record.name_map.keys()
            if removed >= current_names:
                remove_records.append(omega_record)
            elif not removed.isdisjoint(current_names):
                # one or more in the record
                omega_record.remove(removed & current_names)
                omega_record.renumber(next_eta)
                # FIXME: No handling of OMEGA(1,1) etc in code
                full_map.update({key: f'ETA({value})'
                                 for key, value in omega_record.name_map.items()})
                next_eta += len(omega_record)
            else:
                # keep all
                omega_record.renumber(next_eta)
                full_map.update({key: f'ETA({value})'
                                 for key, value in omega_record.name_map.items()})
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
