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
                full_map.update({key: f'ETA({value})' for key, value in omega_record.name_map.items()})
                next_eta += len(omega_record)
            else:
                # keep all
                omega_record.renumber(next_eta)
                full_map.update({key: f'ETA({value})' for key, value in omega_record.name_map.items()})
                next_eta += len(omega_record)
        model.control_stream.remove_records(remove_records)

        code_record = model.get_pred_pk_record()
        code_record.update(full_map)
        error = model._get_error_record()
        if error:
            error.update(full_map)
