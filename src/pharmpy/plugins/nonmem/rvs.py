def update_random_variables(model, old, new):
    new_names = {rv.name for rv in new}
    old_names = {rv.name for rv in old}
    removed = old_names - new_names
    if removed:
        remove_records = []
        next_eta = 1
        for omega_record in model.control_stream.get_records('OMEGA'):
            name_map = omega_record.name_map
            if removed <= name_map.keys():
                remove_records.append(omega_record)
            elif not removed.isdisjoint(name_map.keys()):
                # one or more in the record

                omega_record.renumber(next_eta)
                next_eta += len(omega_record)
            else:
                # keep all
                omega_record.renumber(next_eta)
                next_eta += len(omega_record)
