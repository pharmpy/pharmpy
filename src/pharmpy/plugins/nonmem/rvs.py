def update_random_variables(model, old, new):
    new_names = {rv.name for rv in new}
    old_names = {rv.name for rv in old}
    removed = old_names - new_names
    removed
    # for name in removed:
    #    self._old_random_variables[name]
