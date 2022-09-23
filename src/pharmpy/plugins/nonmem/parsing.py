import warnings

import pharmpy.plugins.nonmem
from pharmpy.deps import sympy


def parameter_translation(control_stream, reverse=False, remove_idempotent=False, as_symbols=False):
    """Get a dict of NONMEM name to Pharmpy parameter name
    i.e. {'THETA(1)': 'TVCL', 'OMEGA(1,1)': 'IVCL'}
    """
    d = dict()
    for theta_record in control_stream.get_records('THETA'):
        for key, value in theta_record.name_map.items():
            nonmem_name = f'THETA({value})'
            d[nonmem_name] = key
    for record in control_stream.get_records('OMEGA'):
        for key, value in record.name_map.items():
            nonmem_name = f'OMEGA({value[0]},{value[1]})'
            d[nonmem_name] = key
    for record in control_stream.get_records('SIGMA'):
        for key, value in record.name_map.items():
            nonmem_name = f'SIGMA({value[0]},{value[1]})'
            d[nonmem_name] = key
    if remove_idempotent:
        d = {key: val for key, val in d.items() if key != val}
    if reverse:
        d = {val: key for key, val in d.items()}
    if as_symbols:
        d = {sympy.Symbol(key): sympy.Symbol(val) for key, val in d.items()}
    return d


def create_name_trans(control_stream, rvs, statements):
    conf_functions = {
        'comment': _name_as_comments(control_stream, statements),
        'abbr': _name_as_abbr(control_stream, rvs),
        'basic': _name_as_basic(control_stream),
    }

    abbr = control_stream.abbreviated.replace
    pset_current = {
        **parameter_translation(control_stream, reverse=True),
        **{rv: rv for rv in rvs.names},
    }
    sset_current = {
        **abbr,
        **{
            rv: rv
            for rv in rvs.names
            if rv not in abbr.keys() and sympy.Symbol(rv) in statements.free_symbols
        },
        **{
            p: p
            for p in pset_current.values()
            if p not in abbr.keys() and sympy.Symbol(p) in statements.free_symbols
        },
    }

    trans_sset, trans_pset = dict(), dict()
    names_sset_translated, names_pset_translated, names_basic = [], [], []
    clashing_symbols = set()

    for setting in pharmpy.plugins.nonmem.conf.parameter_names:
        trans_sset_setting, trans_pset_setting = conf_functions[setting]
        if setting != 'basic':
            clashing_symbols.update(
                _clashing_symbols(statements, {**trans_sset_setting, **trans_pset_setting})
            )
        for name_current, name_new in trans_sset_setting.items():
            name_nonmem = sset_current[name_current]

            if sympy.Symbol(name_new) in clashing_symbols or name_nonmem in names_sset_translated:
                continue

            name_in_sset_current = {v: k for k, v in sset_current.items()}[name_nonmem]
            trans_sset[name_in_sset_current] = name_new
            names_sset_translated.append(name_nonmem)

            if name_nonmem in pset_current.values() and name_new in pset_current.keys():
                names_pset_translated.append(name_nonmem)

        for name_current, name_new in trans_pset_setting.items():
            name_nonmem = pset_current[name_current]

            if sympy.Symbol(name_new) in clashing_symbols or name_nonmem in names_pset_translated:
                continue

            trans_pset[name_current] = name_new
            names_pset_translated.append(name_nonmem)

        if setting == 'basic':
            params_left = [k for k in pset_current.keys() if k not in names_pset_translated]
            params_left += [rv for rv in rvs.names if rv not in names_sset_translated]
            names_basic = [name for name in params_left if name not in names_sset_translated]
            break

    if clashing_symbols:
        warnings.warn(
            f'The parameter names {clashing_symbols} are also names of variables '
            f'in the model code. Falling back to the in naming scheme config '
            f'names for these.'
        )

    names_nonmem_all = rvs.names + [key for key in parameter_translation(control_stream).keys()]

    if set(names_nonmem_all) - set(names_sset_translated + names_pset_translated + names_basic):
        raise ValueError(
            'Mismatch in number of parameter names, all have not been accounted for. If basic '
            'NONMEM-names are desired as fallback, double-check that "basic" is included in '
            'config-settings for parameter_names.'
        )
    return trans_sset, trans_pset


def _name_as_comments(control_stream, statements):
    params_current = parameter_translation(control_stream, remove_idempotent=True)
    for name_abbr, name_nonmem in control_stream.abbreviated.replace.items():
        if name_nonmem in params_current.keys():
            params_current[name_abbr] = params_current.pop(name_nonmem)
    trans_params = {
        name_comment: name_comment
        for name_current, name_comment in params_current.items()
        if sympy.Symbol(name_current) not in statements.free_symbols
    }
    trans_statements = {
        name_current: name_comment
        for name_current, name_comment in params_current.items()
        if sympy.Symbol(name_current) in statements.free_symbols
    }
    return trans_statements, trans_params


def _name_as_abbr(control_stream, rvs):
    pharmpy_names = control_stream.abbreviated.translate_to_pharmpy_names()
    params_current = parameter_translation(control_stream, remove_idempotent=True, reverse=True)
    trans_params = {
        name_nonmem: name_abbr
        for name_nonmem, name_abbr in pharmpy_names.items()
        if name_nonmem in parameter_translation(control_stream).keys() or name_nonmem in rvs.names
    }
    for name_nonmem, name_abbr in params_current.items():
        if name_abbr in trans_params.keys():
            trans_params[name_nonmem] = trans_params.pop(name_abbr)
    trans_statements = {
        name_abbr: pharmpy_names[name_nonmem]
        for name_abbr, name_nonmem in control_stream.abbreviated.replace.items()
    }
    return trans_statements, trans_params


def _name_as_basic(control_stream):
    trans_params = {
        name_current: name_nonmem
        for name_current, name_nonmem in parameter_translation(control_stream, reverse=True).items()
        if name_current != name_nonmem
    }
    trans_statements = control_stream.abbreviated.replace
    return trans_statements, trans_params


def _clashing_symbols(statements, trans_statements):
    # Find symbols in the statements that are also in parameters
    parameter_symbols = {sympy.Symbol(symb) for _, symb in trans_statements.items()}
    clashing_symbols = parameter_symbols & statements.free_symbols
    return clashing_symbols
