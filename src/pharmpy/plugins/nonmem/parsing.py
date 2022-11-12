from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pharmpy.plugins.nonmem
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.internals.fs.path import path_absolute
from pharmpy.model import (
    Assignment,
    ColumnInfo,
    DataInfo,
    DatasetError,
    EstimationStep,
    EstimationSteps,
    ExplicitODESystem,
    ModelSyntaxError,
    Parameters,
    RandomVariables,
    Statements,
)
from pharmpy.plugins.nonmem.table import NONMEMTableFile, PhiTable

from .advan import _compartmental_model
from .dataset import read_nonmem_dataset
from .nmtran_parser import NMTranControlStream
from .parameters import parameter_translation


def parse_parameters(control_stream) -> Parameters:
    next_theta = 1
    params = []
    for theta_record in control_stream.get_records('THETA'):
        thetas = theta_record.parameters(next_theta, seen_labels={p.name for p in params})
        params.extend(thetas)
        next_theta += len(thetas)
    next_omega = 1
    previous_size = None
    for omega_record in control_stream.get_records('OMEGA'):
        omegas, next_omega, previous_size = omega_record.parameters(
            next_omega, previous_size, seen_labels={p.name for p in params}
        )
        params.extend(omegas)
    next_sigma = 1
    previous_size = None
    for sigma_record in control_stream.get_records('SIGMA'):
        sigmas, next_sigma, previous_size = sigma_record.parameters(
            next_sigma, previous_size, seen_labels={p.name for p in params}
        )
        params.extend(sigmas)
    return Parameters(params)


def parse_random_variables(control_stream) -> RandomVariables:
    dists = RandomVariables.create(())
    next_omega = 1
    prev_start = 1
    prev_cov = None

    for omega_record in control_stream.get_records('OMEGA'):
        etas, next_omega, prev_start, prev_cov, _ = omega_record.random_variables(
            next_omega, prev_start, prev_cov
        )
        dists += etas
    dists = _adjust_iovs(dists)
    next_sigma = 1
    prev_start = 1
    prev_cov = None
    for sigma_record in control_stream.get_records('SIGMA'):
        epsilons, next_sigma, prev_start, prev_cov, _ = sigma_record.random_variables(
            next_sigma, prev_start, prev_cov
        )
        dists += epsilons
    rvs = RandomVariables.create(dists)
    return rvs


def parse_statements(
    di: DataInfo,
    dataset: Callable[[], pd.DataFrame],
    control_stream: NMTranControlStream,
) -> Tuple[Statements, Optional[Dict[str, int]]]:
    rec = control_stream.get_pred_pk_record()
    statements = rec.statements

    des = control_stream.get_des_record()
    error = control_stream.get_error_record()
    comp_map = None

    if error:
        sub = control_stream.get_records('SUBROUTINES')[0]
        comp = _compartmental_model(di, dataset, control_stream, sub.advan, sub.trans, des)
        trans_amounts = {}
        if comp is None:
            statements += ExplicitODESystem((), {})  # FIXME: Placeholder for ODE-system
            # FIXME: Dummy link statement
            statements += Assignment(sympy.Symbol('F'), sympy.Symbol('F'))
        else:
            cm, link, comp_map = comp
            statements += [cm, link]
            for i, amount in enumerate(cm.amounts, start=1):
                trans_amounts[sympy.Symbol(f"A({i})")] = amount
        statements += error.statements
        if trans_amounts:
            statements = statements.subs(trans_amounts)

    return statements, comp_map


def _adjust_iovs(rvs: RandomVariables) -> RandomVariables:
    n = len(rvs)
    if n <= 1:
        return rvs

    updated = []
    for i in range(n - 1):
        dist, next_dist = rvs[i], rvs[i + 1]
        if dist.level != 'IOV' and next_dist.level == 'IOV':
            # NOTE The first distribution for an IOV will have been parsed as
            # IIV since we did not know what came after.
            new_dist = dist.derive(level='IOV')
            updated.append(new_dist)
        else:
            updated.append(dist)

    updated.append(rvs[-1])  # NOTE The last distribution does not need update
    return RandomVariables.create(updated)


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

    trans_sset, trans_pset = {}, {}
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

    names_nonmem_all = rvs.names + list(parameter_translation(control_stream).keys())

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


def parse_value_type(control_stream, statements):
    ests = control_stream.get_records('ESTIMATION')
    # Assuming that a model cannot be fully likelihood or fully prediction
    # at the same time
    for est in ests:
        if est.likelihood:
            tp = 'LIKELIHOOD'
            break
        elif est.loglikelihood:
            tp = '-2LL'
            break
    else:
        tp = 'PREDICTION'
    f_flag = sympy.Symbol('F_FLAG')
    if f_flag in statements.free_symbols:
        tp = f_flag
    return tp


def parse_description(control_stream) -> str:
    rec = control_stream.get_records('PROBLEM')[0]
    return rec.title


def parse_estimation_steps(control_stream, random_variables) -> EstimationSteps:
    steps = []
    records = control_stream.get_records('ESTIMATION')
    covrec = control_stream.get_records('COVARIANCE')
    solver, tol, atol = parse_solver(control_stream)

    # Read eta and epsilon derivatives
    etaderiv_names = None
    epsilonderivs_names = None
    table_records = control_stream.get_records('TABLE')
    for table in table_records:
        etaderivs = table.eta_derivatives
        if etaderivs:
            etas = random_variables.etas
            etaderiv_names = [etas.names[i - 1] for i in etaderivs]
        epsderivs = table.epsilon_derivatives
        if epsderivs:
            epsilons = random_variables.epsilons
            epsilonderivs_names = [epsilons.names[i - 1] for i in epsderivs]

    for record in records:
        value = record.get_option('METHOD')
        if value is None or value == '0' or value == 'ZERO':
            name = 'fo'
        elif value == '1' or value == 'CONDITIONAL' or value == 'COND':
            name = 'foce'
        else:
            name = value
        interaction = False
        evaluation = False
        maximum_evaluations = None
        cov = False
        laplace = False
        isample = None
        niter = None
        auto = None
        keep_every_nth_iter = None

        if record.has_option('INTERACTION') or record.has_option('INTER'):
            interaction = True
        maxeval_opt = record.get_option('MAXEVAL') if not None else record.get_option('MAXEVALS')
        if maxeval_opt is not None:
            if (name.upper() == 'FO' or name.upper() == 'FOCE') and int(maxeval_opt) == 0:
                evaluation = True
            else:
                maximum_evaluations = int(maxeval_opt)
        eval_opt = record.get_option('EONLY')
        if eval_opt is not None and int(eval_opt) == 1:
            evaluation = True
        if covrec:
            cov = True
        if record.has_option('LAPLACIAN') or record.has_option('LAPLACE'):
            laplace = True
        if record.has_option('ISAMPLE'):
            isample = int(record.get_option('ISAMPLE'))
        if record.has_option('NITER'):
            niter = int(record.get_option('NITER'))
        if record.has_option('AUTO'):
            auto_opt = record.get_option('AUTO')
            if auto_opt is not None and int(auto_opt) in [0, 1]:
                auto = bool(auto_opt)
            else:
                raise ValueError('Currently only AUTO=0 and AUTO=1 is supported')
        if record.has_option('PRINT'):
            keep_every_nth_iter = int(record.get_option('PRINT'))

        protected_names = [
            name.upper(),
            'EONLY',
            'INTERACTION',
            'INTER',
            'LAPLACE',
            'LAPLACIAN',
            'MAXEVAL',
            'MAXEVALS',
            'METHOD',
            'METH',
            'ISAMPLE',
            'NITER',
            'AUTO',
            'PRINT',
        ]

        tool_options = {
            option.key: option.value
            for option in record.all_options
            if option.key not in protected_names
        }
        if not tool_options:
            tool_options = None

        try:
            meth = EstimationStep(
                name,
                interaction=interaction,
                cov=cov,
                evaluation=evaluation,
                maximum_evaluations=maximum_evaluations,
                laplace=laplace,
                isample=isample,
                niter=niter,
                auto=auto,
                keep_every_nth_iter=keep_every_nth_iter,
                tool_options=tool_options,
                solver=solver,
                solver_rtol=tol,
                solver_atol=atol,
                eta_derivatives=etaderiv_names,
                epsilon_derivatives=epsilonderivs_names,
            )
        except ValueError:
            raise ModelSyntaxError(f'Non-recognized estimation method in: {str(record.root)}')
        steps.append(meth)

    steps = EstimationSteps(steps)

    return steps


def parse_solver(control_stream):
    subs_records = control_stream.get_records('SUBROUTINES')
    if not subs_records:
        return None, None, None
    record = subs_records[0]
    advan = record.advan
    # Currently only reading non-linear solvers
    # These can then be used if the model needs to use a non-linear solver
    if advan == 'ADVAN6':
        solver = 'DVERK'
    elif advan == 'ADVAN8':
        solver = 'DGEAR'
    elif advan == 'ADVAN9':
        solver = 'LSODI'
    elif advan == 'ADVAN13':
        solver = 'LSODA'
    elif advan == 'ADVAN14':
        solver = 'CVODES'
    elif advan == 'ADVAN15':
        solver = 'IDA'
    else:
        solver = None
    return solver, record.tol, record.atol


def parse_initial_individual_estimates(control_stream, rvs, basepath) -> Optional[pd.DataFrame]:
    """Initial individual estimates

    These are taken from the $ETAS FILE. 0 FIX ETAs are removed.
    If no $ETAS is present None will be returned.

    Setter assumes that all IDs are present
    """
    etas = control_stream.get_records('ETAS')
    if etas:
        path = Path(etas[0].path)
        if not path.is_absolute():
            if basepath is None:
                raise ValueError("Cannot resolve path for $ETAS")
            path = path_absolute(basepath / path)
        phi_tables = NONMEMTableFile(path)
        rv_names = [rv for rv in rvs.names if rv.startswith('ETA')]
        phitab = phi_tables[0]
        assert isinstance(phitab, PhiTable)
        names = [name for name in rv_names if name in phitab.etas.columns]
        return phitab.etas[names]
    else:
        return None


def parse_dataset_path(control_stream, basepath) -> Optional[Path]:
    record = next(iter(control_stream.get_records('DATA')), None)

    if record is None:
        return None

    path = Path(record.filename)
    if basepath is not None and not path.is_absolute():
        path = basepath.parent / path

    return path_absolute(path)


def _synonym(key, value):
    """Return a tuple reserved name and synonym"""
    _reserved_column_names = [
        'ID',
        'L1',
        'L2',
        'DV',
        'MDV',
        'RAW_',
        'MRG_',
        'RPT_',
        'TIME',
        'DATE',
        'DAT1',
        'DAT2',
        'DAT3',
        'EVID',
        'AMT',
        'RATE',
        'SS',
        'II',
        'ADDL',
        'CMT',
        'PCMT',
        'CALL',
        'CONT',
    ]
    if key in _reserved_column_names:
        return (key, value)
    elif value in _reserved_column_names:
        return (value, key)
    else:
        raise DatasetError(
            f'A column name "{key}" in $INPUT has a synonym to a non-reserved '
            f'column name "{value}"'
        )


def parse_column_info(control_stream):
    """List all column names in order.
    Use the synonym when synonym exists.
    return tuple of two lists, colnames, and drop together with a dictionary
    of replacements for reserved names (aka synonyms).
    Anonymous columns, i.e. DROP or SKIP alone, will be given unique names _DROP1, ...
    """
    input_records = control_stream.get_records("INPUT")
    colnames = []
    drop = []
    synonym_replacement = {}
    given_names = []
    next_anonymous = 1
    for record in input_records:
        for key, value in record.all_options:
            if value:
                if key == 'DROP' or key == 'SKIP':
                    colnames.append(value)
                    given_names.append(value)
                    drop.append(True)
                elif value == 'DROP' or value == 'SKIP':
                    colnames.append(key)
                    given_names.append(key)
                    drop.append(True)
                else:
                    (reserved_name, synonym) = _synonym(key, value)
                    synonym_replacement[reserved_name] = synonym
                    given_names.append(synonym)
                    colnames.append(synonym)
                    drop.append(False)
            else:
                if key == 'DROP' or key == 'SKIP':
                    name = f'_DROP{next_anonymous}'
                    next_anonymous += 1
                    colnames.append(name)
                    given_names.append(None)
                    drop.append(True)
                else:
                    colnames.append(key)
                    given_names.append(key)
                    drop.append(False)
    return colnames, drop, synonym_replacement, given_names


def parse_datainfo(control_stream, path) -> DataInfo:
    resolved_dataset_path = parse_dataset_path(control_stream, path)
    (colnames, drop, replacements, _) = parse_column_info(control_stream)

    if resolved_dataset_path is not None:
        dipath = resolved_dataset_path.with_suffix('.datainfo')

        if dipath.is_file():
            di = DataInfo.read_json(dipath)
            di = di.derive(path=resolved_dataset_path)
            different_drop, cols_new = [], []
            for colinfo, coldrop in zip(di, drop):
                if colinfo.drop != coldrop:
                    colinfo_new = colinfo.derive(drop=coldrop)
                    different_drop.append(colinfo.name)
                    cols_new.append(colinfo_new)
                else:
                    cols_new.append(colinfo)

            if different_drop:
                di_new = di.derive(columns=tuple(cols_new))
                warnings.warn(
                    "NONMEM .mod and dataset .datainfo disagree on "
                    f"DROP for columns {', '.join(different_drop)}."
                )
                return di_new

            return di

    column_info = []
    have_pk = control_stream.get_pk_record()
    for colname, coldrop in zip(colnames, drop):
        if coldrop and colname not in ['DATE', 'DAT1', 'DAT2', 'DAT3']:
            info = ColumnInfo(colname, drop=coldrop, datatype='str')
        elif colname == 'ID' or colname == 'L1':
            info = ColumnInfo(colname, drop=coldrop, datatype='int32', type='id', scale='nominal')
        elif colname == 'DV' or colname == replacements.get('DV', None):
            info = ColumnInfo(colname, drop=coldrop, type='dv')
        elif colname == 'TIME' or colname == replacements.get('TIME', None):
            if not set(colnames).isdisjoint({'DATE', 'DAT1', 'DAT2', 'DAT3'}):
                datatype = 'nmtran-time'
            else:
                datatype = 'float64'
            info = ColumnInfo(colname, drop=coldrop, type='idv', scale='ratio', datatype=datatype)
        elif colname in ['DATE', 'DAT1', 'DAT2', 'DAT3']:
            # Always DROP in mod-file, but actually always used
            info = ColumnInfo(colname, drop=False, scale='interval', datatype='nmtran-date')
        elif colname == 'EVID' and have_pk:
            info = ColumnInfo(colname, drop=coldrop, type='event', scale='nominal')
        elif colname == 'MDV' and have_pk:
            if 'EVID' in colnames:
                tp = 'mdv'
            else:
                tp = 'event'
            info = ColumnInfo(colname, drop=coldrop, type=tp, scale='nominal', datatype='int32')
        elif colname == 'II' and have_pk:
            info = ColumnInfo(colname, drop=coldrop, type='ii', scale='ratio')
        elif colname == 'SS' and have_pk:
            info = ColumnInfo(colname, drop=coldrop, type='ss', scale='nominal')
        elif colname == 'ADDL' and have_pk:
            info = ColumnInfo(colname, drop=coldrop, type='additional', scale='ordinal')
        elif (colname == 'AMT' or colname == replacements.get('AMT', None)) and have_pk:
            info = ColumnInfo(colname, drop=coldrop, type='dose', scale='ratio')
        elif colname == 'CMT' and have_pk:
            info = ColumnInfo(colname, drop=coldrop, type='compartment', scale='nominal')
        elif colname == 'RATE' and have_pk:
            info = ColumnInfo(colname, drop=coldrop, type='rate')
        else:
            info = ColumnInfo(colname, drop=coldrop)
        column_info.append(info)

    di = DataInfo(column_info, path=resolved_dataset_path)
    return di


def get_zero_fix_rvs(control_stream, eta=True):
    zero_fix = []
    if eta:
        prev_cov = None
        next_omega = 1
        prev_start = 1
        for omega_record in control_stream.get_records('OMEGA'):
            _, next_omega, prev_start, prev_cov, new_zero_fix = omega_record.random_variables(
                next_omega, prev_start, prev_cov
            )
            zero_fix += new_zero_fix
    else:
        prev_cov = None
        next_sigma = 1
        prev_start = 1
        for sigma_record in control_stream.get_records('SIGMA'):
            _, next_sigma, prev_start, prev_cov, new_zero_fix = sigma_record.random_variables(
                next_sigma, prev_start, prev_cov
            )
            zero_fix += new_zero_fix
    return zero_fix


def replace_synonym_in_filters(filters, replacements):
    result = []
    for f in filters:
        col = f.leaf('COLUMN').value
        if col in replacements:
            s = ''
            for child in f.children:
                if child.rule == 'COLUMN':
                    value = replacements[col]
                else:
                    value = str(child)
                s += value
        else:
            s = str(f)
        result.append(s)
    return result


def parse_dataset(
    di: DataInfo,
    control_stream: NMTranControlStream,
    raw: bool = False,
    parse_columns: Tuple[str, ...] = (),
):
    data_records = control_stream.get_records('DATA')
    ignore_character = data_records[0].ignore_character
    null_value = data_records[0].null_value
    (colnames, drop, replacements, _) = parse_column_info(control_stream)

    if raw:
        ignore = None
        accept = None
    else:
        # FIXME: All direct handling of control stream spanning
        # over one or more records should move
        ignore = data_records[0].ignore
        accept = data_records[0].accept
        # FIXME: This should really only be done if setting the dataset
        if ignore:
            ignore = replace_synonym_in_filters(ignore, replacements)
        else:
            accept = replace_synonym_in_filters(accept, replacements)

    df = read_nonmem_dataset(
        di.path,
        raw,
        ignore_character,
        colnames,
        drop,
        null_value=null_value,
        parse_columns=parse_columns,
        ignore=ignore,
        accept=accept,
        dtype=None if raw else di.get_dtype_dict(),
    )
    # Let TIME be the idv in both $PK and $PRED models
    # Remove individuals without observations
    col_names = list(df.columns)
    have_pk = control_stream.get_pk_record()
    if have_pk:
        if 'EVID' in col_names:
            df_obs = df.astype({'EVID': 'float'}).query('EVID == 0')
        elif 'MDV' in col_names:
            df_obs = df.astype({'MDV': 'float'}).query('MDV == 0')
        elif 'AMT' in col_names:
            df_obs = df.astype({'AMT': 'float'}).query('AMT == 0')
        else:
            raise DatasetError('Could not identify observation rows in dataset')
        have_obs = set(df_obs['ID'].unique())
        all_ids = set(df['ID'].unique())
        ids_to_remove = all_ids - have_obs
        df = df[~df['ID'].isin(ids_to_remove)]
    return df
