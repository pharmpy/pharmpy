from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.internals.fs.path import path_absolute
from pharmpy.internals.immutable import frozenmapping
from pharmpy.internals.math import triangular_root
from pharmpy.model import (
    Assignment,
    ColumnInfo,
    DataInfo,
    DatasetError,
    EstimationStep,
    EstimationSteps,
    JointNormalDistribution,
    ModelSyntaxError,
    NormalDistribution,
    ODESystem,
    Parameter,
    Parameters,
    RandomVariables,
    Statements,
)

from .advan import _compartmental_model, des_assign_statements
from .dataset import read_nonmem_dataset
from .nmtran_parser import NMTranControlStream
from .table import NONMEMTableFile, PhiTable


def parse_thetas(control_stream):
    names = []
    bounds = []
    inits = []
    fixs = []
    for theta_record in control_stream.get_records('THETA'):
        bounds.extend(theta_record.bounds)
        inits.extend(theta_record.inits)
        fixs.extend(theta_record.fixs)
        names.extend(theta_record.comment_names)
    return names, bounds, inits, fixs


def parse_omegas_sigmas(control_stream, record_name):
    blocks = []
    for record in control_stream.get_records(record_name):
        curblocks = record.parse()
        blocks.extend(curblocks)
    return blocks


def parameters_from_blocks(blocks, all_names, record_name):
    row = 1
    col = 1
    prev_size = None
    parameters = []
    name_map = {}
    for names, inits, fix, same in blocks:
        if same:
            if prev_size is None:
                raise ModelSyntaxError(f"First {record_name} block cannot be SAME")
            row += prev_size
            col += prev_size
        else:
            block_row = row
            col = row
            for i, name in enumerate(names):
                if name in all_names:
                    duplicated_name = name
                    name = None
                else:
                    duplicated_name = None
                if name is None:
                    name = f'{record_name}_{row}_{col}'
                    while name in all_names:
                        name += "_"
                if duplicated_name is not None:
                    warnings.warn(
                        f'The parameter name {duplicated_name} is duplicated. '
                        f'Falling back to using {name} instead.'
                    )
                all_names.add(name)
                nonmem_name = f'{record_name}({row},{col})'
                name_map[nonmem_name] = name
                lower = 0 if row == col else None
                parameter = Parameter.create(name, init=inits[i], lower=lower, upper=None, fix=fix)
                parameters.append(parameter)
                if row == col:
                    row += 1
                    col = block_row
                else:
                    col += 1
            prev_size = row - block_row
    return parameters, name_map


def rvs_from_blocks(abbr_names, blocks, parameters, rvtype):
    next_same = False
    parameters_index = 0
    eta_index = 1
    rvs = []
    previous_cov = None
    name_map = {}
    for block_index, (_, inits, _, same) in enumerate(blocks):
        try:
            next_block = blocks[block_index + 1]
        except IndexError:
            next_same = False
        else:
            next_same = next_block[3]

        if not same:
            n = triangular_root(len(inits))

        if rvtype == 'EPS':
            level = 'ruv'
        else:
            if same or next_same:
                level = 'iov'
            else:
                level = 'iiv'

        names = []
        for i in range(n):
            nonmem_name = f'{rvtype}({eta_index + i})'
            if nonmem_name in abbr_names:
                name = abbr_names[nonmem_name]
            else:
                name = f'{rvtype}_{eta_index + i}'
            name_map[nonmem_name] = name
            if rvtype == 'EPS':
                alt_name = f'ERR({eta_index + i})'
                name_map[alt_name] = name
            names.append(name)

        if same:
            cov = previous_cov
        else:
            if n == 1:
                cov = parameters[parameters_index].symbol
                parameters_index += 1
            else:
                cov = sympy.zeros(n)
                for row in range(n):
                    for col in range(0, row + 1):
                        symb = parameters[parameters_index].symbol
                        parameters_index += 1
                        cov[row, col] = symb
                        cov[col, row] = symb

        if n == 1:
            dist = NormalDistribution.create(names[0], level, 0, cov)
        else:
            means = [0] * n
            dist = JointNormalDistribution.create(names, level, means, cov)

        rvs.append(dist)
        previous_cov = cov

        eta_index += n

    return RandomVariables.create(rvs), name_map


def parse_parameters(control_stream, statements):
    symbols = statements.free_symbols
    all_names = {s.name for s in symbols}
    theta_names, theta_bounds, theta_inits, theta_fixs = parse_thetas(control_stream)
    abbr_map = control_stream.abbreviated.translate_to_pharmpy_names()
    theta_parameters = []
    name_map = {}
    for i, name in enumerate(theta_names):
        nonmem_name = f'THETA({i + 1})'
        if nonmem_name in abbr_map:
            name = abbr_map[nonmem_name]
        if name in all_names:
            duplicated_name = name
            name = None
        else:
            duplicated_name = None
        if name is None:
            name = f'THETA_{i + 1}'
            while name in all_names:
                name += "_"
        if duplicated_name is not None:
            warnings.warn(
                f'The parameter name {duplicated_name} is duplicated. '
                f'Falling back to using {name} instead.'
            )
        name_map[nonmem_name] = name
        all_names.add(name)
        parameter = Parameter.create(
            name,
            init=theta_inits[i],
            lower=theta_bounds[i][0],
            upper=theta_bounds[i][1],
            fix=theta_fixs[i],
        )
        theta_parameters.append(parameter)

    omega_blocks = parse_omegas_sigmas(control_stream, "OMEGA")
    sigma_blocks = parse_omegas_sigmas(control_stream, "SIGMA")
    omega_parameters, omega_map = parameters_from_blocks(omega_blocks, all_names, 'OMEGA')
    sigma_parameters, sigma_map = parameters_from_blocks(sigma_blocks, all_names, 'SIGMA')
    name_map.update(omega_map)
    name_map.update(sigma_map)
    parameters = Parameters.create(theta_parameters + omega_parameters + sigma_parameters)

    etas, eta_map = rvs_from_blocks(abbr_map, omega_blocks, omega_parameters, 'ETA')
    epsilons, eps_map = rvs_from_blocks(abbr_map, sigma_blocks, sigma_parameters, 'EPS')
    name_map.update(eta_map)
    name_map.update(eps_map)

    rvs = etas + epsilons

    return parameters, rvs, name_map


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
        des_assign = des_assign_statements(control_stream, des)
        if des_assign is not None:
            for s in des_assign:
                statements += Assignment(s.lhs, s.rhs)
        comp = _compartmental_model(di, dataset, control_stream, sub.advan, sub.trans, des)
        trans_amounts = {}
        if comp is None:
            statements += ODESystem()
            # FIXME: Dummy link statement
            statements += Assignment(sympy.Symbol('F'), sympy.Symbol('F'))
        else:
            cm, link, comp_map = comp
            statements += [cm, link]
            if des:
                rec_model = control_stream.get_records('MODEL')[0]
                comps = [c for c, _ in rec_model.compartments()]
                for i, c in enumerate(comps, 1):
                    trans_amounts[sympy.Symbol(f"A({i})")] = sympy.Function(f'A_{c}')(
                        sympy.Symbol('t')
                    )
                    trans_amounts[sympy.Symbol(f"A_0({i})")] = sympy.Function(f'A_{c}')(0)
            else:
                for i, amount in enumerate(cm.amounts, start=1):
                    trans_amounts[sympy.Symbol(f"A({i})")] = sympy.Function(amount.name)(
                        sympy.Symbol('t')
                    )
                    trans_amounts[sympy.Symbol(f"A_0({i})")] = sympy.Function(amount.name)(0)

        statements += error.statements
        if trans_amounts:
            statements = statements.subs(trans_amounts)

    return statements, comp_map


def convert_dvs(statements, control_stream):
    # Conversion of IF (DVID.EQ.n) THEN to non-piecewise
    # could partly be done in code_record
    after = statements.error
    kept = []
    dvs = frozenmapping({sympy.Symbol('Y'): 1})
    obs_trans = frozenmapping({sympy.Symbol('Y'): sympy.Symbol('Y')})
    change = False
    yind = after.find_assignment_index('Y')
    if yind is None:
        return statements, dvs, obs_trans
    yind = yind - 1
    for s in after:
        expr = s.expression
        if isinstance(expr, sympy.Piecewise) and sympy.Symbol("DVID") in expr.free_symbols:
            cond = expr.args[0][1]
            if cond.lhs == sympy.Symbol("DVID") and cond.rhs == 1:
                ass1 = s.replace(symbol=sympy.Symbol('Y_1'), expression=expr.args[0][0])
                ass2 = s.replace(symbol=sympy.Symbol('Y_2'), expression=expr.args[1][0])
                kept.append(ass1)
                kept.append(ass2)
                dvs = frozenmapping({sympy.Symbol('Y_1'): 1, sympy.Symbol('Y_2'): 2})
                obs_trans = {sympy.Symbol('Y_1'): sympy.Symbol('Y_1')}
                change = True
                continue
        kept.append(s)
    after = Statements(tuple(kept))
    if statements.ode_system is not None:
        statements = statements.before_odes + statements.ode_system + after
        if change:  # $ERROR
            rec = control_stream.get_records('ERROR')[0]
    else:
        statements = after
        if change:  # $PRED
            rec = control_stream.get_records('PRED')[0]
    if change:
        rec._statements = after[1:]
        first = True
        inds = []
        for ni, nj, si, sj in rec._index:
            if first and si == yind:
                first = False
                ind = (ni, nj, si, si + 2)
            else:
                if first:
                    ind = (ni, nj, si, si)
                else:
                    ind = (ni, nj, si + 2, sj + 2)
            inds.append(ind)
        rec._index = inds
    return statements, dvs, obs_trans


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
        cov = None
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
            cov = 'SANDWICH'
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
            meth = EstimationStep.create(
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

    steps = EstimationSteps.create(steps)

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


def parse_initial_individual_estimates(
    control_stream, name_map, basepath
) -> Optional[pd.DataFrame]:
    """Initial individual estimates

    These are taken from the $ETAS FILE.
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
        try:
            phi_tables = NONMEMTableFile(path)
        except FileNotFoundError:
            return None
        phitab = phi_tables[0]
        assert isinstance(phitab, PhiTable)
        df = phitab.etas.rename(columns=name_map)
        return df
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
    di_nonmem = create_nonmem_datainfo(control_stream, resolved_dataset_path)
    (colnames, drop, replacements, _) = parse_column_info(control_stream)

    if resolved_dataset_path is None:
        return di_nonmem

    dipath = resolved_dataset_path.with_suffix('.datainfo')
    if dipath.is_file():
        di_pharmpy = DataInfo.read_json(dipath)
        di_pharmpy = di_pharmpy.replace(path=resolved_dataset_path)
        di_pharmpy = validate_datainfo(di_pharmpy, di_nonmem)
        return di_pharmpy
    return di_nonmem


def validate_datainfo(di_pharmpy, di_nonmem):
    different_drop, cols_new = [], []
    for col_pharmpy, col_nonmem in zip(di_pharmpy, di_nonmem):
        if col_pharmpy.drop != col_nonmem.drop:
            colinfo_new = col_pharmpy.replace(drop=col_nonmem.drop)
            cols_new.append(colinfo_new)
            if col_nonmem.datatype != 'nmtran-date':
                different_drop.append(col_pharmpy.name)
        else:
            cols_new.append(col_pharmpy)

    if cols_new:
        di_new = di_pharmpy.replace(columns=tuple(cols_new))
        if different_drop:
            warnings.warn(
                "NONMEM .mod and dataset .datainfo disagree on "
                f"DROP for columns {', '.join(different_drop)}."
            )
        return di_new
    return di_pharmpy


def create_nonmem_datainfo(control_stream, resolved_dataset_path):
    (colnames, drop, replacements, _) = parse_column_info(control_stream)

    column_info = []
    have_pk = control_stream.get_pk_record()
    for colname, coldrop in zip(colnames, drop):
        if coldrop and colname not in ['DATE', 'DAT1', 'DAT2', 'DAT3']:
            info = ColumnInfo.create(colname, drop=coldrop, datatype='str')
        elif colname == 'ID' or colname == 'L1':
            info = ColumnInfo.create(
                colname, drop=coldrop, datatype='int32', type='id', scale='nominal'
            )
        elif colname == 'DV' or colname == replacements.get('DV', None):
            info = ColumnInfo.create(colname, drop=coldrop, type='dv')
        elif colname == 'TIME' or colname == replacements.get('TIME', None):
            if not set(colnames).isdisjoint({'DATE', 'DAT1', 'DAT2', 'DAT3'}):
                datatype = 'nmtran-time'
            else:
                datatype = 'float64'
            info = ColumnInfo.create(
                colname, drop=coldrop, type='idv', scale='ratio', datatype=datatype
            )
        elif colname in ['DATE', 'DAT1', 'DAT2', 'DAT3']:
            # Always DROP in mod-file, but actually always used
            info = ColumnInfo.create(colname, drop=False, scale='interval', datatype='nmtran-date')
        elif colname == 'EVID' and have_pk:
            info = ColumnInfo.create(colname, drop=coldrop, type='event', scale='nominal')
        elif colname == 'MDV' and have_pk:
            if 'EVID' in colnames:
                tp = 'mdv'
            else:
                tp = 'event'
            info = ColumnInfo.create(
                colname, drop=coldrop, type=tp, scale='nominal', datatype='int32'
            )
        elif colname == 'II' and have_pk:
            info = ColumnInfo.create(colname, drop=coldrop, type='ii', scale='ratio')
        elif colname == 'SS' and have_pk:
            info = ColumnInfo.create(colname, drop=coldrop, type='ss', scale='nominal')
        elif colname == 'ADDL' and have_pk:
            info = ColumnInfo.create(colname, drop=coldrop, type='additional', scale='ordinal')
        elif (colname == 'AMT' or colname == replacements.get('AMT', None)) and have_pk:
            info = ColumnInfo.create(colname, drop=coldrop, type='dose', scale='ratio')
        elif colname == 'CMT' and have_pk:
            info = ColumnInfo.create(colname, drop=coldrop, type='compartment', scale='nominal')
        elif colname == 'RATE' and have_pk:
            info = ColumnInfo.create(colname, drop=coldrop, type='rate')
        elif colname == 'BLQ':
            info = ColumnInfo.create(colname, drop=coldrop, type='blq', scale='nominal')
        elif colname == 'LLOQ':
            info = ColumnInfo.create(colname, drop=coldrop, type='lloq')
        else:
            info = ColumnInfo.create(colname, drop=coldrop)
        column_info.append(info)

    di = DataInfo.create(column_info, path=resolved_dataset_path)
    return di


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
    if not data_records:
        return None
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
        df = filter_observations(df, col_names)
    return df


def filter_observations(df, col_names):
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
    return df[~df['ID'].isin(ids_to_remove)]


def parse_table_columns(control_stream, netas):
    # Handle synonyms and appended columns

    reserved = {'PRED', 'IPRED', 'CIPREDI'}
    synonyms = dict()
    all_columns = []

    code_recs = (
        control_stream.get_records('PK')
        + control_stream.get_records('PRED')
        + control_stream.get_records('ERROR')
    )
    symbs = set()
    for rec in code_recs:
        for s in rec.statements:
            symbs.add(s.symbol.name)

    (colnames, _, _, _) = parse_column_info(control_stream)
    symbs |= set(colnames)

    for table_record in control_stream.get_records('TABLE'):
        noappend = False
        columns = []
        for opt, key, value in table_record.parse_options(nonoptions=symbs, netas=netas):
            if key == 'NOAPPEND':
                noappend = True
            elif opt.need_value is None:
                if value is None:
                    if key in synonyms:
                        columns.append(synonyms[key])
                    else:
                        columns.append(key)
                else:
                    if key in reserved:
                        columns.append(key)
                        synonyms[value] = key
                    elif value in reserved:
                        columns.append(value)
                        synonyms[key] = value
                    else:
                        # FIXME: Fallback since we don't know all reserved names
                        columns.append(key)

        if not noappend:
            toremove = ['PRED', 'RES', 'WRES']
            toappend = ['DV'] + toremove
            # Remove appended columns explicitly in $TABLE except DV
            columns = [col for col in columns if col not in toremove]
            columns.extend(toappend)

        all_columns.append(columns)

    return all_columns
