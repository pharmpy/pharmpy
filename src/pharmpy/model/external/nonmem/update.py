from __future__ import annotations

import re
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy, sympy_printing
from pharmpy.internals.code_generator import CodeGenerator
from pharmpy.internals.parse import AttrTree
from pharmpy.internals.sequence.lcs import diff
from pharmpy.model import (
    Assignment,
    Bolus,
    Compartment,
    CompartmentalSystem,
    CompartmentalSystemBuilder,
    Distribution,
    Infusion,
    ODESystem,
    Parameter,
    Parameters,
    RandomVariables,
    Statements,
    data,
    output,
)
from pharmpy.model.model import update_datainfo
from pharmpy.modeling import get_admid, get_ids, simplify_expression

from .records.parsers import CodeRecordParser

if TYPE_CHECKING:
    from .model import Model

from .nmtran_parser import NMTranControlStream
from .parsing import parse_column_info
from .records import code_record
from .records.code_record import CodeRecord
from .records.etas_record import EtasRecord
from .records.factory import create_record
from .records.model_record import ModelRecord
from .records.sizes_record import SizesRecord
from .table import NONMEMTableFile, PhiTable


def update_description(control_stream, old, new):
    if new != old:
        probrec = control_stream.get_records('PROBLEM')[0]
        new = probrec.set_title(new)
        control_stream = control_stream.replace_records([probrec], [new])
    return control_stream


def reorder_diff(diff, kept_names):
    # Reorder diff so that add comes just before the corresponding remove
    # Allow other remove inbetween
    new_diff = []
    diff = list(diff)
    handled = set()
    for i, (op, param) in enumerate(diff):
        if op == -1 and param.name in kept_names:
            for j in range(i + 1, len(diff)):
                curop = diff[j][0]
                curpar = diff[j][1]
                if curop == 1 and curpar.name == param.name:
                    new_diff.append((curop, curpar))
                    new_diff.append((op, param))
                    handled.add(j)
                    break
                elif curop == 0:
                    new_diff.append((op, param))
                    break
            else:
                new_diff.append((op, param))
        else:
            if i not in handled:
                new_diff.append((op, param))
    return new_diff


def update_thetas(model: Model, control_stream, old: Parameters, new: Parameters):
    new_thetas = [p for p in new if p.symbol not in model.random_variables.free_symbols]
    old_thetas = [
        p for p in old if p.symbol not in model.internals.old_random_variables.free_symbols
    ]

    diff_thetas = diff(old_thetas, new_thetas)
    theta_records = control_stream.get_records('THETA')
    record_index = 0
    old_theta_names = {p.name for p in old_thetas}
    new_theta_names = {p.name for p in new_thetas}
    kept_theta_names = old_theta_names.intersection(new_theta_names)

    new_diff = reorder_diff(diff_thetas, kept_theta_names)

    new_theta_records = []
    cur_to_change = []
    cur_to_remove = []
    i = 0

    for op, param in new_diff:
        if op == 1:
            if param.name in kept_theta_names:
                # Changed
                cur_to_change.append(param)
                i += 1
            else:
                # Added
                new = create_theta_record(param)
                new_theta_records.append(new)
        elif op == -1:
            if param.name not in kept_theta_names:
                # Removed
                cur_to_remove.append(i)
                i += 1
            else:
                # Changed: handled in + case
                pass
        else:
            record = theta_records[record_index]
            if len(record) == 1:
                new_theta_records.append(record)
                record_index += 1
            else:
                cur_to_change.append(param)
                i += 1
        if record_index < len(theta_records) and len(theta_records[record_index]) == i:
            if len(cur_to_remove) != len(theta_records[record_index]):
                # Don't remove all
                new = theta_records[record_index].remove(cur_to_remove).update(cur_to_change)
                new_theta_records.append(new)
            i = 0
            cur_to_remove = []
            cur_to_change = []
            record_index += 1

    control_stream = control_stream.replace_all('THETA', new_theta_records)
    return control_stream


def update_random_variables(model: Model, old: RandomVariables, new: RandomVariables):
    _validate_eta_names(new.names)

    rvs_diff_eta = diff(old.etas, new.etas)
    new_omegas = update_random_variable_records(model, rvs_diff_eta, 'OMEGA')
    control_stream = model.internals.control_stream.replace_all('OMEGA', new_omegas)

    rvs_diff_eps = diff(old.epsilons, new.epsilons)
    new_sigmas = update_random_variable_records(model, rvs_diff_eps, 'SIGMA')
    control_stream = control_stream.replace_all('SIGMA', new_sigmas)
    return control_stream


def _validate_eta_names(names):
    if any(name.lower() == 'eta' for name in names):
        raise ValueError('NONMEM does not allow etas named `eta`')


def update_random_variable_records(model: Model, rvs_diff, record_type):
    records = model.internals.control_stream.get_records(record_type)
    kept = []
    recindex = 0
    diag_index = 0  # DIAG(n) counter
    diag_remove = []
    diag_change = []

    if record_type == 'OMEGA':
        old_names = set(model.internals.old_random_variables.etas.parameter_names)
        new_names = set(model.random_variables.etas.parameter_names)
    else:
        old_names = set(model.internals.old_random_variables.epsilons.parameter_names)
        new_names = set(model.random_variables.epsilons.parameter_names)
    kept_names = old_names.intersection(new_names)

    eta_number = 1

    rvs_diff = list(rvs_diff)

    recindex = 0
    for op, rvs in rvs_diff:
        in_diag = recindex < len(records) and len(rvs) == 1 and len(records[recindex]) > 1
        if op == 1:
            if rvs in model.internals.old_random_variables and set(rvs.parameter_names).issubset(
                kept_names
            ):
                # Changed
                if in_diag:
                    param = model.parameters[rvs.variance.name]
                    diag_change.append(param)
                else:
                    params = []
                    if len(rvs) > 1:
                        for row in range(0, len(rvs)):
                            for col in range(0, row + 1):
                                param = model.parameters[rvs.variance[row, col].name]
                                params.append(param)
                    else:
                        param = model.parameters[rvs.variance.name]
                        params.append(param)
                    newrec = records[recindex].update(params)
                    kept.append(newrec)
                    recindex += 1
            else:
                # Added
                if len(rvs) == 1:
                    newrec = create_omega_single(model, rvs, eta_number)
                else:
                    newrec = create_omega_block(model, rvs, eta_number)
                kept.append(newrec)
            eta_number += len(rvs)
        elif op == -1:
            if not (
                rvs in model.internals.old_random_variables
                and set(rvs.parameter_names).issubset(kept_names)
            ):
                # Removed
                if in_diag:
                    diag_remove.append((diag_index, 0))
                    diag_index += 1
                else:
                    recindex += 1
            else:
                # Changed handled in + case
                pass
        else:
            if in_diag:
                param = model.parameters[rvs.variance.name]
                diag_change.append(param)
                diag_index += 1
            else:
                params = []
                if len(rvs) > 1:
                    for row in range(0, len(rvs)):
                        for col in range(0, row + 1):
                            param = model.parameters[rvs.variance[row, col].name]
                            params.append(param)
                else:
                    param = model.parameters[rvs.variance.name]
                    params.append(param)
                new = records[recindex].update(params)
                kept.append(new)
                recindex += 1
            eta_number += len(rvs)
        if recindex < len(records) and diag_index == len(records[recindex]):
            if len(diag_remove) != len(records[recindex]):
                newrec = records[recindex].remove(diag_remove)
                if diag_change:
                    newrec = newrec.update(diag_change)
                kept.append(newrec)
            diag_index = 0
            diag_remove = []
            diag_change = []
            recindex += 1
    return kept


def create_theta_record(param: Parameter):
    code = '$THETA  '

    if param.init == 0.0:
        init = 0
    else:
        init = param.init

    if param.lower == 0.0:
        lower = 0
    else:
        lower = param.lower

    if param.upper == 0.0:
        upper = 0
    else:
        upper = param.upper

    if upper < 1000000:
        if lower <= -1000000:
            code += f'(-INF,{init},{upper})'
        else:
            code += f'({lower},{init},{upper})'
    else:
        if lower <= -1000000:
            code += f'{init}'
        else:
            code += f'({lower},{init})'
    if param.fix:
        code += ' FIX'

    code += f' ; {param.name}\n'
    record = create_record(code)
    return record


def create_omega_single(model: Model, rv: Distribution, eta_number: int):
    rvs, pset = model.random_variables, model.parameters

    if rv.level == 'RUV':
        record_type = 'SIGMA'
    else:
        record_type = 'OMEGA'

    variance_param = pset[rv.parameter_names[0]]

    no_name = False
    if rv.level == 'IOV':
        code = f'${record_type}  BLOCK(1)'
        first_iov = next(filter(lambda iov: iov.parameter_names == rv.parameter_names, rvs.iov))
        if rv == first_iov:
            code += f'\n{variance_param.init}'
        else:
            no_name = True
            code += ' SAME'
    else:
        code = f'${record_type}  {variance_param.init}'

    if variance_param.fix:
        code += " FIX"

    if (
        not re.match(f'{record_type}_{eta_number}_{eta_number}', variance_param.name)
        and not no_name
    ):
        code += f' ; {variance_param.name}'

    code += '\n'
    record = create_record(code)
    return record


def create_omega_block(model: Model, distribution: Distribution, eta_number: int):
    rvs = RandomVariables.create([distribution])
    cm = rvs.covariance_matrix

    rv = rvs[0]

    if rv.level == 'RUV':
        record_type = 'SIGMA'
    else:
        record_type = 'OMEGA'

    code = f'${record_type} BLOCK({cm.shape[0]})'

    if rv.level == 'IOV' and rv != next(
        filter(lambda iov: iov.parameter_names == rv.parameter_names, model.random_variables.iov)
    ):
        code += ' SAME\n'
    else:
        code += '\n'
        for row in range(cm.shape[0]):
            for col in range(row + 1):
                elem = cm.row(row).col(col)
                name = str(elem[0])
                omega = model.parameters[name]
                code += f'{omega.init}'.upper()

                if not re.match(f'{record_type}_{row + eta_number}_{col + eta_number}', omega.name):
                    code += f'\t; {omega.name}'

                code += '\n'

            code = f'{code.rstrip()}\n'

    record = create_record(code)
    return record


def update_ode_system(model: Model, old: Optional[CompartmentalSystem], new: CompartmentalSystem):
    """Update ODE system

    Handle changes from to CompartmentSystem
    """
    if old is None:
        old = CompartmentalSystem(CompartmentalSystemBuilder())
    model = update_lag_time(model, old, new)
    model = update_bio(model, old, new)
    model, updated_dataset = update_cmt_column(model, old, new)

    advan, trans, nonlin, haszo = new_advan_trans(model)

    if nonlin or haszo:
        model = to_des(model, new)
    else:
        if isinstance(new.dosing_compartment[0].dose, Bolus) and 'RATE' in model.datainfo.names:
            df = model.dataset.drop(columns=['RATE'])
            model = model.replace(dataset=df)

        model = pk_param_conversion(model, advan=advan, trans=trans)
        model = add_needed_pk_parameters(model, advan, trans)
        model = update_subroutines_record(model, advan, trans)
        model = update_model_record(model, advan)

        if not is_nonlinear_odes(model):
            model = from_des(model, advan)

    if not updated_dataset:
        model, updated_dataset = update_infusion(model, old)
    else:
        model, _ = update_infusion(model, old)
    return model, updated_dataset


def update_cmt_column(model, old, new):
    if model.dataset is not None:
        if (
            "admid" in model.datainfo.types
            and len(model.dataset[model.datainfo.typeix["admid"].names[0]].unique()) != 1
        ):
            cs = model.statements.ode_system
            newmap = new_compartmental_map(cs)

            d = {}
            for dose_comp in model.statements.ode_system.dosing_compartment:
                d[dose_comp.dose.admid] = newmap[dose_comp.name]

            cmt_col = get_admid(model)
            cmt_col = cmt_col.replace(d)

            dataset = model.dataset.copy()
            dataset['CMT'] = cmt_col
            di = update_datainfo(model.datainfo, dataset)
            colinfo = di['CMT'].replace(type='compartment')
            model = model.replace(datainfo=di.set_column(colinfo), dataset=dataset)

            updated_dataset = True
        elif "CMT" in model.datainfo.names and len(old.compartment_names) != len(
            new.compartment_names
        ):
            dataset = model.dataset.copy()

            # Make sure column is a number and not string
            dataset["CMT"] = pd.to_numeric(dataset["CMT"])

            # Differ in amount of compartment -> Change cmt numbering
            # The cmt number should be the same as the dosing compartment
            oldmap = model.internals.compartment_map
            assert oldmap is not None
            cs = model.statements.ode_system
            newmap = new_compartmental_map(cs)
            oldmap = oldmap.copy()
            remap = create_compartment_remap(oldmap, newmap)

            for dose_comp in old.dosing_compartment:
                if dose_comp != old.central_compartment:
                    # Remap oral doses to new dosing compartment
                    remap[oldmap[dose_comp.name]] = newmap[new.dosing_compartment[0].name]

            dataset = dataset.replace({"CMT": remap})
            model = model.replace(dataset=dataset)

            updated_dataset = True
        else:
            # Could verify that the cmt column is the same
            updated_dataset = False
    else:
        updated_dataset = False

    return model, updated_dataset


def is_nonlinear_odes(model: Model):
    """Check if ode system is nonlinear"""
    odes = model.statements.ode_system
    M = odes.compartmental_matrix
    return odes.t in M.free_symbols


def has_zero_order_inputs(model: Model):
    odes = model.statements.ode_system
    zo = odes.zero_order_inputs
    return not all(a == 0 for a in zo)


def update_infusion(model: Model, old: ODESystem):
    statements = model.statements
    new = statements.ode_system
    assert new is not None
    if isinstance(new.dosing_compartment[0].dose, Infusion) and not statements.find_assignment(
        'D1'
    ):
        # Handle direct moving of Infusion dose
        statements = statements.subs({'D2': 'D1'})

    if isinstance(new.dosing_compartment[0].dose, Infusion) and isinstance(
        old.dosing_compartment[0].dose, Bolus
    ):
        dose = new.dosing_compartment[0].dose
        if dose.rate is None:
            # FIXME: Not always D1 here!
            ass = Assignment(sympy.Symbol('D1'), dose.duration)
            cb = CompartmentalSystemBuilder(new)
            cb.set_dose(new.dosing_compartment[0], Infusion(dose.amount, duration=ass.symbol))
            statements = statements.before_odes + CompartmentalSystem(cb) + statements.after_odes
        else:
            raise NotImplementedError("First order infusion rate is not yet supported")
        statements = statements.before_odes + ass + statements.ode_system + statements.after_odes
        dataset = model.dataset.copy()
        rate = np.where(dataset['AMT'] == 0, 0.0, -2.0)
        dataset['RATE'] = rate
        updated_dataset = True
        model = model.replace(dataset=dataset)
    else:
        updated_dataset = False
    model = model.replace(statements=statements)
    return model, updated_dataset


def from_des(model, advan):
    cs = model.internals.control_stream
    old_des = cs.get_records('DES')
    newcs = cs.remove_records(old_des)

    subs = cs.get_records('SUBROUTINES')[0]
    newrec = subs.remove_option('TOL')

    trans = None

    odes = model.statements.ode_system

    output_rate = odes.get_flow(odes.central_compartment, output)

    if isinstance(output_rate, sympy.Symbol):
        trans = 'TRANS1'
    else:
        if advan in ('ADVAN1', 'ADVAN2'):
            trans = 'TRANS2'
        if advan in ('ADVAN3', 'ADVAN11'):
            rates = []
            for cmt in odes.compartment_names:
                outflows = odes.get_compartment_outflows(cmt)
                rates.extend([rate[1] for rate in outflows])

            def _is_symb_quotient(expr):
                numer, denom = expr.as_numer_denom()
                return isinstance(numer, sympy.Symbol) and isinstance(denom, sympy.Symbol)

            def _is_symb_expr_quotient(expr):
                numer, denom = expr.as_numer_denom()
                return isinstance(numer, sympy.Symbol) and isinstance(denom, sympy.Expr)

            if all(_is_symb_quotient(rate) for rate in rates):
                trans = 'TRANS4'
            elif advan == 'ADVAN3':
                if all(_is_symb_quotient(rate) or _is_symb_expr_quotient(rate) for rate in rates):
                    trans = 'TRANS3'

    if trans is not None:
        newrec = newrec.remove_option_startswith('TRANS')
        newrec = newrec.append_option(trans)

    newcs = newcs.replace_records([subs], [newrec])
    model = model.replace(internals=model.internals.replace(control_stream=newcs))
    return model


def to_des(model: Model, new: ODESystem):
    cs = model.internals.control_stream
    old_des = cs.get_records('DES')
    cs = cs.remove_records(old_des)
    subs = cs.get_records('SUBROUTINES')[0]
    newrec = subs.remove_option_startswith('TRANS')
    newrec = newrec.remove_option_startswith('ADVAN')
    newrec = newrec.remove_option('TOL')
    cs = cs.replace_records([subs], [newrec])
    subs = newrec
    step = model.estimation_steps[0]
    solver = step.solver
    if solver:
        advan = solver_to_advan(solver)
        newrec = subs.append_option(advan)
    else:
        newrec = subs.append_option('ADVAN13')
    if not subs.has_option('TOL'):
        newrec = newrec.append_option('TOL', '9')
    cs = cs.replace_records([subs], [newrec])
    des = create_record('$DES\nDUMMY=0\n')
    cs = cs.insert_record(des)
    assert isinstance(des, CodeRecord)

    to_odes = []
    for s in model.statements.before_odes:
        if model.statements.ode_system.t in s.free_symbols:
            to_odes.append(s)

    newdes = des.from_odes(new, to_odes)
    cs = cs.replace_records([des], [newdes])
    cs = cs.remove_records(model.internals.control_stream.get_records('MODEL'))
    mod = create_record('$MODEL\n')
    cs = cs.insert_record(mod)
    old_mod = mod
    assert isinstance(mod, ModelRecord)
    dosecmt_name = new.dosing_compartment[0].name
    for eq in new.eqs:
        name = eq.lhs.args[0].name[2:]
        if name == dosecmt_name:
            dose = True
        else:
            dose = False
        mod = mod.add_compartment(name, dosing=dose)
    cs = cs.replace_records([old_mod], [mod])
    model = model.replace(internals=model.internals.replace(control_stream=cs))
    return model


def update_ics(statements, odes):
    if odes is not None:
        trans = {}
        for i, amount in enumerate(odes.amounts, start=1):
            trans[sympy.Function(amount.name)(0)] = sympy.Function('A_0')(i)

        new = []
        update = False
        for s in statements:
            if s.symbol.is_Function:
                newsymb = trans[s.symbol]
                s = s.replace(symbol=newsymb)
                update = True
            new.append(s)
        if update:
            return Statements(new)

    return statements


def update_statements(model: Model, old: Statements, new: Statements, trans):
    trans['NaN'] = int(data.conf.na_rep)

    new_odes = new.ode_system
    updated_dataset = False

    old_solver, new_solver = None, None
    if len(model.estimation_steps) > 0:
        new_solver = model.estimation_steps[0].solver
    if len(model.internals.old_estimation_steps) > 0:
        old_solver = model.internals.old_estimation_steps[0].solver

    if old == new and old_solver == new_solver:
        return model, updated_dataset

    if new_odes is not None:
        old_odes = old.ode_system
        if new_odes != old_odes:
            model, updated_dataset = update_ode_system(model, old_odes, new_odes)
        else:
            if new_solver:
                if new_solver != old_solver:
                    advan = solver_to_advan(new_solver)
                    subs = model.internals.control_stream.get_records('SUBROUTINES')[0]
                    newsubs = subs.set_advan(advan)
                    newcs = model.internals.control_stream.replace_records([subs], [newsubs])
                    model = model.replace(internals=model.internals.replace(control_stream=newcs))
                    model = update_model_record(model, advan)

    main_statements = model.statements.before_odes
    main_statements = update_ics(main_statements, new_odes)
    if model.statements.ode_system is not None:
        keep = []
        for s in main_statements:
            if model.statements.ode_system.t not in s.free_symbols:
                keep.append(s)
        main_statements = Statements(tuple(keep))

    error_statements = model.statements.after_odes
    rec = model.internals.control_stream.get_pred_pk_record()
    newrec = rec.update_statements(main_statements.subs(trans), model.random_variables, trans)
    newcs = model.internals.control_stream.replace_records([rec], [newrec])
    model = model.replace(internals=model.internals.replace(control_stream=newcs))

    error = model.internals.control_stream.get_error_record()
    if not error and len(error_statements) > 0:
        empty_error = create_record('$ERROR\n')
        newcs = model.internals.control_stream.insert_record(empty_error)
        model = model.replace(internals=model.internals.replace(control_stream=newcs))
    if error:
        for i, s in enumerate(error_statements):
            if s.symbol.name == 'F':
                error_statements = error_statements[0:i] + error_statements[i + 1 :]
                break
        new_ode_system = new.ode_system
        if new_ode_system is not None:
            amounts = list(new_ode_system.amounts)
            for i, amount in enumerate(amounts, start=1):
                trans[amount] = sympy.Symbol(f"A({i})")
                trans[sympy.Function(amount.name)(new_ode_system.t)] = sympy.Symbol(f"A({i})")
        new_error = error.update_statements(
            error_statements.subs(trans), model.random_variables, trans
        )
        new_error = new_error.update_extra_nodes(model.dependent_variables)
        newcs = model.internals.control_stream.replace_records([error], [new_error])
        model = model.replace(internals=model.internals.replace(control_stream=newcs))
    return model, updated_dataset


def update_dependent_variables(model: Model, trans):
    # FIXME: This is currently not used.
    # Does it conflict with code_record.update_extra_nodes ?
    old_dvs = model.internals.old_dependent_variables
    new_dvs = model.dependent_variables
    if old_dvs != new_dvs:
        # Will replace AST nodes (representing plain assignments) for all dvs
        # with new block IF structure. All statements are kept
        # FIXME: Assumes DVs added from 1
        # FIXME: Assumes DVID colname
        # FIXME: Move to code_record? To use proper code generator
        cg = CodeGenerator()
        for i, (dv, dvid) in enumerate(new_dvs.items()):
            if i == 0:
                cg.add(f'IF (DVID.EQ.{dvid}) THEN')
            elif i == 1 and len(new_dvs) == 2:
                cg.add('ELSE')
            else:
                cg.add(f'ELSE IF (DVID.EQ.{dvid}) THEN')
            yass = model.statements.find_assignment(dv)
            cg.indent()
            cg.add(f'Y = {yass.expression.subs(trans)}')
            cg.dedent()
        cg.add('END IF')
        node = CodeRecordParser(str(cg) + '\n').root.children[0]

        cs = model.internals.control_stream
        rec = cs.get_error_pred_record()
        newinds = []
        nodes = []
        first = True
        diff = len(new_dvs) - 1
        nextnode = 0
        for ni, nj, si, sj in rec._index:
            if ni > nextnode:
                nodes += rec.root.children[nextnode:ni]
            nextnode = nj
            s = rec._statements[si]
            if isinstance(s, Assignment) and s.symbol in new_dvs:
                if first:
                    nodes += [node]
                    ind = (ni, ni + 1, si, si + len(new_dvs))
                    newinds.append(ind)
                    first = False
            else:
                nodes += rec.root.children[ni:nj]
                if first:
                    ind = (ni, nj, si, sj)
                else:
                    ind = (ni + diff, nj + diff, si, sj)
                newinds.append(ind)
        if nextnode < len(rec.root.children):
            nodes += rec.root.children[nextnode : len(rec.root.children)]

        new_tree = AttrTree(rec.root.rule, tuple(nodes))
        new_rec = CodeRecord(
            rec.name, rec.raw_name, new_tree, index=newinds, statements=rec._statements
        )
        cs = cs.replace_records([rec], [new_rec])
        internals = model.internals.replace(control_stream=cs)
        model = model.replace(internals=internals)
    return model


def update_lag_time(model: Model, old: CompartmentalSystem, new: CompartmentalSystem):
    new_dosing = new.dosing_compartment[0]
    new_lag_time = new_dosing.lag_time
    old_lag_time = old.dosing_compartment[0].lag_time
    if new_lag_time != old_lag_time and new_lag_time != 0:
        ass = Assignment(sympy.Symbol('ALAG1'), new_lag_time)
        cb = CompartmentalSystemBuilder(new)
        cb.set_lag_time(new_dosing, ass.symbol)
        model = model.replace(
            statements=model.statements.before_odes
            + ass
            + CompartmentalSystem(cb)
            + model.statements.after_odes
        )
    return model


def update_bio(model, old, new):
    """
    Update all bioavailability statements to match the numbering of the
    compartments in NONMEM.
    Is based on the order of dosing compartments
    """
    newmap = new_compartmental_map(new)
    for dose in new.dosing_compartment:
        # If the dose is not already correctly set (i.e dose numbering has
        # changed), it should be update to match the new number.
        if (
            not isinstance(dose.bioavailability, sympy.Number)
            and dose.bioavailability != f'F{newmap[dose.name]}'
        ):
            model = model.replace(
                statements=model.statements.subs(
                    {sympy.Symbol(f'{dose.bioavailability}'): sympy.Symbol(f'F{newmap[dose.name]}')}
                )
            )

    return model


def new_compartmental_map(cs: CompartmentalSystem):
    """Create compartmental map for updated model
    cs - new compartmental system
    """
    compmap = {name: i for i, name in enumerate(cs.compartment_names, start=1)}
    return compmap


def create_compartment_remap(oldmap, newmap):
    """Creates a map from old compartment number to new compartment number

    For all compartments where remapping is needed
    Assume that compartments with same name in new and old are the same compartments
    """
    remap = {}
    for name, number in oldmap.items():
        if name in newmap:
            remap[number] = newmap[name]
    return remap


def pk_param_conversion(model: Model, advan, trans):
    """Conversion map for pk parameters for removed or added compartment"""
    all_subs = model.internals.control_stream.get_records('SUBROUTINES')
    if not all_subs:
        return model
    subs = all_subs[0]
    from_advan = subs.advan
    statements = model.statements
    cs = statements.ode_system
    assert isinstance(cs, CompartmentalSystem)
    oldmap = model.internals.compartment_map
    assert oldmap is not None
    newmap = new_compartmental_map(cs)
    newmap['OUTPUT'] = len(newmap) + 1
    oldmap = oldmap.copy()
    oldmap['OUTPUT'] = len(oldmap) + 1
    remap = create_compartment_remap(oldmap, newmap)
    d = {}
    for old, new in remap.items():
        d[sympy.Symbol(f'S{old}')] = sympy.Symbol(f'S{new}')
        # FIXME: F should also be moved with dose compartment (?)
        # d[sympy.Symbol(f'F{old}')] = sympy.Symbol(f'F{new}')
        # FIXME: R, D and ALAG should be moved with dose compartment
        # d[sympy.Symbol(f'R{old}')] = sympy.Symbol(f'R{new}')
        # d[sympy.Symbol(f'D{old}')] = sympy.Symbol(f'D{new}')
        # d[sympy.Symbol(f'ALAG{old}')] = sympy.Symbol(f'ALAG{new}')
        d[sympy.Symbol(f'A({old})')] = sympy.Symbol(f'A({new})')
    if from_advan == 'ADVAN5' or from_advan == 'ADVAN7':
        reverse_map = {v: k for k, v in newmap.items()}
        for i, j in product(range(1, len(oldmap)), range(0, len(oldmap))):
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
                if cs.get_flow(from_comp, to_comp) != 0:
                    d[sympy.Symbol(f'K{i}{j}')] = sympy.Symbol(f'K{to_i}{to_j}')
                    d[sympy.Symbol(f'K{i}T{j}')] = sympy.Symbol(f'K{to_i}T{to_j}')
        if advan == 'ADVAN3':
            n = len(oldmap)
            for i in range(1, n):
                d[sympy.Symbol(f'K{i}0')] = sympy.Symbol('K')
                d[sympy.Symbol(f'K{i}T0')] = sympy.Symbol('K')
                d[sympy.Symbol(f'K{i}{n}')] = sympy.Symbol('K')
                d[sympy.Symbol(f'K{i}T{n}')] = sympy.Symbol('K')
    elif from_advan == 'ADVAN1':
        if advan == 'ADVAN3' or advan == 'ADVAN11':
            d[sympy.Symbol('V')] = sympy.Symbol('V1')
        elif advan == 'ADVAN4' or advan == 'ADVAN12':
            d[sympy.Symbol('V')] = sympy.Symbol('V2')
    elif from_advan == 'ADVAN2':
        if advan == 'ADVAN3' and trans != 'TRANS1':
            d[sympy.Symbol('V')] = sympy.Symbol('V1')
        elif advan == 'ADVAN4' and trans != 'TRANS1':
            d[sympy.Symbol('V')] = sympy.Symbol('V2')
    elif from_advan == 'ADVAN3':
        if advan == 'ADVAN1':
            if trans == 'TRANS2':
                d[sympy.Symbol('V1')] = sympy.Symbol('V')
        elif advan == 'ADVAN4':
            if trans == 'TRANS4':
                d[sympy.Symbol('V1')] = sympy.Symbol('V2')
                d[sympy.Symbol('V2')] = sympy.Symbol('V3')
            elif trans == 'TRANS6':
                d[sympy.Symbol('K21')] = sympy.Symbol('K32')
            else:  # TRANS1
                d[sympy.Symbol('K12')] = sympy.Symbol('K23')
                d[sympy.Symbol('K21')] = sympy.Symbol('K32')
        elif advan == 'ADVAN11':
            if trans == 'TRANS4':
                d.update({sympy.Symbol('Q'): sympy.Symbol('Q2')})
    elif from_advan == 'ADVAN4':
        if advan == 'ADVAN2':
            if trans == 'TRANS2':
                d[sympy.Symbol('V2')] = sympy.Symbol('V')
        if advan == 'ADVAN3':
            if trans == 'TRANS4':
                d.update(
                    {sympy.Symbol('V2'): sympy.Symbol('V1'), sympy.Symbol('V3'): sympy.Symbol('V2')}
                )
            elif trans == 'TRANS6':
                d.update({sympy.Symbol('K32'): sympy.Symbol('K21')})
            else:  # TRANS1
                d.update(
                    {
                        sympy.Symbol('K23'): sympy.Symbol('K12'),
                        sympy.Symbol('K32'): sympy.Symbol('K21'),
                    }
                )
        elif advan == 'ADVAN12':
            if trans == 'TRANS4':
                d.update({sympy.Symbol('Q'): sympy.Symbol('Q3')})
    elif from_advan == 'ADVAN11':
        if advan == 'ADVAN1':
            if trans == 'TRANS2':
                d[sympy.Symbol('V1')] = sympy.Symbol('V')
        elif advan == 'ADVAN3':
            if trans == 'TRANS4':
                d[sympy.Symbol('Q2')] = sympy.Symbol('Q')
        elif advan == 'ADVAN12':
            if trans == 'TRANS4':
                d.update(
                    {
                        sympy.Symbol('V1'): sympy.Symbol('V2'),
                        sympy.Symbol('Q2'): sympy.Symbol('Q3'),
                        sympy.Symbol('V2'): sympy.Symbol('V3'),
                        sympy.Symbol('Q3'): sympy.Symbol('Q4'),
                        sympy.Symbol('V3'): sympy.Symbol('V4'),
                    }
                )
            elif trans == 'TRANS6':
                d.update(
                    {
                        sympy.Symbol('K31'): sympy.Symbol('K42'),
                        sympy.Symbol('K21'): sympy.Symbol('K32'),
                    }
                )
            else:  # TRANS1
                d.update(
                    {
                        sympy.Symbol('K12'): sympy.Symbol('K23'),
                        sympy.Symbol('K21'): sympy.Symbol('K32'),
                        sympy.Symbol('K13'): sympy.Symbol('K24'),
                        sympy.Symbol('K31'): sympy.Symbol('K42'),
                    }
                )
    elif from_advan == 'ADVAN12':
        if advan == 'ADVAN2':
            if trans == 'TRANS2':
                d[sympy.Symbol('V2')] = sympy.Symbol('V')
        elif advan == 'ADVAN4':
            if trans == 'TRANS4':
                d[sympy.Symbol('Q3')] = sympy.Symbol('Q')
        elif advan == 'ADVAN11':
            if trans == 'TRANS4':
                d.update(
                    {
                        sympy.Symbol('V2'): sympy.Symbol('V1'),
                        sympy.Symbol('Q3'): sympy.Symbol('Q2'),
                        sympy.Symbol('V3'): sympy.Symbol('V2'),
                        sympy.Symbol('Q4'): sympy.Symbol('Q3'),
                        sympy.Symbol('V4'): sympy.Symbol('V3'),
                    }
                )
            elif trans == 'TRANS6':
                d.update(
                    {
                        sympy.Symbol('K42'): sympy.Symbol('K31'),
                        sympy.Symbol('K32'): sympy.Symbol('K21'),
                    }
                )
            else:  # TRANS1
                d.update(
                    {
                        sympy.Symbol('K23'): sympy.Symbol('K12'),
                        sympy.Symbol('K32'): sympy.Symbol('K21'),
                        sympy.Symbol('K24'): sympy.Symbol('K13'),
                        sympy.Symbol('K42'): sympy.Symbol('K31'),
                    }
                )
    if advan == 'ADVAN5' or advan == 'ADVAN7':
        n = newmap['CENTRAL']
        if from_advan not in ('ADVAN5', 'ADVAN7'):
            d[sympy.Symbol('K')] = sympy.Symbol(f'K{n}0')
        else:
            n_old = oldmap['CENTRAL']
            d[sympy.Symbol(f'K{n_old}0')] = sympy.Symbol(f'K{n}0')
    model = model.replace(statements=statements.subs(d))
    return model


def match_advan1(odes):
    return len(odes) == 1


def match_advan2(statements):
    odes = statements.ode_system
    if len(odes) != 2:
        return False
    dosing = odes.dosing_compartment[0]
    outflows = odes.get_compartment_outflows(dosing)
    if len(outflows) != 1:
        return False
    central = outflows[0][0]
    central_rate = outflows[0][1]

    # Check if rate is depending on CL or V assignments
    dep_assigns = set()
    expr = central_rate
    for s in reversed(statements.before_odes):
        if s.symbol in expr.free_symbols:
            dep_assigns.add(s.symbol)
            expr = expr.subs(s.symbol, s.expression)

    if {sympy.Symbol('CL'), sympy.Symbol('V')} & dep_assigns:
        # Cannot use reserved symbols
        return False
    central_outflows = odes.get_compartment_outflows(central)
    if len(central_outflows) != 1:
        return False
    return True


def match_advan3(odes):
    if len(odes) != 2:
        return False
    central = odes.dosing_compartment[0]
    bidir = odes.get_bidirectionals(central)
    if len(bidir) != 1:
        return False
    if odes.get_flow(bidir[0], output) != 0:
        return False
    return True


def match_advan4(odes):
    if len(odes) != 3:
        return False
    dosing = odes.dosing_compartment[0]
    outflows = odes.get_compartment_outflows(dosing)
    if len(outflows) != 1:
        return False
    central = outflows[0][0]
    bidir = odes.get_bidirectionals(central)
    if len(bidir) != 1:
        return False
    if odes.get_flow(bidir[0], output) != 0 or odes.get_flow(bidir[0], dosing) != 0:
        return False
    return True


def match_advan11(odes):
    if len(odes) != 3:
        return False
    central = odes.dosing_compartment[0]
    bidir = odes.get_bidirectionals(central)
    if len(bidir) != 2:
        return False
    if (
        odes.get_flow(bidir[0], output) != 0
        or odes.get_flow(bidir[1], output) != 0
        or odes.get_flow(bidir[0], bidir[1]) != 0
    ):
        return False
    return True


def match_advan12(odes):
    if len(odes) != 4:
        return False
    dosing = odes.dosing_compartment[0]
    outflows = odes.get_compartment_outflows(dosing)
    if len(outflows) != 1:
        return False
    central = outflows[0][0]
    bidir = odes.get_bidirectionals(central)
    if len(bidir) != 2:
        return False
    if (
        odes.get_flow(bidir[0], output) != 0
        or odes.get_flow(bidir[1], output) != 0
        or odes.get_flow(bidir[0], bidir[1]) != 0
    ):
        return False
    return True


def new_advan_trans(model: Model):
    """Decide which new advan and trans to be used"""
    all_subs = model.internals.control_stream.get_records('SUBROUTINES')
    if all_subs:
        subs = all_subs[0]
        oldtrans = subs.get_option_startswith('TRANS')
    else:
        oldtrans = None
    odes = model.statements.ode_system
    nonlin = is_nonlinear_odes(model)
    has_zo = has_zero_order_inputs(model)
    if nonlin or has_zo:
        advan = 'ADVAN13'
    elif match_advan1(odes):
        advan = 'ADVAN1'
    elif match_advan2(model.statements):
        advan = 'ADVAN2'
    elif match_advan3(odes):
        advan = 'ADVAN3'
    elif match_advan4(odes):
        advan = 'ADVAN4'
    elif match_advan11(odes):
        advan = 'ADVAN11'
    elif match_advan12(odes):
        advan = 'ADVAN12'
    else:  # General linear
        # We could use ADVAN7 if has_linear_odes_with_real_eigenvalues
        # but ADVAN7 seems to be unstable in NONMEM
        advan = 'ADVAN5'

    if nonlin:
        trans = None
    elif oldtrans == 'TRANS1':
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
    elif oldtrans is None:
        central = odes.central_compartment
        elimination_rate = odes.get_flow(central, output)
        num, den = elimination_rate.as_numer_denom()
        if num.is_Symbol and den.is_Symbol:
            if advan in ['ADVAN1', 'ADVAN2']:
                trans = 'TRANS2'
            else:
                trans = 'TRANS4'
        else:
            trans = 'TRANS1'
    else:
        trans = 'TRANS1'

    return advan, trans, nonlin, has_zo


def update_subroutines_record(model: Model, advan, trans):
    """Update $SUBROUTINES with new advan and trans"""
    all_subs = model.internals.control_stream.get_records('SUBROUTINES')
    if not all_subs:
        content = f'$SUBROUTINES {advan} {trans}\n'
        subsrec = create_record(content)
        newcs = model.internals.control_stream.insert_record(subsrec)
        model = model.replace(internals=model.internals.replace(control_stream=newcs))
        return model
    subs = all_subs[0]
    oldadvan = subs.advan
    oldtrans = subs.trans

    if advan != oldadvan:
        newsubs = subs.replace_option(oldadvan, advan)
    else:
        newsubs = subs
    if trans != oldtrans:
        if trans is None:
            newsubs = newsubs.remove_option_startswith('TRANS')
        else:
            newsubs = newsubs.replace_option(oldtrans, trans)
    newcs = model.internals.control_stream.replace_records([subs], [newsubs])
    model = model.replace(internals=model.internals.replace(control_stream=newcs))
    return model


def update_model_record(model: Model, advan):
    """Update $MODEL"""
    odes = model.statements.ode_system
    if not isinstance(odes, CompartmentalSystem):
        return model

    oldmap = model.internals.compartment_map
    if oldmap is None:
        return model
    newmap = new_compartmental_map(odes)

    replace_dict = {'compartment_map': newmap}

    if advan in ['ADVAN1', 'ADVAN2', 'ADVAN3', 'ADVAN4', 'ADVAN10', 'ADVAN11', 'ADVAN12']:
        newcs = model.internals.control_stream.remove_records(
            model.internals.control_stream.get_records('MODEL')
        )
        replace_dict['control_stream'] = newcs
    else:
        if oldmap != newmap or model.estimation_steps[0].solver:
            newcs = model.internals.control_stream.remove_records(
                model.internals.control_stream.get_records('MODEL')
            )
            mod = create_record('$MODEL\n')
            newcs = newcs.insert_record(mod)
            old_mod = mod
            assert isinstance(mod, ModelRecord)
            comps = {v: k for k, v in newmap.items()}
            i = 1
            while True:
                if i not in comps:
                    break
                if i == 1:
                    mod = mod.add_compartment(comps[i], dosing=True)
                else:
                    mod = mod.add_compartment(comps[i], dosing=False)
                i += 1
            newcs = newcs.replace_records([old_mod], [mod])
            replace_dict['control_stream'] = newcs
    model = model.replace(internals=model.internals.replace(**replace_dict))
    return model


def add_needed_pk_parameters(model: Model, advan, trans):
    """Add missing pk parameters that NONMEM needs"""
    statements = model.statements
    odes = statements.ode_system
    assert isinstance(odes, CompartmentalSystem)
    if advan == 'ADVAN2' or advan == 'ADVAN4' or advan == 'ADVAN12':
        if not statements.find_assignment('KA'):
            comp, rate = odes.get_compartment_outflows(odes.find_depot(statements))[0]
            ass = Assignment(sympy.Symbol('KA'), rate)
            if rate != ass.symbol:
                cb = CompartmentalSystemBuilder(odes)
                cb.add_flow(odes.find_depot(statements), comp, ass.symbol)
                model = model.replace(
                    statements=statements.before_odes
                    + ass
                    + CompartmentalSystem(cb)
                    + statements.after_odes
                )
                statements = model.statements
                odes = statements.ode_system
    if advan in ['ADVAN1', 'ADVAN2'] and trans == 'TRANS2':
        central = odes.central_compartment
        model = add_parameters_ratio(model, 'CL', 'V', central, output)
    elif advan == 'ADVAN3' and trans == 'TRANS4':
        central = odes.central_compartment
        peripheral = odes.peripheral_compartments[0]
        model = add_parameters_ratio(model, 'CL', 'V1', central, output)
        model = add_parameters_ratio(model, 'Q', 'V2', peripheral, central)
        model = add_parameters_ratio(model, 'Q', 'V1', central, peripheral)
    elif advan == 'ADVAN4':
        central = odes.central_compartment
        peripheral = odes.peripheral_compartments[0]
        if trans == 'TRANS1':
            rate1 = odes.get_flow(central, peripheral)
            model = add_rate_assignment_if_missing(model, 'K23', rate1, central, peripheral)
            rate2 = odes.get_flow(peripheral, central)
            model = add_rate_assignment_if_missing(model, 'K32', rate2, peripheral, central)
        if trans == 'TRANS4':
            model = add_parameters_ratio(model, 'CL', 'V2', central, output)
            model = add_parameters_ratio(model, 'Q', 'V3', peripheral, central)
    elif advan == 'ADVAN12' and trans == 'TRANS4':
        central = odes.central_compartment
        peripheral1 = odes.peripheral_compartments[0]
        peripheral2 = odes.peripheral_compartments[1]
        model = add_parameters_ratio(model, 'CL', 'V2', central, output)
        model = add_parameters_ratio(model, 'Q3', 'V3', peripheral1, central)
        model = add_parameters_ratio(model, 'Q4', 'V4', peripheral2, central)
    elif advan == 'ADVAN11' and trans == 'TRANS4':
        central = odes.central_compartment
        peripheral1 = odes.peripheral_compartments[0]
        peripheral2 = odes.peripheral_compartments[1]
        model = add_parameters_ratio(model, 'CL', 'V1', central, output)
        model = add_parameters_ratio(model, 'Q2', 'V2', peripheral1, central)
        model = add_parameters_ratio(model, 'Q3', 'V3', peripheral2, central)
    elif advan == 'ADVAN5' or advan == 'ADVAN7':
        oldmap = model.internals.compartment_map
        assert oldmap is not None
        newmap = new_compartmental_map(odes)
        newmap['OUTPUT'] = len(newmap) + 1
        for source in newmap.keys():
            if source == 'OUTPUT':
                continue
            for dest in newmap.keys():
                if source != dest:  # NOTE Skip same
                    source_comp = odes.find_compartment(source)
                    if dest == 'OUTPUT':
                        dest_comp = output
                    else:
                        dest_comp = odes.find_compartment(dest)
                    rate = odes.get_flow(source_comp, dest_comp)
                    if rate != 0:
                        assert isinstance(source_comp, Compartment)
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
                        model = add_rate_assignment_if_missing(
                            model, param, rate, source_comp, dest_comp, synonyms=names
                        )
    return model


def add_parameters_ratio(model: Model, numpar, denompar, source, dest):
    statements = model.statements
    if not statements.find_assignment(numpar) or not statements.find_assignment(denompar):
        odes = statements.ode_system
        assert isinstance(odes, CompartmentalSystem)
        rate = odes.get_flow(source, dest)
        numer, denom = rate.as_numer_denom()
        par1 = Assignment(sympy.Symbol(numpar), numer)
        par2 = Assignment(sympy.Symbol(denompar), denom)
        new_statement1 = Statements()
        new_statement2 = Statements()
        if rate != par1.symbol / par2.symbol:
            if not statements.find_assignment(numpar):
                odes = odes.subs({numer: sympy.Symbol(numpar)})
                new_statement1 = par1
            if not statements.find_assignment(denompar):
                odes = odes.subs({denom: sympy.Symbol(denompar)})
                new_statement2 = par2
        cb = CompartmentalSystemBuilder(odes)
        cb.add_flow(source, dest, par1.symbol / par2.symbol)
        model = model.replace(
            statements=(
                statements.before_odes
                + new_statement1
                + new_statement2
                + CompartmentalSystem(cb)
                + statements.after_odes
            )
        )
    return model


def define_parameter(
    model: Model, name: str, value: sympy.Expr, synonyms: Optional[List[str]] = None
):
    """Define a parameter in statments if not defined
    Update if already defined as other value
    return True if new assignment was added
    """
    if synonyms is None:
        synonyms = [name]
    for syn in synonyms:
        i = model.statements.find_assignment_index(syn)
        if i is not None:
            ass = model.statements[i]
            assert isinstance(ass, Assignment)
            if value != ass.expression and value != sympy.Symbol(name):
                replacement_ass = Assignment(ass.symbol, value)
                model = model.replace(
                    statements=model.statements[:i] + replacement_ass + model.statements[i + 1 :]
                )
            return model, False
    new_ass = Assignment(sympy.Symbol(name), value)
    # Put new rate before output rate in statements
    central = model.statements.ode_system.central_compartment
    output_rate = model.statements.ode_system.get_flow(central, output)
    if isinstance(output_rate, sympy.Symbol):
        out_ind = model.statements.find_assignment_index(output_rate)
    else:
        out_ind = None
    if out_ind:
        before_odes = (
            model.statements.before_odes[:out_ind]
            + new_ass
            + model.statements.before_odes[out_ind:]
        )
    else:
        before_odes = model.statements.before_odes + new_ass

    model = model.replace(
        statements=before_odes + model.statements.ode_system + model.statements.after_odes
    )
    return model, True


def add_rate_assignment_if_missing(
    model: Model,
    name: str,
    value: sympy.Expr,
    source: Compartment,
    dest: Compartment,
    synonyms: Optional[List[str]] = None,
):
    model, added = define_parameter(model, name, value, synonyms=synonyms)
    if added:
        cb = CompartmentalSystemBuilder(model.statements.ode_system)
        cb.add_flow(source, dest, sympy.Symbol(name))
        model = model.replace(
            statements=model.statements.before_odes
            + CompartmentalSystem(cb)
            + model.statements.after_odes
        )
    return model


def update_abbr_record(model: Model, rv_trans):
    trans = {}

    # Remove not used ABBR, keep ABBR that are not REPLACE
    keep = []
    recs = model.internals.control_stream.get_records('ABBREVIATED')
    for rec in recs:
        rec_map = rec.translate_to_pharmpy_names()
        for nmname, ppname in rec_map.items():
            if not (ppname in rec_map and rec_map[ppname] == nmname):
                break
        else:
            keep.append(rec)

    control_stream = model.internals.control_stream.replace_all('ABBREVIATED', keep)

    if rv_trans:
        for pharmpy_name, nonmem_name in rv_trans.items():
            abbr_record_code = f'$ABBR REPLACE {pharmpy_name}={nonmem_name}\n'
            abbr_record = create_record(abbr_record_code)
            trans[nonmem_name] = pharmpy_name
            control_stream = control_stream.insert_record(abbr_record)

    model = model.replace(internals=model.internals.replace(control_stream=control_stream))
    return model, trans


def update_estimation(control_stream, model):
    old = model.internals.old_estimation_steps
    new = model.estimation_steps
    if old == new:
        return control_stream

    delta = code_record.diff(old, new)
    old_records = control_stream.get_records('ESTIMATION')
    i = 0
    new_records = []

    prev = (None, None)
    for op, est in delta:
        if op == 1:
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
                op_prev, est_prev = prev
                if not (
                    est.method.startswith('FO')
                    and op_prev == -1
                    and est.evaluation
                    and est_prev is not None
                    and not est_prev.evaluation
                    and est_prev.maximum_evaluations == est.maximum_evaluations
                ):
                    if set(protected_attributes).intersection({'MAXEVALS', 'MAXEVAL'}):
                        raise ValueError('MAXEVAL already set by evaluation=True')
                    est_code += f' MAXEVAL={est.maximum_evaluations}'
                protected_attributes += ['MAXEVALS', 'MAXEVAL']
            if est.isample is not None:
                est_code += f' ISAMPLE={est.isample}'
            if est.niter is not None:
                est_code += f' NITER={est.niter}'
            if est.auto is not None:
                est_code += f' AUTO={int(est.auto)}'
            if est.keep_every_nth_iter is not None:
                est_code += f' PRINT={est.keep_every_nth_iter}'
            if est.tool_options:
                option_names = set(est.tool_options.keys())
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
        elif op == -1:
            i += 1
        else:
            new_records.append(old_records[i])
            i += 1
        prev = (op, est)

    if old_records:
        control_stream = control_stream.replace_records(old_records, new_records)
    else:
        for rec in new_records:
            newrec = create_record(str(rec))
            control_stream = control_stream.insert_record(newrec)

    # Initiate old_cov
    old_cov = None
    for est in old:
        old_cov = est.cov

    # Initiate new_cov
    new_cov = None
    for est in new:
        new_cov = est.cov

    if old_cov is None and new_cov is not None:
        # Add $COV
        last_est_rec = control_stream.get_records('ESTIMATION')[-1]
        idx_cov = control_stream.records.index(last_est_rec)
        if new_cov == 'SANDWICH':
            covrec_ = '$COVARIANCE'
        elif new_cov == 'CPG':
            covrec_ = '$COVARIANCE MATRIX=S'
        elif new_cov == 'OFIM':
            covrec_ = '$COVARIANCE MATRIX=R'
        covrec = create_record(f'{covrec_}\n')
        control_stream = control_stream.insert_record(covrec, at_index=idx_cov + 1)
    elif old_cov is not None and new_cov is None:
        # Remove $COV
        covrecs = control_stream.get_records('COVARIANCE')
        control_stream = control_stream.remove_records(covrecs)

    # Update $TABLE
    # Currently only adds if did not exist before
    cols = set()
    for estep in new:
        cols.update(estep.predictions)
        cols.update(estep.residuals)
    tables = control_stream.get_records('TABLE')
    if model.dataset is not None and not tables and cols:
        s = f'$TABLE {model.datainfo.id_column.name} {model.datainfo.idv_column.name} '
        s += f'{model.datainfo.dv_column.name} '
        s += f'{" ".join(cols)} FILE=mytab NOAPPEND NOPRINT'
        if any(id_val > 99999 for id_val in get_ids(model)):
            s += ' FORMAT=s1PE16.8'
        tabrec = create_record(s)
        control_stream = control_stream.insert_record(tabrec)
    return control_stream


def solver_to_advan(solver):
    if solver == 'LSODA':
        return 'ADVAN13'
    elif solver == 'CVODES':
        return 'ADVAN14'
    elif solver == 'DGEAR':
        return 'ADVAN8'
    elif solver == 'DVERK':
        return 'ADVAN6'
    elif solver == 'IDA':
        return 'ADVAN15'
    elif solver == 'LSODI':
        return 'ADVAN9'

    raise ValueError(solver)


def update_ccontra(model: Model, path=None, force=False):
    # FIXME: handle multiple DVs
    if len(model.observation_transformation) > 1:
        raise ValueError('Cannot create ccontra for multiple DVs')

    obs_trans = model.observation_transformation
    h = model.observation_transformation[next(iter(obs_trans))]

    # FIXME: handle other DVs?
    y = list(model.dependent_variables.keys())[0]
    dhdy = sympy.diff(h, y)
    ll = -2 * sympy.log(dhdy)
    ll = ll.subs(y, sympy.Symbol('y', real=True, positive=True))
    ll = simplify_expression(model, ll)
    ll = ll.subs(sympy.Symbol('y', real=True, positive=True), y)

    tr = create_name_map(model)
    tr = {sympy.Symbol(key): sympy.Symbol(value) for key, value in tr.items()}
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
        e1 = sympy_printing.fortran.fcode(h.subs(y, sympy.Symbol('y(1)')), assign_to='y(1)')
        fh.write(e1)
        fh.write(ccontr2)
        e2 = sympy_printing.fortran.fcode(
            sympy.Symbol('c1') + ll.subs(y, sympy.Symbol('y(1)')), assign_to='c1'
        )
        fh.write(e2)
        fh.write(ccontr3)


def update_name_of_tables(control_stream: NMTranControlStream, new_name: str):
    m = re.search(r'.*?(\d+)$', new_name)
    if m:
        n = int(m.group(1))
        for table in control_stream.get_records('TABLE'):
            table_path = table.path
            table_name = table_path.stem
            m = re.search(r'(.*?)(\d+)$', table_name)
            if m:
                table_stem = m.group(1)
                new_table_name = f'{table_stem}{n}'
                new_table = table.set_path(table_path.parent / new_table_name)
                control_stream = control_stream.replace_records([table], [new_table])
    return control_stream


def update_sizes(control_stream, model: Model):
    """Update $SIZES if needed"""
    all_sizes = control_stream.get_records('SIZES')
    sizes = all_sizes[0] if all_sizes else create_record('$SIZES ')
    assert isinstance(sizes, SizesRecord)
    odes = model.statements.ode_system

    if odes is not None and isinstance(odes, CompartmentalSystem):
        n_compartments = len(odes)
        sizes = sizes.set_PC(n_compartments)
    thetas = [p for p in model.parameters if p.symbol not in model.random_variables.free_symbols]
    sizes = sizes.set_LTH(len(thetas))

    if len(str(sizes)) > 7:
        if len(all_sizes) == 0:
            sizesrec = create_record(str(sizes))
            control_stream = control_stream.insert_record(sizesrec)
        else:
            control_stream = control_stream.replace_records([all_sizes[0]], [sizes])
    return control_stream


def update_input(control_stream, model: Model):
    """Update $INPUT"""
    input_records = control_stream.get_records("INPUT")
    _, drop, _, colnames = parse_column_info(control_stream)
    keep = []
    i = 0
    for child in input_records[0].root.children:
        if child.rule != 'option':
            keep.append(child)
            continue

        if (colnames[i] is not None and (colnames[i] != model.datainfo[i].name)) or (
            not drop[i] and (model.datainfo[i].drop or model.datainfo[i].datatype == 'nmtran-date')
        ):
            dropped = model.datainfo[i].drop or model.datainfo[i].datatype == 'nmtran-date'
            anonymous = colnames[i] is None
            key = 'DROP' if anonymous and dropped else model.datainfo[i].name
            value = 'DROP' if not anonymous and dropped else None
            new = input_records[0]._create_option(key, value)
            keep.append(new)
        else:
            keep.append(child)

        i += 1

        if i >= len(model.datainfo):
            last_child = input_records[0].root.children[-1]
            if last_child.rule == 'NEWLINE':
                keep.append(last_child)
            break

    newroot = AttrTree(input_records[0].root.rule, tuple(keep))
    new_input = input_records[0].replace(root=newroot)

    for ci in model.datainfo[len(colnames) :]:
        new_input = new_input.append_option(ci.name, 'DROP' if ci.drop else None)
    control_stream = control_stream.replace_records([input_records[0]], [new_input])
    return control_stream


def get_zero_fix_rvs(model, eta=True):
    zero_fix = []

    if eta:
        dists = model.random_variables.etas
    else:
        dists = model.random_variables.epsilons

        for dist in dists:
            for parname in dist.parameter_names:
                param = model.parameters[parname]
                if not (param.init == 0.0 and param.fix):
                    break
            else:
                zero_fix.extend(dist.names)

    return zero_fix


def update_initial_individual_estimates(model: Model, path, nofiles=False):
    """Update $ETAS

    Could have 0 FIX in model. Need to read these
    """
    if path == 'DUMMYPATH':
        phi_path = path
    else:
        if path is None:  # What to do here?
            phi_path = Path('.')
        else:
            phi_path = path.parent
        phi_path /= f'{model.name}_input.phi'

    estimates = model.initial_individual_estimates
    assert estimates is not None
    rv_names = {rv for rv in model.random_variables.names if rv.startswith('ETA')}
    columns = set(estimates.columns)
    if columns < rv_names:
        raise ValueError(
            f'Cannot set initial estimate for random variable not in the model:'
            f' {rv_names - columns}'
        )
    diff = columns - rv_names
    # If not setting all etas automatically set remaining to 0 for all individuals
    if len(diff) > 0:
        for name in diff:
            estimates = estimates.copy(deep=True)
            estimates[name] = 0
        estimates = _sort_eta_columns(estimates)

    etas = estimates
    zero_fix = get_zero_fix_rvs(model, eta=True)
    if zero_fix:
        for eta in zero_fix:
            etas[eta] = 0
    etas = _sort_eta_columns(etas)
    if not nofiles:
        phi = PhiTable(df=etas)
        table_file = NONMEMTableFile(tables=[phi])
        table_file.write(phi_path)
    control_stream = model.internals.control_stream
    eta_records = control_stream.get_records('ETAS')
    if eta_records:
        record = eta_records[0]
    else:
        record = create_record('$ETAS ')
        control_stream = control_stream.insert_record(record)
    assert isinstance(record, EtasRecord)
    newrecord = record.set_path(phi_path)
    control_stream = control_stream.replace_records([record], [newrecord])

    first_est_record = control_stream.get_records('ESTIMATION')[0]
    try:
        first_est_record.option_pairs['MCETA']
    except KeyError:
        newrec = first_est_record.set_option('MCETA', '1')
        control_stream = control_stream.replace_records([first_est_record], [newrec])
    return control_stream


def _sort_eta_columns(df: pd.DataFrame):
    return df.reindex(sorted(df.columns), axis=1)


def abbr_translation(model: Model, rv_trans):
    abbr_pharmpy = model.internals.control_stream.abbreviated.translate_to_pharmpy_names()
    abbr_replace = model.internals.control_stream.abbreviated.replace
    model, abbr_trans = update_abbr_record(model, rv_trans)
    abbr_recs = {
        sympy.Symbol(abbr_pharmpy[value]): sympy.Symbol(key)
        for key, value in abbr_replace.items()
        if value in abbr_pharmpy.keys()
    }
    abbr_trans.update(abbr_recs)
    return model, abbr_trans


def create_name_map(model):
    trans = {}
    thetas = [p for p in model._parameters if p.symbol not in model.random_variables.free_symbols]
    for i, theta in enumerate(thetas):
        trans[theta.name] = f'THETA({i + 1})'

    def add_rv_params(rvs, param_name):
        cov = rvs.covariance_matrix
        for row in range(0, cov.rows):
            for col in range(0, row + 1):
                if cov[row, col] != 0:
                    nonmem_name = f'{param_name}({row + 1},{col + 1})'
                    name = cov[row, col].name
                    if name not in trans:
                        # Do not add more than once to handle IOV SAME
                        trans[name] = nonmem_name

        i = 1
        for dist in rvs:
            for name in dist.names:
                prefix = 'ETA' if param_name == 'OMEGA' else 'EPS'
                nonmem_name = f'{prefix}({i})'
                trans[name] = nonmem_name
                if param_name == 'EPS':
                    nonmem_name = f'ERR({i})'
                    trans[name] = nonmem_name
                i += 1

    add_rv_params(model.random_variables.etas, 'OMEGA')
    add_rv_params(model.random_variables.epsilons, 'SIGMA')

    return trans
