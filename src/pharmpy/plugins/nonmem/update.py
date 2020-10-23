import re

import numpy as np

from pharmpy import data
from pharmpy.random_variables import VariabilityLevel
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
    new_names = {rv.name for rv in new}
    old_names = {rv.name for rv in old}
    removed = old_names - new_names
    if removed:
        remove_records = []
        next_eta = 1
        for omega_record in model.control_stream.get_records(
            'OMEGA'
        ) + model.control_stream.get_records('SIGMA'):
            current_names = omega_record.eta_map.keys()
            if removed >= current_names:
                remove_records.append(omega_record)
            elif not removed.isdisjoint(current_names):
                # one or more in the record
                omega_record.remove(removed & current_names)
                omega_record.renumber(next_eta)
                # FIXME: No handling of OMEGA(1,1) etc in code
                next_eta += len(omega_record)
            else:
                # keep all
                omega_record.renumber(next_eta)
                next_eta += len(omega_record)
        model.control_stream.remove_records(remove_records)

    new_maps = []
    for rv in new:
        if rv.name not in old_names:
            omega_name = (rv.pspace.distribution.std ** 2).name
            if omega_name not in old.all_parameters():
                rv_name = rv.name.upper()
                omega = model.parameters[omega_name]

                if rv.variability_level == VariabilityLevel.RUV:
                    record_name = 'SIGMA'
                else:
                    record_name = 'OMEGA'

                record, eta_number = create_omega_single(model, omega, record_name)
                record.add_omega_name_comment(omega_name)

                new_maps.append(
                    (record, {omega_name: (eta_number, eta_number)}, {rv_name: eta_number})
                )

    rvs_new, dist_new = new.distributions_as_list()
    rvs_old, dist_old = old.distributions_as_list()

    for entry, combined_dist in zip(rvs_new, dist_new):  # TODO: better names
        if entry not in rvs_old and entry[0].name in old_names:
            rvs = [rv.name for rv in entry]
            dist = combined_dist

            records = get_omega_records(model, rvs)
            model.control_stream.remove_records(records)
            omega_new = create_omega_block(model, dist)

            next_omega = 1
            previous_size = None
            prev_cov = None

            for omega_record in model.control_stream.get_records('OMEGA'):
                next_omega_cur = next_omega

                omegas, next_omega, previous_size = omega_record.parameters(
                    next_omega_cur, previous_size
                )
                etas, next_eta, prev_cov, _ = omega_record.random_variables(
                    next_omega_cur, prev_cov
                )
                if omega_record == omega_new:
                    m_1 = dist.args[1]
                    m_2 = etas[0].pspace.distribution.args[1]
                    for row in range(m_1.shape[0]):
                        for col in range(m_1.shape[1]):
                            if row > col:

                                elem_1 = m_1.row(row).col(col)
                                name_1 = str(elem_1[0])

                                elem_2 = m_2.row(row).col(col)
                                name_2 = str(elem_2[0])

                                omega_record.name_map[name_1] = omega_record.name_map.pop(name_2)

    # FIXME: Setting the maps needs to be done here and not in loop. Automatic renumbering is
    #        probably the culprit. There should be a difference between added parameters and
    #        original parameters when it comes to which naming scheme to use
    if new_maps:
        for record, name_map, eta_map in new_maps:
            record.name_map = name_map
            record.eta_map = eta_map


def get_omega_records(model, params):
    records = []
    next_omega = 1
    prev_cov = None

    for omega_record in model.control_stream.get_records('OMEGA'):
        etas, next_omega, prev_cov, _ = omega_record.random_variables(next_omega, prev_cov)
        for eta in etas:
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


def create_omega_single(model, param, record='OMEGA'):
    eta_number, previous_size = get_next_eta(model, record)

    param_str = f'${record}  {param.init}\n'

    record = model.control_stream.insert_record(param_str)

    record.parameters(eta_number, previous_size)
    record.random_variables(eta_number)

    return record, eta_number


def create_omega_block(model, dist):
    m = dist.args[1]
    param_str = f'$OMEGA BLOCK({m.shape[0]})\n'

    for row in range(m.shape[0]):
        for col in range(m.shape[1]):
            if row >= col:
                elem = m.row(row).col(col)
                name = str(elem[0])
                omega = model.parameters[name]
                param_str += f'{omega.init}\t'

        param_str += '\n'

    record = model.control_stream.insert_record(param_str)

    return record


def update_ode_system(model, old, new):
    """Update ODE system

    Handle changes from CompartmentSystem to ExplicitODESystem
    """
    if type(old) == CompartmentalSystem and type(new) == ExplicitODESystem:
        subs = model.control_stream.get_records('SUBROUTINES')[0]
        subs.remove_option_startswith('TRANS')
        subs.remove_option_startswith('ADVAN')
        subs.append_option('ADVAN6')
        subs.append_option('TOL', 3)
        des = model.control_stream.insert_record('$DES\nDUMMY=0')
        des.from_odes(new)
        mod = model.control_stream.insert_record('$MODEL\n')
        for eq, ic in zip(new.odes[:-1], list(new.ics.keys())[:-1]):
            name = eq.lhs.args[0].name[2:]
            if new.ics[ic] != 0:
                dose = True
            else:
                dose = False
            mod.add_compartment(name, dosing=dose)
    elif type(old) == CompartmentalSystem and type(new) == CompartmentalSystem:
        update_lag_time(model, old, new)
        if isinstance(new.find_dosing().dose, Bolus) and 'RATE' in model.dataset.columns:
            df = model.dataset
            df.drop(columns=['RATE'], inplace=True)
            model.dataset = df
        subs = model.control_stream.get_records('SUBROUTINES')[0]
        advan = subs.get_option_startswith('ADVAN')
        trans = subs.get_option_startswith('TRANS')
        if advan == 'ADVAN5' or advan == 'ADVAN7':
            remove_compartments(model, old, new)
            add_compartments(model, old, new)
        if not old.find_depot() and new.find_depot():
            # Depot was added
            statements = model.statements
            comp, rate = new.get_compartment_outflows(new.find_depot())[0]
            ass = Assignment('KA', rate)
            statements.add_before_odes(ass)
            new.add_flow(new.find_depot(), comp, ass.symbol)
            if advan == 'ADVAN1':
                subs.replace_option('ADVAN1', 'ADVAN2')
                secondary = secondary_pk_param_conversion_map(len(old), 1, removed=False)
                statements.subs(secondary)
            elif advan == 'ADVAN3':
                subs.replace_option('ADVAN3', 'ADVAN4')
                secondary = secondary_pk_param_conversion_map(len(old), 1, removed=False)
                statements.subs(secondary)
                if trans == 'TRANS1':
                    statements.subs({symbol('K12'): symbol('K23'), symbol('K21'): symbol('K32')})
                elif trans == 'TRANS4':
                    statements.subs({symbol('V1'): symbol('V2'), symbol('V2'): symbol('V3')})
                elif trans == 'TRANS6':
                    statements.subs({symbol('K21'): symbol('K32')})
            elif advan == 'ADVAN11':
                subs.replace_option('ADVAN11', 'ADVAN12')
                secondary = secondary_pk_param_conversion_map(len(old), 1, removed=False)
                statements.subs(secondary)
                if trans == 'TRANS1':
                    statements.subs(
                        {
                            symbol('K12'): symbol('K23'),
                            symbol('K21'): symbol('K32'),
                            symbol('K13'): symbol('K24'),
                            symbol('K31'): symbol('K42'),
                        }
                    )
                elif trans == 'TRANS4':
                    statements.subs(
                        {
                            symbol('V1'): symbol('V2'),
                            symbol('Q2'): symbol('Q3'),
                            symbol('V2'): symbol('V3'),
                            symbol('Q3'): symbol('Q4'),
                            symbol('V3'): symbol('V4'),
                        }
                    )
                elif trans == 'TRANS6':
                    statements.subs({symbol('K31'): symbol('K42'), symbol('K21'): symbol('K32')})
            elif advan == 'ADVAN5' or advan == 'ADVAN7':
                model_record = model.control_stream.get_records('MODEL')[0]
                added = set(new.names) - set(old.names)
                added_name = list(added)[0]  # Assume only one!
                model_record.add_compartment(added_name, dosing=True)
                primary = primary_pk_param_conversion_map(len(old), 1)
                statements.subs(primary)
                secondary = secondary_pk_param_conversion_map(len(old), 1, removed=True)
                statements.subs(secondary)
            if isinstance(new.find_depot().dose, Infusion) and not statements.find_assignment('D1'):
                # Handle direct moving of Infusion to depot compartment
                statements.subs({'D2': 'D1'})
        elif old.find_depot() and not new.find_depot():
            # Depot was removed
            statements = model.statements
            if advan == 'ADVAN2':
                subs.replace_option('ADVAN2', 'ADVAN1')
                secondary = secondary_pk_param_conversion_map(len(old), 1)
                statements.subs(secondary)
            elif advan == 'ADVAN4':
                subs.replace_option('ADVAN4', 'ADVAN3')
                secondary = secondary_pk_param_conversion_map(len(old), 1)
                statements.subs(secondary)
                if trans == 'TRANS1':
                    statements.subs({symbol('K23'): symbol('K12'), symbol('K32'): symbol('K21')})
                elif trans == 'TRANS4':
                    statements.subs({symbol('V2'): symbol('V1'), symbol('V3'): symbol('V2')})
                elif trans == 'TRANS6':
                    statements.subs({symbol('K32'): symbol('K21')})
            elif advan == 'ADVAN12':
                subs.replace_option('ADVAN12', 'ADVAN11')
                secondary = secondary_pk_param_conversion_map(len(old), 1)
                statements.subs(secondary)
                if trans == 'TRANS1':
                    statements.subs(
                        {
                            symbol('K23'): symbol('K12'),
                            symbol('K32'): symbol('K21'),
                            symbol('K24'): symbol('K13'),
                            symbol('K42'): symbol('K31'),
                        }
                    )
                elif trans == 'TRANS4':
                    statements.subs(
                        {
                            symbol('V2'): symbol('V1'),
                            symbol('Q3'): symbol('Q2'),
                            symbol('V3'): symbol('V2'),
                            symbol('Q4'): symbol('Q3'),
                            symbol('V4'): symbol('V3'),
                        }
                    )
                elif trans == 'TRANS6':
                    statements.subs({symbol('K42'): symbol('K31'), symbol('K32'): symbol('K21')})
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


def primary_pk_param_conversion_map(ncomp, removed):
    """Conversion map for pk parameters for one removed compartment"""
    d = dict()
    for i in range(0, ncomp + 1):
        for j in range(0, ncomp + 1):
            if i == j or i == removed or j == removed:
                continue
            if i > removed:
                to_i = i - 1
            else:
                to_i = i
            if j > removed:
                to_j = j - 1
            else:
                to_j = j
            if (
                not (to_j == j and to_i == i)
                and i != 0
                and to_i != 0
                and not (i == ncomp and j == 0)
                and not (i == 0 and j == ncomp)
            ):
                d.update(
                    {
                        symbol(f'K{i}{j}'): symbol(f'K{to_i}{to_j}'),
                        symbol(f'K{i}T{j}'): symbol(f'K{to_i}T{to_j}'),
                    }
                )
    return d


def secondary_pk_param_conversion_map(ncomp, compno, removed=True):
    """Conversion map for pk parameters for one removed or added compartment

    ncomp - total number of compartments before removing/adding (including output)
    compno - number of removed/added compartment
    """
    d = dict()
    if removed:
        for i in range(compno + 1, ncomp + 1):
            d.update(
                {
                    symbol(f'S{i}'): symbol(f'S{i - 1}'),
                    symbol(f'F{i}'): symbol(f'F{i - 1}'),
                    symbol(f'R{i}'): symbol(f'R{i - 1}'),
                    symbol(f'D{i}'): symbol(f'D{i - 1}'),
                    symbol(f'ALAG{i}'): symbol(f'ALAG{i - 1}'),
                }
            )
    else:
        for i in range(compno, ncomp + 1):
            d.update(
                {
                    symbol(f'S{i}'): symbol(f'S{i + 1}'),
                    symbol(f'F{i}'): symbol(f'F{i + 1}'),
                    symbol(f'R{i}'): symbol(f'R{i + 1}'),
                    symbol(f'D{i}'): symbol(f'D{i + 1}'),
                    symbol(f'ALAG{i}'): symbol(f'ALAG{i + 1}'),
                }
            )
    return d


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
            error_statements.append(s)
        else:
            main_statements.append(s)

    main_statements.subs(trans)
    rec = model.get_pred_pk_record()
    rec.statements = main_statements
    error = model._get_error_record()
    if error:
        if len(error_statements) > 0:
            error_statements.pop(0)  # Remove the link statement
        error_statements.subs(trans)
        error.statements = error_statements


def update_lag_time(model, old, new):
    new_dosing = new.find_dosing()
    new_lag_time = new_dosing.lag_time
    old_lag_time = old.find_dosing().lag_time
    if new_lag_time != old_lag_time and new_lag_time != 0:
        ass = Assignment('ALAG1', new_lag_time)
        model.statements.add_before_odes(ass)
        new_dosing.lag_time = ass.symbol


def remove_compartments(model, old, new):
    """Remove compartments for ADVAN5 and ADVAN7"""
    model_record = model.control_stream.get_records('MODEL')[0]
    removed = set(old.names) - set(new.names)

    # Check if dosing was removed
    dose_comp = old.find_dosing()
    if dose_comp.name in removed:
        model_record.set_dosing(new.find_dosing().name)

    statements = model.statements
    for removed_name in removed:
        n = model_record.get_compartment_number(removed_name)
        model_record.remove_compartment(removed_name)
        primary = primary_pk_param_conversion_map(len(old), n)
        statements.subs(primary)
        secondary = secondary_pk_param_conversion_map(len(old), n)
        statements.subs(secondary)


def add_compartments(model, old, new):
    """Add compartments for ADVAN5 and ADVAN7"""
    model_record = model.control_stream.get_records('MODEL')[0]
    added = set(new.names) - set(old.names)
    statements = model.statements
    for added_name in added:
        model_record.add_compartment(added_name)
        primary = primary_pk_param_conversion_map(len(old), 1)
        statements.subs(primary)
        secondary = secondary_pk_param_conversion_map(len(old), 1, removed=True)
        statements.subs(secondary)
    if added:
        model_record.set_dosing(added_name)
