import re

import numpy as np

from pharmpy import data
from pharmpy.statements import (Assignment, Bolus, CompartmentalSystem, ExplicitODESystem, Infusion,
                                ModelStatements, ODESystem)
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
        for omega_record in model.control_stream.get_records('OMEGA'):
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
        if isinstance(new.find_dosing().dose, Bolus) and 'RATE' in model.dataset.columns:
            df = model.dataset
            df.drop(columns=['RATE'], inplace=True)
            model.dataset = df
        if isinstance(new.find_dosing().dose, Infusion) and \
                isinstance(old.find_dosing().dose, Bolus):
            dose = new.find_dosing().dose
            if dose.rate is None:
                ass = Assignment('D1', dose.duration)
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
        if not old.find_depot() and new.find_depot():
            # Depot was added
            subs = model.control_stream.get_records('SUBROUTINES')[0]
            advan = subs.get_option_startswith('ADVAN')
            trans = subs.get_option_startswith('TRANS')
            statements = model.statements
            _, rate = new.get_compartment_flows(new.find_depot(), out=True)[0]
            statements.add_before_odes(Assignment('KA', rate))
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
                    statements.subs({symbol('K12'): symbol('K23'), symbol('K21'): symbol('K32'),
                                     symbol('K13'): symbol('K24'), symbol('K31'): symbol('K42')})
                elif trans == 'TRANS4':
                    statements.subs({symbol('V1'): symbol('V2'), symbol('Q2'): symbol('Q3'),
                                     symbol('V2'): symbol('V3'), symbol('Q3'): symbol('Q4'),
                                     symbol('V3'): symbol('V4')})
                elif trans == 'TRANS6':
                    statements.subs({symbol('K31'): symbol('K42'), symbol('K21'): symbol('K32')})
            elif advan == 'ADVAN5' or advan == 'ADVAN7':
                model_record = model.control_stream.get_records('MODEL')[0]
                added = set(new.names) - set(old.names)
                added_name = list(added)[0]     # Assume only one!
                model_record.add_compartment(added_name, dosing=True)
                primary = primary_pk_param_conversion_map(len(old), 1, removed=True)
                statements.subs(primary)
                secondary = secondary_pk_param_conversion_map(len(old), 1, removed=True)
                statements.subs(secondary)
        elif old.find_depot() and not new.find_depot():
            # Depot was removed
            subs = model.control_stream.get_records('SUBROUTINES')[0]
            advan = subs.get_option_startswith('ADVAN')
            trans = subs.get_option_startswith('TRANS')
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
                    statements.subs({symbol('K23'): symbol('K12'), symbol('K32'): symbol('K21'),
                                     symbol('K24'): symbol('K13'), symbol('K42'): symbol('K31')})
                elif trans == 'TRANS4':
                    statements.subs({symbol('V2'): symbol('V1'), symbol('Q3'): symbol('Q2'),
                                     symbol('V3'): symbol('V2'), symbol('Q4'): symbol('Q3'),
                                     symbol('V4'): symbol('V3')})
                elif trans == 'TRANS6':
                    statements.subs({symbol('K42'): symbol('K31'), symbol('K32'): symbol('K21')})
            elif advan == 'ADVAN5' or advan == 'ADVAN7':
                model_record = model.control_stream.get_records('MODEL')[0]
                removed = set(old.names) - set(new.names)
                removed_name = list(removed)[0]     # Assume only one!
                dose_comp = new.find_dosing()
                model_record.set_dosing(dose_comp.name)
                n = model_record.get_compartment_number(removed_name)
                model_record.remove_compartment(removed_name)
                primary = primary_pk_param_conversion_map(len(old), n)
                statements.subs(primary)
                secondary = secondary_pk_param_conversion_map(len(old), n)
                statements.subs(secondary)


def primary_pk_param_conversion_map(ncomp, removed):
    """Conversion map for pk parameters for one removed compartment
    """
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
            if not (to_j == j and to_i == i) and i != 0 and to_i != 0 and \
                    not (i == ncomp and j == 0) and not (i == 0 and j == ncomp):
                d.update({symbol(f'K{i}{j}'): symbol(f'K{to_i}{to_j}'),
                          symbol(f'K{i}T{j}'): symbol(f'K{to_i}T{to_j}')})
    return d


def secondary_pk_param_conversion_map(ncomp, compno, removed=True):
    """Conversion map for pk parameters for one removed or added compartment

        ncomp - total number of compartments before removing/adding (including output)
        compno - number of removed/added compartment
    """
    d = dict()
    if removed:
        for i in range(compno + 1, ncomp + 1):
            d.update({symbol(f'S{i}'): symbol(f'S{i - 1}'),
                      symbol(f'F{i}'): symbol(f'F{i - 1}'),
                      symbol(f'R{i}'): symbol(f'R{i - 1}'),
                      symbol(f'D{i}'): symbol(f'D{i - 1}'),
                      symbol(f'ALAG{i}'): symbol(f'ALAG{i - 1}')})
    else:
        for i in range(compno, ncomp + 1):
            d.update({symbol(f'S{i}'): symbol(f'S{i + 1}'),
                      symbol(f'F{i}'): symbol(f'F{i + 1}'),
                      symbol(f'R{i}'): symbol(f'R{i + 1}'),
                      symbol(f'D{i}'): symbol(f'D{i + 1}'),
                      symbol(f'ALAG{i}'): symbol(f'ALAG{i + 1}')})
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
            error_statements.pop(0)        # Remove the link statement
        error_statements.subs(trans)
        error.statements = error_statements
