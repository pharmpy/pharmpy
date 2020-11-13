import itertools
import re

import numpy as np
import sympy

import pharmpy.modeling as modeling
from pharmpy import data
from pharmpy.plugins.nonmem.advan import (
    _advan3_trans,
    _advan4_trans,
    _advan11_trans,
    _advan12_trans,
)
from pharmpy.random_variables import RandomVariables, VariabilityLevel
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
    # FIXME: Setting the maps needs to be done here and not in loop. Automatic renumbering is
    #        probably the culprit. There should be a difference between added parameters and
    #        original parameters when it comes to which naming scheme to use
    if new_maps:
        for record, name_map, eta_map in new_maps:
            record.name_map = name_map
            record.eta_map = eta_map

    rvs_old = [rvs[0] for rvs in old.distributions()]

    for rvs, dist in new.distributions():
        rv_names = [rv.name for rv in rvs]
        if rvs not in rvs_old and set(rv_names).issubset(old_names):
            records = get_omega_records(model, rv_names)
            model.control_stream.remove_records(records)

            if len(rvs) == 1:
                omega_new, _ = create_omega_single(model, model.parameters[str(dist.std ** 2)])
            else:
                omega_new = create_omega_block(model, dist)

            omega_start = 1
            previous_size = None

            for omega_record in model.control_stream.get_records('OMEGA'):
                if omega_record != omega_new:
                    omega_start += len(omega_record)
                    previous_size = len(omega_record)
                else:
                    omega_record.parameters(omega_start, previous_size)
                    etas, _, _, _ = omega_record.random_variables(omega_start)

                    create_record_maps(etas, rvs, omega_record.name_map, omega_record.eta_map)

    next_eta = 1
    for omega_record in model.control_stream.get_records('OMEGA'):
        omega_record.renumber(next_eta)
        next_eta += len(omega_record)


def get_omega_records(model, params):
    records = []

    for omega_record in model.control_stream.get_records('OMEGA'):
        _, eta_map = omega_record.name_map, omega_record.eta_map

        for eta in eta_map.keys():
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
        for col in range(row + 1):
            elem = m.row(row).col(col)
            name = str(elem[0])
            omega = model.parameters[name]
            param_str += f'{omega.init}\t'

        param_str = f'{param_str.rstrip()}\n'
    record = model.control_stream.insert_record(param_str)
    return record


def create_record_maps(etas, rvs, name_map, eta_map):
    cov_rvs = RandomVariables(rvs).covariance_matrix()
    cov_rec = RandomVariables(etas).covariance_matrix()

    for row in range(cov_rvs.shape[0]):
        for col in range(cov_rvs.shape[1]):
            if row >= col:
                elem_new = cov_rvs.row(row).col(col)[0]
                elem_rec = cov_rec.row(row).col(col)[0]
                name_map[str(elem_new)] = name_map.pop(str(elem_rec))
    for rv, eta in zip(rvs, etas):
        eta_map[str(rv)] = eta_map.pop(str(eta))


def update_ode_system(model, old, new):
    """Update ODE system

    Handle changes from CompartmentSystem to ExplicitODESystem
    """
    if type(old) == CompartmentalSystem and type(new) == ExplicitODESystem:
        to_des(model, new)
    elif type(old) == CompartmentalSystem and type(new) == CompartmentalSystem:
        subs = model.control_stream.get_records('SUBROUTINES')[0]
        advan = subs.get_option_startswith('ADVAN')
        trans = subs.get_option_startswith('TRANS')
        if len(new) > 5 or new.n_connected(new.find_central()) != len(new) - 1:
            change_advan(model, 'ADVAN5', advan, trans)
            advan = 'ADVAN5'
        update_lag_time(model, old, new)
        if isinstance(new.find_dosing().dose, Bolus) and 'RATE' in model.dataset.columns:
            df = model.dataset
            df.drop(columns=['RATE'], inplace=True)
            model.dataset = df
        statements = model.statements
        to_advan = advan  # Default to not change ADVAN
        if not old.find_depot(model._old_statements) and new.find_depot(statements):
            # Depot was added
            comp, rate = new.get_compartment_outflows(new.find_depot(statements))[0]
            ass = Assignment('KA', rate)
            statements.add_before_odes(ass)
            new.add_flow(new.find_depot(statements), comp, ass.symbol)
            if advan == 'ADVAN1':
                to_advan = 'ADVAN2'
            elif advan == 'ADVAN3':
                to_advan = 'ADVAN4'
            elif advan == 'ADVAN11':
                to_advan = 'ADVAN12'
            if advan not in ['ADVAN5', 'ADVAN7']:
                subs.replace_option(advan, to_advan)
        elif old.find_depot(model._old_statements) and not new.find_depot(statements):
            # Depot was removed
            statements = model.statements
            if advan == 'ADVAN2':
                to_advan = 'ADVAN1'
            elif advan == 'ADVAN4':
                to_advan = 'ADVAN3'
            elif advan == 'ADVAN12':
                to_advan = 'ADVAN11'
            subs.replace_option(advan, to_advan)

        param_conversion = pk_param_conversion_map(
            new, model._compartment_map, from_advan=advan, to_advan=to_advan, trans=trans
        )
        statements.subs(param_conversion)

        if advan == 'ADVAN5' or advan == 'ADVAN7':
            remove_compartments(model, old, new)
            add_compartments(model, old, new)

        if isinstance(new.find_dosing().dose, Infusion) and not statements.find_assignment('D1'):
            # Handle direct moving of Infusion dose
            statements.subs({'D2': 'D1'})

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

    force_des(model, new)


def force_des(model, odes):
    """Switch to $DES if necessary"""
    if isinstance(odes, ExplicitODESystem):
        return

    amounts = {sympy.Function(amt.name)(symbol('t')) for amt in odes.amounts}
    if odes.atoms(sympy.Function) & amounts:
        modeling.explicit_odes(model)
        new = model.statements.ode_system
        to_des(model, new)


def to_des(model, new):
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

    for removed_name in removed:
        model_record.remove_compartment(removed_name)


def add_compartments(model, old, new):
    """Add compartments for ADVAN5 and ADVAN7

    Adds compartments to the beginning
    """
    model_record = model.control_stream.get_records('MODEL')[0]
    added = set(new.names) - set(old.names)
    order = {key: i for i, key in enumerate(new.names)}
    added = sorted(added, key=lambda d: order[d])
    statements = model.statements
    compmap = new_compartmental_map(new, model._compartment_map)
    for added_name in added:
        model_record.prepend_compartment(added_name)
        comp = new.find_compartment(added_name)
        comp_no = compmap[comp.name]
        for to_comp, rate in new.get_compartment_outflows(comp):
            to_comp_no = compmap[to_comp.name]
            if len(str(comp_no)) > 1 or len(str(to_comp_no)) > 1:
                separator = 'T'
            else:
                separator = ''
            ass = Assignment(f'K{comp_no}{separator}{to_comp_no}', rate)
            statements.add_before_odes(ass)

    if added:
        model_record.move_dosing_first()


def new_compartmental_map(cs, oldmap):
    """Create compartmental map for updated model
    cs - new compartmental system
    old - old compartmental map

    Can handle compartments from dosing to central
    """
    comp = cs.find_dosing()
    central = cs.find_central()
    i = 1
    compmap = dict()
    while True:
        compmap[comp.name] = i
        if comp is central:
            break
        comp, _ = cs.get_compartment_outflows(comp)[0]
        i += 1

    diff = len(cs) - len(oldmap)
    for name in cs.names:
        if name not in compmap.keys():
            compmap[name] = oldmap[name] + diff
    return compmap


def create_compartment_remap(oldmap, newmap):
    """Creates a map from old compartment number to new compartment number

    For all compartments where remapping is needed
    Assume that compartments with same name in new and old are the same compartments
    """
    remap = dict()
    for name, number in oldmap.items():
        if name in newmap:
            remap[number] = newmap[name]
    return remap


def pk_param_conversion_map(cs, oldmap, from_advan=None, to_advan=None, trans=None):
    """Conversion map for pk parameters for removed or added compartment"""
    newmap = new_compartmental_map(cs, oldmap)
    remap = create_compartment_remap(oldmap, newmap)
    d = dict()
    for old, new in remap.items():
        d[symbol(f'S{old}')] = symbol(f'S{new}')
        d[symbol(f'F{old}')] = symbol(f'F{new}')
        d[symbol(f'R{old}')] = symbol(f'R{new}')
        d[symbol(f'D{old}')] = symbol(f'D{new}')
        d[symbol(f'ALAG{old}')] = symbol(f'ALAG{new}')
    if from_advan is None or from_advan == 'ADVAN5' or from_advan == 'ADVAN7':
        for i, j in itertools.product(range(1, len(oldmap)), range(0, len(oldmap))):
            if i != j and (i in remap and (j in remap or j == 0)):
                if i in remap:
                    to_i = remap[i]
                else:
                    to_i = i
                if j in remap:
                    to_j = remap[j]
                else:
                    to_j = j
                d[symbol(f'K{i}{j}')] = symbol(f'K{to_i}{to_j}')
                d[symbol(f'K{i}T{j}')] = symbol(f'K{to_i}T{to_j}')
    elif from_advan == 'ADVAN3':
        if to_advan == 'ADVAN4':
            if trans == 'TRANS4':
                d[symbol('V1')] = symbol('V2')
                d[symbol('V2')] = symbol('V3')
            elif trans == 'TRANS6':
                d[symbol('K21')] = symbol('K32')
            else:  # TRANS1
                d[symbol('K12')] = symbol('K23')
                d[symbol('K21')] = symbol('K32')
    elif from_advan == 'ADVAN4':
        if to_advan == 'ADVAN3':
            if trans == 'TRANS4':
                d.update({symbol('V2'): symbol('V1'), symbol('V3'): symbol('V2')})
            elif trans == 'TRANS6':
                d.update({symbol('K32'): symbol('K21')})
            else:  # TRANS1
                d.update({symbol('K23'): symbol('K12'), symbol('K32'): symbol('K21')})
    elif from_advan == 'ADVAN11':
        if to_advan == 'ADVAN12':
            if trans == 'TRANS4':
                d.update(
                    {
                        symbol('V1'): symbol('V2'),
                        symbol('Q2'): symbol('Q3'),
                        symbol('V2'): symbol('V3'),
                        symbol('Q3'): symbol('Q4'),
                        symbol('V3'): symbol('V4'),
                    }
                )
            elif trans == 'TRANS6':
                d.update({symbol('K31'): symbol('K42'), symbol('K21'): symbol('K32')})
            else:  # TRANS1
                d.update(
                    {
                        symbol('K12'): symbol('K23'),
                        symbol('K21'): symbol('K32'),
                        symbol('K13'): symbol('K24'),
                        symbol('K31'): symbol('K42'),
                    }
                )
    elif from_advan == 'ADVAN12':
        if to_advan == 'ADVAN11':
            if trans == 'TRANS4':
                d.update(
                    {
                        symbol('V2'): symbol('V1'),
                        symbol('Q3'): symbol('Q2'),
                        symbol('V3'): symbol('V2'),
                        symbol('Q4'): symbol('Q3'),
                        symbol('V4'): symbol('V3'),
                    }
                )
            elif trans == 'TRANS6':
                d.update({symbol('K42'): symbol('K31'), symbol('K32'): symbol('K21')})
            else:  # TRANS1
                d.update(
                    {
                        symbol('K23'): symbol('K12'),
                        symbol('K32'): symbol('K21'),
                        symbol('K24'): symbol('K13'),
                        symbol('K42'): symbol('K31'),
                    }
                )
    return d


def change_advan(model, advan, oldadvan, oldtrans):
    """Change from one advan to another"""
    assignments = []
    newtrans = None
    if advan == oldadvan:
        return
    subs = model.control_stream.get_records('SUBROUTINES')[0]
    if advan == 'ADVAN5' or advan == 'ADVAN7':
        newtrans = 'TRANS1'
        if oldadvan == 'ADVAN1':
            if oldtrans == 'TRANS1':
                ass = Assignment('K10', symbol('K'))
            else:  # TRANS2
                ass = Assignment('K10', symbol('CL') / symbol('V'))
            assignments.append(ass)
        elif oldadvan == 'ADVAN2':
            # FIXME: Might create too many new parameters
            assignments.append(Assignment('K12', symbol('KA')))
            if oldtrans == 'TRANS1':
                ass = Assignment('K20', symbol('K'))
            else:  # TRANS2
                ass = Assignment('K20', symbol('CL') / symbol('V'))
            assignments.append(ass)
        elif oldadvan == 'ADVAN3':
            k, k12, k21 = _advan3_trans(oldtrans)
            ass1 = Assignment('K12', k12)
            ass2 = Assignment('K21', k21)
            ass3 = Assignment('K20', k)
            assignments.extend([ass1, ass2, ass3])
        elif oldadvan == 'ADVAN4':
            k, k23, k32, ka = _advan4_trans(oldtrans)
            ass1 = Assignment('K12', ka)
            ass2 = Assignment('K23', k23)
            ass3 = Assignment('K32', k32)
            ass4 = Assignment('K30', k)
            assignments.extend([ass1, ass2, ass3, ass4])
        elif oldadvan == 'ADVAN11':
            k, k12, k21, k13, k31 = _advan11_trans(oldtrans)
            ass1 = Assignment('K12', k12)
            ass2 = Assignment('K21', k21)
            ass3 = Assignment('K13', k13)
            ass4 = Assignment('K31', k31)
            ass5 = Assignment('K30', k)
            assignments.extend([ass1, ass2, ass3, ass4, ass5])
        elif oldadvan == 'ADVAN12':
            k, k23, k32, k24, k42, ka = _advan12_trans(oldtrans)
            ass1 = Assignment('K12', ka)
            ass2 = Assignment('K23', k23)
            ass3 = Assignment('K32', k32)
            ass4 = Assignment('K24', k24)
            ass5 = Assignment('K42', k42)
            ass6 = Assignment('K40', k)
            assignments.extend([ass1, ass2, ass3, ass4, ass5, ass6])

        for ass in assignments:
            model.statements.add_before_odes(ass)

        subs.replace_option(oldadvan, advan)
        if newtrans is not None:
            subs.replace_option(oldtrans, newtrans)

        mod = model.control_stream.insert_record('$MODEL\n')
        comps = {v: k for k, v in model._compartment_map.items()}
        i = 1
        while True:
            if i not in comps:
                break
            if i == 1:
                mod.add_compartment(comps[i], dosing=True)
            else:
                mod.add_compartment(comps[i], dosing=False)
            i += 1
