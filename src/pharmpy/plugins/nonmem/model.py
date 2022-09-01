# The NONMEM Model class

import re
import shutil
import warnings
from io import StringIO
from pathlib import Path

import sympy
from sympy import Symbol as S

import pharmpy.data
import pharmpy.model
import pharmpy.plugins.nonmem
import pharmpy.plugins.nonmem.dataset
from pharmpy.data import DatasetError
from pharmpy.datainfo import ColumnInfo, DataInfo
from pharmpy.estimation import EstimationStep, EstimationSteps
from pharmpy.model import ModelSyntaxError
from pharmpy.modeling.write_csv import write_csv
from pharmpy.parameters import Parameter, Parameters
from pharmpy.plugins.nonmem.results import NONMEMChainedModelfitResults
from pharmpy.plugins.nonmem.table import NONMEMTableFile, PhiTable
from pharmpy.random_variables import RandomVariable, RandomVariables
from pharmpy.statements import Assignment, CompartmentalSystem, ODESystem
from pharmpy.workflows import NullModelDatabase, default_model_database

from .advan import compartmental_model
from .nmtran_parser import NMTranParser
from .records.factory import create_record
from .update import (
    update_abbr_record,
    update_ccontra,
    update_estimation,
    update_parameters,
    update_random_variables,
    update_statements,
)


def detect_model(src, *args, **kwargs):
    """Check if src represents a NONMEM control stream
    i.e. check if it is a file that contain $PRO
    """
    if not isinstance(src, str):
        return None
    is_control_stream = re.search(r'^\s*\$PRO', src, re.MULTILINE)
    if is_control_stream:
        return Model
    else:
        return None


def convert_model(model):
    """Convert any model into a NONMEM model"""
    if isinstance(model, Model):
        return model
    from pharmpy.modeling import convert_model

    model = convert_model(model, 'generic')
    code = '$PROBLEM\n'
    code += (
        '$INPUT '
        + ' '.join(
            f'{column.name}=DROP' if column.drop else column.name for column in model.datainfo
        )
        + '\n'
    )
    code += '$DATA file.csv IGNORE=@\n'
    if model.statements.ode_system is None:
        code += '$PRED\nY=X\n'
    else:
        code += '$SUBROUTINES ADVAN1 TRANS2\n'
        code += '$PK\nY=X\n'
        code += '$ERROR\nA=B'
    nm_model = pharmpy.model.Model.create_model(StringIO(code))
    nm_model._datainfo = model.datainfo
    nm_model.random_variables = model.random_variables
    nm_model._parameters = model.parameters
    nm_model._old_parameters = Parameters()
    nm_model.statements = model.statements
    if hasattr(model, 'name'):
        nm_model.name = model.name
    # FIXME: No handling of other DVs
    nm_model.dependent_variable = sympy.Symbol('Y')
    nm_model.value_type = model.value_type
    nm_model._data_frame = model.dataset
    nm_model._estimation_steps = model.estimation_steps
    nm_model._initial_individual_estimates = model.initial_individual_estimates
    nm_model.observation_transformation = model.observation_transformation.subs(
        model.dependent_variable, nm_model.dependent_variable
    )
    nm_model.description = model.description
    nm_model.update_source()
    try:
        nm_model.database = model.database
    except AttributeError:
        pass
    if model.statements.ode_system:
        nm_model._compartment_map = {
            name: i for i, name in enumerate(model.statements.ode_system.compartment_names, start=1)
        }
    return nm_model


class Model(pharmpy.model.Model):
    def __init__(self, code, path=None, **kwargs):
        self.modelfit_results = None
        parser = NMTranParser()
        if path is None:
            self._name = 'run1'
            self.database = NullModelDatabase()
            self.filename_extension = '.ctl'
        else:
            self._name = path.stem
            self.database = default_model_database(path=path.parent)
            self.filename_extension = path.suffix
        self.control_stream = parser.parse(code)
        self._create_datainfo()
        self._initial_individual_estimates_updated = False
        self._updated_etas_file = None
        self._dataset_updated = False
        self._parent_model = None
        self.dependent_variable = S('Y')
        self.observation_transformation = self.dependent_variable
        self._old_observation_transformation = self.dependent_variable
        if path is None:
            self._modelfit_results = None
        else:
            self.read_modelfit_results(path.parent)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, new_name):
        m = re.search(r'.*?(\d+)$', new_name)
        if m:
            n = int(m.group(1))
            for table in self.control_stream.get_records('TABLE'):
                table_path = table.path
                table_name = table_path.stem
                m = re.search(r'(.*?)(\d+)$', table_name)
                if m:
                    table_stem = m.group(1)
                    new_table_name = f'{table_stem}{n}'
                    table.path = table_path.parent / new_table_name
        self._name = new_name

    @property
    def description(self):
        rec = self.control_stream.get_records('PROBLEM')[0]
        return rec.title

    @description.setter
    def description(self, value):
        rec = self.control_stream.get_records('PROBLEM')[0]
        rec.title = value

    @property
    def value_type(self):
        try:
            return self._value_type
        except AttributeError:
            pass
        ests = self.control_stream.get_records('ESTIMATION')
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
        if f_flag in self.statements.free_symbols:
            tp = f_flag
        self._value_type = tp
        return tp

    @value_type.setter
    def value_type(self, value):
        super(Model, self.__class__).value_type.fset(self, value)

    @property
    def modelfit_results(self):
        return self._modelfit_results

    @modelfit_results.setter
    def modelfit_results(self, res):
        self._modelfit_results = res

    def update_source(self, path=None, force=False, nofiles=False):
        """Update the source

        path - path to modelfile
        nofiles - Set to not write any files (i.e. dataset, phi input etc)
        """
        self._update_initial_individual_estimates(path, nofiles)
        if hasattr(self, '_random_variables'):
            # FIXME: better solution would be to have system for handling dummy parameters etc.
            if not self.random_variables.etas:
                omega = Parameter('DUMMYOMEGA', init=0, fix=True)
                eta = RandomVariable.normal('eta_dummy', 'iiv', 0, omega.symbol)
                statement = Assignment(sympy.Symbol('DUMMYETA'), sympy.Symbol(eta.name))
                self.statements = statement + self.statements
                self.random_variables.append(eta)
                self.parameters = Parameters([p for p in self.parameters] + [omega])
            update_random_variables(self, self._old_random_variables, self._random_variables)
            self._old_random_variables = self._random_variables.copy()
        if hasattr(self, '_parameters'):
            update_parameters(self, self._old_parameters, self._parameters)
            self._old_parameters = self._parameters
        trans = self.parameter_translation(reverse=True, remove_idempotent=True, as_symbols=True)
        rv_trans = self.rv_translation(reverse=True, remove_idempotent=True, as_symbols=True)
        trans.update(rv_trans)
        if pharmpy.plugins.nonmem.conf.write_etas_in_abbr:
            abbr_trans = self._abbr_translation(rv_trans)
            trans.update(abbr_trans)
        if trans:
            self.statements  # Read statements unless read
        if hasattr(self, '_statements'):
            update_statements(self, self._old_statements, self._statements, trans)
            self._old_statements = self._statements

        if (
            self._dataset_updated
            or self.datainfo != self._old_datainfo
            or self.datainfo.path != self._old_datainfo.path
        ):
            # FIXME: If no name set use the model name. Set that when setting dataset to input!
            if self.datainfo.path is None:  # or self.datainfo.path == self._old_datainfo.path:
                if path is not None:
                    dir_path = path.parent
                else:
                    dir_path = self.name + ".csv"
                if not nofiles:
                    datapath = write_csv(self, path=dir_path, force=force)
                    self.datainfo = self.datainfo.derive(path=datapath.name)
                else:
                    self.datainfo = self.datainfo.derive(path=Path(dir_path))
            data_record = self.control_stream.get_records('DATA')[0]

            label = self.datainfo.names[0]
            data_record.ignore_character_from_header(label)
            self._update_input()

            # Remove IGNORE/ACCEPT. Could do diff between old dataset and find simple
            # IGNOREs to add i.e. for filter out certain ID.
            del data_record.ignore
            del data_record.accept
            self._dataset_updated = False
            self._old_datainfo = self.datainfo

            path = self.datainfo.path
            if path is not None:
                assert (
                    not path.exists() or path.is_file()
                ), f'input path change, but no file exists at target {str(path)}'
                data_record = self.control_stream.get_records('DATA')[0]
                data_record.filename = str(path)

        self._update_sizes()
        update_estimation(self)

        if self.observation_transformation != self._old_observation_transformation:
            if not nofiles:
                update_ccontra(self, path, force)

    def _abbr_translation(self, rv_trans):
        abbr_pharmpy = self.control_stream.abbreviated.translate_to_pharmpy_names()
        abbr_replace = self.control_stream.abbreviated.replace
        abbr_trans = update_abbr_record(self, rv_trans)
        abbr_recs = {
            S(abbr_pharmpy[value]): S(key)
            for key, value in abbr_replace.items()
            if value in abbr_pharmpy.keys()
        }
        abbr_trans.update(abbr_recs)
        return abbr_trans

    def _update_sizes(self):
        """Update $SIZES if needed"""
        all_sizes = self.control_stream.get_records('SIZES')
        if len(all_sizes) == 0:
            sizes = create_record('$SIZES ')
        else:
            sizes = all_sizes[0]
        odes = self.statements.ode_system

        if odes is not None and isinstance(odes, CompartmentalSystem):
            n_compartments = len(odes)
            sizes.PC = n_compartments
        thetas = [p for p in self.parameters if p.symbol not in self.random_variables.free_symbols]
        sizes.LTH = len(thetas)

        if len(all_sizes) == 0 and len(str(sizes)) > 7:
            self.control_stream.insert_record(str(sizes))

    def _update_initial_individual_estimates(self, path, nofiles=False):
        """Update $ETAS

        Could have 0 FIX in model. Need to read these
        """
        if path is None:  # What to do here?
            phi_path = Path('.')
        else:
            phi_path = path.parent
        phi_path /= f'{self.name}_input.phi'

        if self._initial_individual_estimates_updated:
            etas = self.initial_individual_estimates
            zero_fix = self._zero_fix_rvs(eta=True)
            if zero_fix:
                for eta in zero_fix:
                    etas[eta] = 0
            etas = self._sort_eta_columns(etas)
            if not nofiles:
                phi = PhiTable(df=etas)
                table_file = NONMEMTableFile(tables=[phi])
                table_file.write(phi_path)
            # FIXME: This is a common operation
            eta_records = self.control_stream.get_records('ETAS')
            if eta_records:
                record = eta_records[0]
            else:
                record = self.control_stream.append_record('$ETAS ')
            record.path = phi_path
        elif self._updated_etas_file:
            eta_records = self.control_stream.get_records('ETAS')
            if eta_records:
                record = eta_records[0]
            else:
                record = self.control_stream.append_record('$ETAS')
            shutil.copy(self._updated_etas_file, phi_path)
            record.path = phi_path

        if self._initial_individual_estimates_updated or self._updated_etas_file:
            first_est_record = self.control_stream.get_records('ESTIMATION')[0]
            try:
                first_est_record.option_pairs['MCETA']
            except KeyError:
                first_est_record.set_option('MCETA', 1)
            self._updated_etas_file = None
            self._initial_individual_estimates_updated = False

    def validate(self):
        """Validates NONMEM model (records) syntactically."""
        self.control_stream.validate()

    @property
    def parameters(self):
        """Get the Parameters of all parameters"""
        try:
            return self._parameters
        except AttributeError:
            pass

        self._read_parameters()
        if (
            'comment' in pharmpy.plugins.nonmem.conf.parameter_names
            or 'abbr' in pharmpy.plugins.nonmem.conf.parameter_names
        ):
            self.statements
            # reading statements might change parameters. Resetting _old_parameters
            self._old_parameters = self._parameters

        if not self.random_variables.validate_parameters(self._parameters.inits):
            nearest = self.random_variables.nearest_valid_parameters(self._parameters.inits)
            before, after = self._compare_before_after_params(self._parameters.inits, nearest)
            warnings.warn(
                f"Adjusting initial estimates to create positive semidefinite "
                f"omega/sigma matrices.\nBefore adjusting:  {before}.\n"
                f"After adjusting: {after}"
            )
            params = self._parameters.set_initial_estimates(nearest)
            self._parameters = params
            self._old_parameters = params
        return self._parameters

    def _read_parameters(self):
        next_theta = 1
        params = []
        for theta_record in self.control_stream.get_records('THETA'):
            thetas = theta_record.parameters(next_theta, seen_labels={p.name for p in params})
            params.extend(thetas)
            next_theta += len(thetas)
        next_omega = 1
        previous_size = None
        for omega_record in self.control_stream.get_records('OMEGA'):
            omegas, next_omega, previous_size = omega_record.parameters(
                next_omega, previous_size, seen_labels={p.name for p in params}
            )
            params.extend(omegas)
        next_sigma = 1
        previous_size = None
        for sigma_record in self.control_stream.get_records('SIGMA'):
            sigmas, next_sigma, previous_size = sigma_record.parameters(
                next_sigma, previous_size, seen_labels={p.name for p in params}
            )
            params.extend(sigmas)
        self._parameters = Parameters(params)
        self._old_parameters = self._parameters

    @parameters.setter
    def parameters(self, params):
        """params can be a Parameters or a dict-like with name: value

        Current restrictions:
         * Parameters only supported for new initial estimates and fix
         * Only set the exact same parameters. No additions and no removing of parameters
        """
        self.parameters
        super(Model, self.__class__).parameters.fset(self, params)

    @property
    def initial_individual_estimates(self):
        """Initial individual estimates

        These are taken from the $ETAS FILE. 0 FIX ETAs are removed.
        If no $ETAS is present None will be returned.

        Setter assumes that all IDs are present
        """
        try:
            return self._initial_individual_estimates
        except AttributeError:
            pass
        etas = self.control_stream.get_records('ETAS')
        if etas:
            path = Path(etas[0].path)
            if not path.is_absolute():
                source_dir = self.database.retrieve_file(
                    self.name, self.name + self.filename_extension
                ).parent
                path = source_dir / path
                path = path.resolve()
            phi_tables = NONMEMTableFile(path)
            rv_names = [rv.name for rv in self.random_variables if rv.name.startswith('ETA')]
            etas = next(phi_tables).etas[rv_names]
            self._initial_individual_estimates = etas
        else:
            self._initial_individual_estimates = None
        return self._initial_individual_estimates

    @initial_individual_estimates.setter
    def initial_individual_estimates(self, estimates):
        rv_names = {rv.name for rv in self.random_variables if rv.name.startswith('ETA')}
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
            estimates = self._sort_eta_columns(estimates)
        self._initial_individual_estimates = estimates
        self._initial_individual_estimates_updated = True
        self._updated_etas_file = None

    def _sort_eta_columns(self, df):
        return df.reindex(sorted(df.columns), axis=1)

    @property
    def statements(self):
        try:
            return self._statements
        except AttributeError:
            pass

        rec = self.get_pred_pk_record()
        statements = rec.statements

        des = self._get_des_record()
        error = self._get_error_record()
        if error:
            sub = self.control_stream.get_records('SUBROUTINES')[0]
            comp = compartmental_model(self, sub.advan, sub.trans, des)
            trans_amounts = dict()
            if comp is not None:
                cm, link = comp
                statements += [cm, link]
                for i, amount in enumerate(cm.amounts, start=1):
                    trans_amounts[sympy.Symbol(f"A({i})")] = amount
            else:
                statements += ODESystem()  # FIXME: Placeholder for ODE-system
                # FIXME: Dummy link statement
                statements += Assignment(S('F'), S('F'))
            statements += error.statements
            if trans_amounts:
                statements = statements.subs(trans_amounts)

        if not hasattr(self, '_parameters'):
            self._read_parameters()

        trans_statements, trans_params = self._create_name_trans(statements)
        for theta in self.control_stream.get_records('THETA'):
            theta.update_name_map(trans_params)
        for omega in self.control_stream.get_records('OMEGA'):
            omega.update_name_map(trans_params)
        for sigma in self.control_stream.get_records('SIGMA'):
            sigma.update_name_map(trans_params)

        d_par = dict()
        d_rv = dict()
        for key, value in trans_params.items():
            if key in self.parameters:
                d_par[key] = value
            else:
                d_rv[S(key)] = value
        self.random_variables.subs(d_rv)
        new = []
        for p in self.parameters:
            if p.name in d_par:
                newparam = Parameter(
                    name=d_par[p.name], init=p.init, lower=p.lower, upper=p.upper, fix=p.fix
                )
            else:
                newparam = p
            new.append(newparam)
        self.parameters = Parameters(new)

        statements = statements.subs(trans_statements)

        self._statements = statements
        self._old_statements = statements
        return statements

    @statements.setter
    def statements(self, statements_new):
        self.statements  # Read in old statements
        self._statements = statements_new

    def get_pred_pk_record(self):
        pred = self.control_stream.get_records('PRED')

        if not pred:
            pk = self.control_stream.get_records('PK')
            if not pk:
                raise ModelSyntaxError('Model has no $PK or $PRED')
            return pk[0]
        else:
            return pred[0]

    def _get_pk_record(self):
        pk = self.control_stream.get_records('PK')
        if pk:
            pk = pk[0]
        return pk

    def _get_error_record(self):
        error = self.control_stream.get_records('ERROR')
        if error:
            error = error[0]
        return error

    def _get_des_record(self):
        des = self.control_stream.get_records('DES')
        if des:
            des = des[0]
        return des

    def _create_name_trans(self, statements):
        rvs = self.random_variables

        conf_functions = {
            'comment': self._name_as_comments(statements),
            'abbr': self._name_as_abbr(rvs),
            'basic': self._name_as_basic(),
        }

        abbr = self.control_stream.abbreviated.replace
        pset_current = {
            **self.parameter_translation(reverse=True),
            **{rv: rv for rv in [rv.name for rv in rvs]},
        }
        sset_current = {
            **abbr,
            **{
                rv.name: rv.name
                for rv in rvs
                if rv.name not in abbr.keys() and rv.symbol in statements.free_symbols
            },
            **{
                p: p
                for p in pset_current.values()
                if p not in abbr.keys() and S(p) in statements.free_symbols
            },
        }

        trans_sset, trans_pset = dict(), dict()
        names_sset_translated, names_pset_translated, names_basic = [], [], []
        clashing_symbols = set()

        for setting in pharmpy.plugins.nonmem.conf.parameter_names:
            trans_sset_setting, trans_pset_setting = conf_functions[setting]
            if setting != 'basic':
                clashing_symbols.update(
                    self._clashing_symbols(statements, {**trans_sset_setting, **trans_pset_setting})
                )
            for name_current, name_new in trans_sset_setting.items():
                name_nonmem = sset_current[name_current]

                if S(name_new) in clashing_symbols or name_nonmem in names_sset_translated:
                    continue

                name_in_sset_current = {v: k for k, v in sset_current.items()}[name_nonmem]
                trans_sset[name_in_sset_current] = name_new
                names_sset_translated.append(name_nonmem)

                if name_nonmem in pset_current.values() and name_new in pset_current.keys():
                    names_pset_translated.append(name_nonmem)

            for name_current, name_new in trans_pset_setting.items():
                name_nonmem = pset_current[name_current]

                if S(name_new) in clashing_symbols or name_nonmem in names_pset_translated:
                    continue

                trans_pset[name_current] = name_new
                names_pset_translated.append(name_nonmem)

            if setting == 'basic':
                params_left = [k for k in pset_current.keys() if k not in names_pset_translated]
                params_left += [rv.name for rv in rvs if rv.name not in names_sset_translated]
                names_basic = [name for name in params_left if name not in names_sset_translated]
                break

        if clashing_symbols:
            warnings.warn(
                f'The parameter names {clashing_symbols} are also names of variables '
                f'in the model code. Falling back to the in naming scheme config '
                f'names for these.'
            )

        names_nonmem_all = [rv.name for rv in rvs] + [
            key for key in self.parameter_translation().keys()
        ]

        if set(names_nonmem_all) - set(names_sset_translated + names_pset_translated + names_basic):
            raise ValueError(
                'Mismatch in number of parameter names, all have not been accounted for. If basic '
                'NONMEM-names are desired as fallback, double-check that "basic" is included in '
                'config-settings for parameter_names.'
            )
        return trans_sset, trans_pset

    def _name_as_comments(self, statements):
        params_current = self.parameter_translation(remove_idempotent=True)
        for name_abbr, name_nonmem in self.control_stream.abbreviated.replace.items():
            if name_nonmem in params_current.keys():
                params_current[name_abbr] = params_current.pop(name_nonmem)
        trans_params = {
            name_comment: name_comment
            for name_current, name_comment in params_current.items()
            if S(name_current) not in statements.free_symbols
        }
        trans_statements = {
            name_current: name_comment
            for name_current, name_comment in params_current.items()
            if S(name_current) in statements.free_symbols
        }
        return trans_statements, trans_params

    def _name_as_abbr(self, rvs):
        pharmpy_names = self.control_stream.abbreviated.translate_to_pharmpy_names()
        params_current = self.parameter_translation(remove_idempotent=True, reverse=True)
        trans_params = {
            name_nonmem: name_abbr
            for name_nonmem, name_abbr in pharmpy_names.items()
            if name_nonmem in self.parameter_translation().keys()
            or name_nonmem in [rv.name for rv in rvs]
        }
        for name_nonmem, name_abbr in params_current.items():
            if name_abbr in trans_params.keys():
                trans_params[name_nonmem] = trans_params.pop(name_abbr)
        trans_statements = {
            name_abbr: pharmpy_names[name_nonmem]
            for name_abbr, name_nonmem in self.control_stream.abbreviated.replace.items()
        }
        return trans_statements, trans_params

    def _name_as_basic(self):
        trans_params = {
            name_current: name_nonmem
            for name_current, name_nonmem in self.parameter_translation(reverse=True).items()
            if name_current != name_nonmem
        }
        trans_statements = self.control_stream.abbreviated.replace
        return trans_statements, trans_params

    @staticmethod
    def _clashing_symbols(statements, trans_statements):
        parameter_symbols = {S(symb) for _, symb in trans_statements.items()}
        clashing_symbols = parameter_symbols & statements.free_symbols
        return clashing_symbols

    def replace_abbr(self, replace):
        for key, value in replace.items():
            try:
                self.parameters[key].name = value
            except KeyError:
                pass
        self.random_variables.rename(replace)

    def _zero_fix_rvs(self, eta=True):
        zero_fix = []
        if eta:
            prev_cov = None
            next_omega = 1
            for omega_record in self.control_stream.get_records('OMEGA'):
                _, next_omega, prev_cov, new_zero_fix = omega_record.random_variables(
                    next_omega, prev_cov
                )
                zero_fix += new_zero_fix
        else:
            prev_cov = None
            next_sigma = 1
            for sigma_record in self.control_stream.get_records('SIGMA'):
                _, next_sigma, prev_cov, new_zero_fix = sigma_record.random_variables(
                    next_sigma, prev_cov
                )
                zero_fix += new_zero_fix
        return zero_fix

    @property
    def random_variables(self):
        try:
            return self._random_variables
        except AttributeError:
            pass
        rvs = RandomVariables()
        next_omega = 1
        prev_cov = None

        for omega_record in self.control_stream.get_records('OMEGA'):
            etas, next_omega, prev_cov, _ = omega_record.random_variables(next_omega, prev_cov)
            rvs.extend(etas)
        self.adjust_iovs(rvs)
        next_sigma = 1
        prev_cov = None
        for sigma_record in self.control_stream.get_records('SIGMA'):
            epsilons, next_sigma, prev_cov, _ = sigma_record.random_variables(next_sigma, prev_cov)
            rvs.extend(epsilons)
        self._random_variables = rvs
        self._old_random_variables = rvs.copy()

        if (
            'comment' in pharmpy.plugins.nonmem.conf.parameter_names
            or 'abbr' in pharmpy.plugins.nonmem.conf.parameter_names
        ):
            self.statements

        return self._random_variables

    @staticmethod
    def adjust_iovs(rvs):
        for i, rv in enumerate(rvs):
            try:
                next_rv = rvs[i + 1]
            except IndexError:
                break

            if rv.level != 'IOV' and next_rv.level == 'IOV':
                associated_rvs = rv.joint_names
                if len(associated_rvs) > 1:
                    for rv_con in associated_rvs:
                        rvs[rv_con].level = 'IOV'
                else:
                    rv.level = 'IOV'

    @random_variables.setter
    def random_variables(self, new):
        self.random_variables  # Read in old random variables
        self._random_variables = new.copy()

    @property
    def estimation_steps(self):
        try:
            return self._estimation_steps
        except AttributeError:
            pass

        steps = parse_estimation_steps(self)
        self._estimation_steps = steps
        self._old_estimation_steps = steps
        return steps

    @estimation_steps.setter
    def estimation_steps(self, value):
        self.estimation_steps
        self._estimation_steps = value

    @property
    def model_code(self):
        self.update_source(nofiles=True)
        return str(self.control_stream)

    def _read_dataset_path(self):
        record = next(iter(self.control_stream.get_records('DATA')), None)
        if record is None:
            return None
        path = Path(record.filename)
        if not path.is_absolute():
            try:
                dbpath = self.database.retrieve_file(self.name, path)
            except FileNotFoundError:
                pass
            else:
                if dbpath is not None:
                    path = dbpath
        try:
            return path.resolve()
        except FileNotFoundError:
            return path

    @property
    def dataset(self):
        try:
            return self._data_frame
        except AttributeError:
            self._data_frame = self._read_dataset(raw=False)
        return self._data_frame

    @dataset.setter
    def dataset(self, df):
        self._dataset_updated = True
        self._data_frame = df
        self.datainfo = self.datainfo.derive(path=None)
        self.update_datainfo()

    def read_raw_dataset(self, parse_columns=tuple()):
        return self._read_dataset(raw=True, parse_columns=parse_columns)

    @staticmethod
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

    def _column_info(self):
        """List all column names in order.
        Use the synonym when synonym exists.
        return tuple of two lists, colnames, and drop together with a dictionary
        of replacements for reserved names (aka synonyms).
        Anonymous columns, i.e. DROP or SKIP alone, will be given unique names _DROP1, ...
        """
        input_records = self.control_stream.get_records("INPUT")
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
                        (reserved_name, synonym) = Model._synonym(key, value)
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

    def _update_input(self):
        """Update $INPUT

        currently supporting append columns at end and removing columns
        And add/remove DROP
        """
        input_records = self.control_stream.get_records("INPUT")
        _, drop, _, colnames = self._column_info()
        keep = []
        i = 0
        for child in input_records[0].root.children:
            if child.rule != 'option':
                keep.append(child)
                continue

            if (colnames[i] is not None and (colnames[i] != self.datainfo[i].name)) or (
                not drop[i]
                and (self.datainfo[i].drop or self.datainfo[i].datatype == 'nmtran-date')
            ):
                dropped = self.datainfo[i].drop or self.datainfo[i].datatype == 'nmtran-date'
                anonymous = colnames[i] is None
                key = 'DROP' if anonymous and dropped else self.datainfo[i].name
                value = 'DROP' if not anonymous and dropped else None
                new = input_records[0]._create_option(key, value)
                keep.append(new)
            else:
                keep.append(child)

            i += 1

            if i >= len(self.datainfo):
                last_child = input_records[0].root.children[-1]
                if last_child.rule == 'ws' and '\n' in str(last_child):
                    keep.append(last_child)
                break

        input_records[0].root.children = keep

        last_input_record = input_records[-1]
        for ci in self.datainfo[len(colnames) :]:
            last_input_record.append_option(ci.name, 'DROP' if ci.drop else None)

    def _replace_synonym_in_filters(filters, replacements):
        result = []
        for f in filters:
            if f.COLUMN in replacements:
                s = ''
                for child in f.children:
                    if child.rule == 'COLUMN':
                        value = replacements[f.COLUMN]
                    else:
                        value = str(child)
                    s += value
            else:
                s = str(f)
            result.append(s)
        return result

    def _create_datainfo(self):
        dataset_path = self._read_dataset_path()
        (colnames, drop, replacements, _) = self._column_info()
        try:
            path = dataset_path.with_suffix('.datainfo')
        except:  # noqa: E722
            # FIXME: dataset_path could fail in so many ways!
            pass
        else:
            if path.is_file():
                di = DataInfo.read_json(path)
                di = di.derive(path=dataset_path)
                self.datainfo = di
                self._old_datainfo = di
                different_drop = []
                for colinfo, coldrop in zip(di, drop):
                    if colinfo.drop != coldrop:
                        colinfo.drop = coldrop
                        different_drop.append(colinfo.name)

                if different_drop:
                    warnings.warn(
                        "NONMEM .mod and dataset .datainfo disagree on "
                        f"DROP for columns {', '.join(different_drop)}."
                    )
                return

        column_info = []
        have_pk = self._get_pk_record()
        for colname, coldrop in zip(colnames, drop):
            if coldrop and colname not in ['DATE', 'DAT1', 'DAT2', 'DAT3']:
                info = ColumnInfo(colname, drop=coldrop, datatype='str')
            elif colname == 'ID' or colname == 'L1':
                info = ColumnInfo(
                    colname, drop=coldrop, datatype='int32', type='id', scale='nominal'
                )
            elif colname == 'DV' or colname == replacements.get('DV', None):
                info = ColumnInfo(colname, drop=coldrop, type='dv')
            elif colname == 'TIME' or colname == replacements.get('TIME', None):
                if not set(colnames).isdisjoint({'DATE', 'DAT1', 'DAT2', 'DAT3'}):
                    datatype = 'nmtran-time'
                else:
                    datatype = 'float64'
                info = ColumnInfo(
                    colname, drop=coldrop, type='idv', scale='ratio', datatype=datatype
                )
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
            else:
                info = ColumnInfo(colname, drop=coldrop)
            column_info.append(info)

        di = DataInfo(column_info, path=dataset_path)
        self.datainfo = di
        self._old_datainfo = di

    def _read_dataset(self, raw=False, parse_columns=tuple()):
        data_records = self.control_stream.get_records('DATA')
        ignore_character = data_records[0].ignore_character
        null_value = data_records[0].null_value
        (colnames, drop, replacements, _) = self._column_info()

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
                ignore = Model._replace_synonym_in_filters(ignore, replacements)
            else:
                accept = Model._replace_synonym_in_filters(accept, replacements)

        df = pharmpy.plugins.nonmem.dataset.read_nonmem_dataset(
            self.datainfo.path,
            raw,
            ignore_character,
            colnames,
            drop,
            null_value=null_value,
            parse_columns=parse_columns,
            ignore=ignore,
            accept=accept,
            dtype=None if raw else self.datainfo.get_dtype_dict(),
        )
        # Let TIME be the idv in both $PK and $PRED models

        # Remove individuals without observations
        try:
            from pharmpy.modeling.data import get_observations

            # This is a hack to be able to use the get_observations function
            # before the dataset has been properly read in.
            self._data_frame = df
            have_obs = set(get_observations(self).index.unique(level=0))
        except DatasetError:
            pass
        else:
            all_ids = set(df['ID'].unique())
            ids_to_remove = all_ids - have_obs
            df = df[~df['ID'].isin(ids_to_remove)]
        return df

    def rv_translation(self, reverse=False, remove_idempotent=False, as_symbols=False):
        self.random_variables
        d = dict()
        for record in self.control_stream.get_records('OMEGA'):
            for key, value in record.eta_map.items():
                nonmem_name = f'ETA({value})'
                d[nonmem_name] = key
        for record in self.control_stream.get_records('SIGMA'):
            for key, value in record.eta_map.items():
                nonmem_name = f'EPS({value})'
                d[nonmem_name] = key
        if remove_idempotent:
            d = {key: val for key, val in d.items() if key != val}
        if reverse:
            d = {val: key for key, val in d.items()}
        if as_symbols:
            d = {S(key): S(val) for key, val in d.items()}
        return d

    def parameter_translation(self, reverse=False, remove_idempotent=False, as_symbols=False):
        """Get a dict of NONMEM name to Pharmpy parameter name
        i.e. {'THETA(1)': 'TVCL', 'OMEGA(1,1)': 'IVCL'}
        """
        self.parameters
        d = dict()
        for theta_record in self.control_stream.get_records('THETA'):
            for key, value in theta_record.name_map.items():
                nonmem_name = f'THETA({value})'
                d[nonmem_name] = key
        for record in self.control_stream.get_records('OMEGA'):
            for key, value in record.name_map.items():
                nonmem_name = f'OMEGA({value[0]},{value[1]})'
                d[nonmem_name] = key
        for record in self.control_stream.get_records('SIGMA'):
            for key, value in record.name_map.items():
                nonmem_name = f'SIGMA({value[0]},{value[1]})'
                d[nonmem_name] = key
        if remove_idempotent:
            d = {key: val for key, val in d.items() if key != val}
        if reverse:
            d = {val: key for key, val in d.items()}
        if as_symbols:
            d = {S(key): S(val) for key, val in d.items()}
        return d

    def read_modelfit_results(self, path: Path):
        try:
            ext_path = path / (self.name + '.ext')
            self._modelfit_results = NONMEMChainedModelfitResults(ext_path, model=self)
            return self._modelfit_results
        except (FileNotFoundError, OSError):
            self._modelfit_results = None
            return None


def parse_estimation_steps(model):
    steps = []
    records = model.control_stream.get_records('ESTIMATION')
    covrec = model.control_stream.get_records('COVARIANCE')
    solver, tol, atol = parse_solver(model)

    # Read eta and epsilon derivatives
    etaderiv_names = None
    epsilonderivs_names = None
    table_records = model.control_stream.get_records('TABLE')
    for table in table_records:
        etaderivs = table.eta_derivatives
        if etaderivs:
            etas = model.random_variables.etas
            etaderiv_names = [etas[i - 1].name for i in etaderivs]
        epsderivs = table.epsilon_derivatives
        if epsderivs:
            epsilons = model.random_variables.epsilons
            epsilonderivs_names = [epsilons[i - 1].name for i in epsderivs]

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


def parse_solver(model):
    subs_records = model.control_stream.get_records('SUBROUTINES')
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
