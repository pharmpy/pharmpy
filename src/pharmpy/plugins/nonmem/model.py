# The NONMEM Model class
import copy
import re
import shutil
import warnings
from os import stat
from pathlib import Path

import pharmpy.data
import pharmpy.model
import pharmpy.plugins.nonmem
from pharmpy.data import DatasetError
from pharmpy.estimation import EstimationMethod, list_supported_est
from pharmpy.model import ModelSyntaxError
from pharmpy.parameter import Parameters
from pharmpy.plugins.nonmem.results import NONMEMChainedModelfitResults
from pharmpy.plugins.nonmem.table import NONMEMTableFile, PhiTable
from pharmpy.random_variables import RandomVariables
from pharmpy.statements import Assignment, CompartmentalSystem, ODESystem
from pharmpy.symbols import symbol as S

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
    is_control_stream = re.search(r'^\s*\$PRO', src.code, re.MULTILINE)
    if is_control_stream:
        return Model
    else:
        return None


class Model(pharmpy.model.Model):
    def __init__(self, src, **kwargs):
        super().__init__()
        parser = NMTranParser()
        self.source = src
        if not self.source.filename_extension:
            self.source.filename_extension = '.ctl'
        self._name = self.source.path.stem
        self.control_stream = parser.parse(src.code)
        self._initial_individual_estimates_updated = False
        self._updated_etas_file = None
        self._dataset_updated = False
        self._modelfit_results = None
        self.dependent_variable = S('Y')
        self.individual_prediction_symbol = S('CIPREDI')
        self.observation_transformation = self.dependent_variable
        self._old_observation_transformation = self.dependent_variable

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
    def modelfit_results(self):
        if self._modelfit_results is None:
            if self.source.path.is_file():
                ext_path = self.source.path.with_suffix('.ext')
                if ext_path.exists() and stat(ext_path).st_size > 0:
                    self._modelfit_results = NONMEMChainedModelfitResults(ext_path, model=self)
                    return self._modelfit_results
            else:
                return None
        else:
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
            update_random_variables(self, self._old_random_variables, self._random_variables)
            self._old_random_variables = self._random_variables.copy()
        if hasattr(self, '_parameters'):
            update_parameters(self, self._old_parameters, self._parameters)
            self._old_parameters = self._parameters.copy()
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
            self._old_statements = self._statements.copy()

        if self._dataset_updated:
            # FIXME: If no name set use the model name. Set that when setting dataset to input!
            if not nofiles:
                if path is not None:
                    dir_path = path.parent
                else:
                    dir_path = None
                datapath = self.dataset.pharmpy.write_csv(path=dir_path, force=force)
                self.dataset_path = datapath.name

            data_record = self.control_stream.get_records('DATA')[0]

            label = self.dataset.columns[0]
            data_record.ignore_character_from_header(label)
            self._update_input(self.dataset.columns)

            # Remove IGNORE/ACCEPT. Could do diff between old dataset and find simple
            # IGNOREs to add i.e. for filter out certain ID.
            del data_record.ignore
            del data_record.accept
            self._dataset_updated = False

        self._update_sizes()
        update_estimation(self)

        if self.observation_transformation != self._old_observation_transformation:
            if not nofiles:
                update_ccontra(self, path, force)

        super().update_source()

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
            self._old_parameters = self._parameters.copy()
        return self._parameters

    def _read_parameters(self):
        next_theta = 1
        params = Parameters()
        for theta_record in self.control_stream.get_records('THETA'):
            thetas = theta_record.parameters(next_theta, seen_labels=set(params.names))
            params.extend(thetas)
            next_theta += len(thetas)
        next_omega = 1
        previous_size = None
        for omega_record in self.control_stream.get_records('OMEGA'):
            omegas, next_omega, previous_size = omega_record.parameters(
                next_omega, previous_size, seen_labels=set(params.names)
            )
            params.extend(omegas)
        next_sigma = 1
        previous_size = None
        for sigma_record in self.control_stream.get_records('SIGMA'):
            sigmas, next_sigma, previous_size = sigma_record.parameters(
                next_sigma, previous_size, seen_labels=set(params.names)
            )
            params.extend(sigmas)
        self._parameters = params
        self._old_parameters = params.copy()

    @parameters.setter
    def parameters(self, params):
        """params can be a Parameters or a dict-like with name: value

        Current restrictions:
         * Parameters only supported for new initial estimates and fix
         * Only set the exact same parameters. No additions and no removing of parameters
        """
        if isinstance(params, Parameters):
            inits = params.inits
        else:
            inits = params
        if not self.random_variables.validate_parameters(inits):
            raise ValueError("New parameter inits are not valid")

        self.parameters
        if not isinstance(params, Parameters):
            self._parameters.inits = params
        else:
            self._parameters = params

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
                source_dir = self.source.path.parent
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

    def update_individual_estimates(self, source):
        """Update initial individual estimates from another model"""
        self._initial_individual_estimates_updated = False
        try:
            del self._initial_individual_estimates
        except AttributeError:
            pass
        self._updated_etas_file = self.source.path.with_suffix('.phi')

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
            advan = sub.get_option_startswith('ADVAN')
            trans = sub.get_option_startswith('TRANS')
            if not trans:
                trans = 'TRANS1'
            comp = compartmental_model(self, advan, trans, des)
            if comp is not None:
                cm, link = comp
                statements += [cm, link]
            else:
                statements.append(ODESystem())  # FIXME: Placeholder for ODE-system
                # FIXME: Dummy link statement
                statements.append(Assignment('F', S('F')))
            statements += error.statements

        if not hasattr(self, '_parameters'):
            self._read_parameters()

        trans_statements, trans_params = self._create_name_trans(statements)
        for key, value in trans_params.items():
            try:
                self.parameters[key].name = value
                for theta in self.control_stream.get_records('THETA'):
                    theta.update_name_map(trans_params)
                for omega in self.control_stream.get_records('OMEGA'):
                    omega.update_name_map(trans_params)
                for sigma in self.control_stream.get_records('SIGMA'):
                    sigma.update_name_map(trans_params)
            except KeyError:
                self.random_variables.subs({S(key): value})

        statements.subs(trans_statements)

        self._statements = statements
        self._old_statements = statements.copy()
        return statements

    @statements.setter
    def statements(self, statements_new):
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
        self._random_variables = new

    @property
    def estimation_steps(self):
        try:
            return self._estimation_steps
        except AttributeError:
            pass

        steps = []
        records = self.control_stream.get_records('ESTIMATION')
        covrec = self.control_stream.get_records('COVARIANCE')
        for record in records:
            value = record.get_option('METHOD')
            if value is None or value == '0' or value == 'ZERO':
                name = 'fo'
            elif value == '1' or value == 'CONDITIONAL' or value == 'COND':
                name = 'foce'
            else:
                if value in list_supported_est():
                    name = value
                else:
                    raise ModelSyntaxError(
                        f'Non-recognized estimation method in: {str(record.root)}'
                    )
            if record.has_option('INTERACTION') or record.has_option('INTER'):
                interaction = True
            else:
                interaction = False
            if covrec:
                cov = True
            else:
                cov = False

            options_stored = ['METHOD', 'METH', 'INTERACTION', 'INTER']
            options_left = [
                option for option in record.all_options if option.key not in options_stored
            ]

            meth = EstimationMethod(name, interaction=interaction, cov=cov, options=options_left)
            steps.append(meth)
        self._estimation_steps = steps
        self._old_estimation_steps = copy.deepcopy(steps)
        return steps

    def add_estimation_step(self, name, interaction, cov, options=[], idx=None):
        meth = EstimationMethod(name, interaction=interaction, cov=cov, options=options)
        if isinstance(idx, int):
            self.estimation_steps.insert(idx, meth)
        else:
            self.estimation_steps.append(meth)

    def remove_estimation_step(self, idx):
        del self.estimation_steps[idx]

    def __str__(self):
        return str(self.control_stream)

    @property
    def dataset_path(self):
        """This property is NONMEM specific"""
        record = self.control_stream.get_records('DATA')[0]
        path = Path(record.filename)
        if not path.is_absolute():
            path = self.source.path.parent / path  # Relative model source file.
        try:
            return path.resolve()
        except FileNotFoundError:
            return path

    @dataset_path.setter
    def dataset_path(self, path):
        path = Path(path)
        assert (
            not path.exists() or path.is_file()
        ), 'input path change, but non-file exists at ' 'target (%s)' % str(path)
        record = self.control_stream.get_records('DATA')[0]
        record.filename = str(path)

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
        return tuple of three lists, colnames, coltypes and drop together with a dictionary
        of replacements for reserved names (aka synonyms).
        Anonymous columns, i.e. DROP or SKIP alone, will be given unique names _DROP1, ...
        """
        input_records = self.control_stream.get_records("INPUT")
        colnames = []
        coltypes = []
        drop = []
        synonym_replacement = {}
        next_anonymous = 1
        for record in input_records:
            for key, value in record.all_options:
                if value:
                    if key == 'DROP' or key == 'SKIP':
                        drop.append(True)
                        colnames.append(value)
                        reserved_name = value
                    elif value == 'DROP' or value == 'SKIP':
                        colnames.append(key)
                        drop.append(True)
                        reserved_name = key
                    else:
                        drop.append(False)
                        (reserved_name, synonym) = Model._synonym(key, value)
                        synonym_replacement[reserved_name] = synonym
                        colnames.append(synonym)
                else:
                    if key == 'DROP' or key == 'SKIP':
                        drop.append(True)
                        name = f'_DROP{next_anonymous}'
                        colnames.append(name)
                        reserved_name = name
                        next_anonymous += 1
                    else:
                        drop.append(False)
                        colnames.append(key)
                        reserved_name = key
                coltypes.append(pharmpy.data.read.infer_column_type(reserved_name))
        return colnames, coltypes, drop, synonym_replacement

    def _update_input(self, new_names):
        """Update $INPUT with new column names

        currently supporting append columns at end and removing columns
        """
        colnames, _, _, _ = self._column_info()
        removed_columns = set(colnames) - set(new_names)
        input_records = self.control_stream.get_records("INPUT")
        for col in removed_columns:
            input_records[0].remove_option(col)
        appended_names = new_names[len(colnames) :]
        last_input_record = input_records[-1]
        for colname in appended_names:
            last_input_record.append_option(colname)

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

    def _read_dataset(self, raw=False, parse_columns=tuple()):
        data_records = self.control_stream.get_records('DATA')
        ignore_character = data_records[0].ignore_character
        null_value = data_records[0].null_value
        (colnames, coltypes, drop, replacements) = self._column_info()

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

        df = pharmpy.data.read_nonmem_dataset(
            self.dataset_path,
            raw,
            ignore_character,
            colnames,
            coltypes,
            drop,
            null_value=null_value,
            parse_columns=parse_columns,
            ignore=ignore,
            accept=accept,
        )
        if self._get_pk_record():
            if 'EVID' in df.columns:
                if 'MDV' in df.columns:
                    df.pharmpy.column_type['MDV'] = pharmpy.data.ColumnType.UNKNOWN
                df.pharmpy.column_type['EVID'] = pharmpy.data.ColumnType.EVENT
            if 'AMT' in df.columns:
                df.pharmpy.column_type['AMT'] = pharmpy.data.ColumnType.DOSE
        # Let TIME be the idv in both $PK and $PRED models
        idvcol = replacements.get('TIME', 'TIME')
        if idvcol in df.columns:
            df.pharmpy.column_type[idvcol] = pharmpy.data.ColumnType.IDV
        df.name = self.dataset_path.stem
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
