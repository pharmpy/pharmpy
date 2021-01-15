# The NONMEM Model class
import re
import shutil
import warnings
from os import stat
from pathlib import Path

from sympy import Symbol

import pharmpy.data
import pharmpy.model
import pharmpy.plugins.nonmem
import pharmpy.symbols as symbols
from pharmpy.data import DatasetError
from pharmpy.model import ModelSyntaxError
from pharmpy.parameter import ParameterSet
from pharmpy.plugins.nonmem.results import NONMEMChainedModelfitResults
from pharmpy.plugins.nonmem.table import NONMEMTableFile, PhiTable
from pharmpy.random_variables import JointDistributionSeparate, RandomVariables, VariabilityLevel
from pharmpy.statements import Assignment, CompartmentalSystem, ODESystem

from .advan import compartmental_model
from .nmtran_parser import NMTranParser
from .records.factory import create_record
from .update import update_parameters, update_random_variables, update_statements


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
        self.dependent_variable_symbol = symbols.symbol('Y')
        self.individual_prediction_symbol = symbols.symbol('CIPREDI')

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

        super().update_source()

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
        """Get the ParameterSet of all parameters"""
        try:
            return self._parameters
        except AttributeError:
            pass

        self._read_parameters()
        if pharmpy.plugins.nonmem.conf.parameter_names == 'comment':
            self.statements
        return self._parameters

    def _read_parameters(self):
        next_theta = 1
        params = ParameterSet()
        for theta_record in self.control_stream.get_records('THETA'):
            thetas = theta_record.parameters(next_theta, seen_labels=set(params.names))
            params.update(thetas)
            next_theta += len(thetas)
        next_omega = 1
        previous_size = None
        for omega_record in self.control_stream.get_records('OMEGA'):
            omegas, next_omega, previous_size = omega_record.parameters(
                next_omega, previous_size, seen_labels=set(params.names)
            )
            params.update(omegas)
        next_sigma = 1
        previous_size = None
        for sigma_record in self.control_stream.get_records('SIGMA'):
            sigmas, next_sigma, previous_size = sigma_record.parameters(
                next_sigma, previous_size, seen_labels=set(params.names)
            )
            params.update(sigmas)
        self._parameters = params
        self._old_parameters = params.copy()

    @parameters.setter
    def parameters(self, params):
        """params can be a ParameterSet or a dict-like with name: value

        Current restrictions:
         * ParameterSet only supported for new initial estimates and fix
         * Only set the exact same parameters. No additions and no removing of parameters
        """
        if isinstance(params, ParameterSet):
            inits = params.inits
        else:
            inits = params
        if not self.random_variables.validate_parameters(inits):
            raise ValueError("New parameter inits are not valid")

        self.parameters
        if not isinstance(params, ParameterSet):
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
                statements.append(Assignment('F', symbols.symbol('F')))
            statements += error.statements

        if pharmpy.plugins.nonmem.conf.parameter_names == 'comment':
            if not hasattr(self, '_parameters'):
                self._read_parameters()
            trans = self.parameter_translation(remove_idempotent=True, as_symbols=True)
            parameter_symbols = {symb for _, symb in trans.items()}
            clashing_symbols = parameter_symbols & statements.free_symbols
            if clashing_symbols:
                warnings.warn(
                    f'The parameter names {clashing_symbols} are also names of variables '
                    f'in the model code. Falling back to the NONMEM default parameter '
                    f'names for these.'
                )
                rev_trans = {val: key for key, val in trans.items()}
                trans = {
                    nm_symb: symb for nm_symb, symb in trans.items() if symb not in clashing_symbols
                }
                for symb in clashing_symbols:
                    self.parameters[symb.name].name = rev_trans[symb].name
            statements.subs(trans)

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
    def error_model(self):
        error = self._get_error_record()
        y_args = error.statements.find_assignment('Y').expression.args

        if len(y_args) == 0:
            return 'NONE'
        elif len(y_args) == 2:
            if isinstance(y_args[0], Symbol) and isinstance(y_args[1], Symbol):
                return 'ADD'
            else:
                return 'PROP'
        else:
            return 'ADD_PROP'

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
            rvs.update(etas)
        self.adjust_iovs(rvs)
        next_sigma = 1
        prev_cov = None
        for sigma_record in self.control_stream.get_records('SIGMA'):
            epsilons, next_sigma, prev_cov, _ = sigma_record.random_variables(next_sigma, prev_cov)
            rvs.update(epsilons)
        self._random_variables = rvs
        self._old_random_variables = rvs.copy()
        return rvs

    @staticmethod
    def adjust_iovs(rvs):
        for i, rv in enumerate(rvs):
            try:
                next_rv = rvs[i + 1]
            except KeyError:
                break

            if (
                rv.variability_level != VariabilityLevel.IOV
                and next_rv.variability_level == VariabilityLevel.IOV
            ):
                if isinstance(rv, JointDistributionSeparate):
                    associated_rvs = rvs.get_rvs_from_same_dist(rv)
                    for rv_con in associated_rvs:
                        rv_con.variability_level = VariabilityLevel.IOV
                else:
                    rv.variability_level = VariabilityLevel.IOV

    @random_variables.setter
    def random_variables(self, new):
        self._random_variables = new

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
        if 'TIME' in df.columns:
            df.pharmpy.column_type['TIME'] = pharmpy.data.ColumnType.IDV
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
            d = {symbols.symbol(key): symbols.symbol(val) for key, val in d.items()}
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
            d = {symbols.symbol(key): symbols.symbol(val) for key, val in d.items()}
        return d
