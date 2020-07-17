# The NONMEM Model class
import re
import shutil
from pathlib import Path

import pharmpy.data
import pharmpy.model
from pharmpy.data import DatasetError
from pharmpy.parameter import ParameterSet
from pharmpy.plugins.nonmem.results import NONMEMChainedModelfitResults
from pharmpy.plugins.nonmem.table import NONMEMTableFile, PhiTable
from pharmpy.random_variables import RandomVariables

from .nmtran_parser import NMTranParser


class Model(pharmpy.model.Model):
    def __init__(self, src, **kwargs):
        super().__init__()
        parser = NMTranParser()
        self.source = src
        if not self.source.filename_extension:
            self.source.filename_extension = '.ctl'
        self.name = self.source.path.stem
        self.control_stream = parser.parse(src.code)
        self._parameters_updated = False
        self._initial_individual_estimates_updated = False
        self._updated_etas_file = None
        self._dataset_updated = False

    @property
    def modelfit_results(self):
        try:
            return self._modelfit_results
        except AttributeError:
            None
        if self.source.path.is_file():
            lst_path = self.source.path.with_suffix('.lst')
            if lst_path.exists():
                num_est = len(self.control_stream.get_records('ESTIMATION'))
                self._modelfit_results = NONMEMChainedModelfitResults(lst_path, num_est, model=self)
                return self._modelfit_results
        else:
            return None

    @staticmethod
    def detect(src, *args, **kwargs):
        """ Check if src represents a NONMEM control stream
        i.e. check if it is a file that contain $PRO
        """
        return bool(re.search(r'^\$PRO', src.code, re.MULTILINE))

    def update_source(self, path=None, force=False):
        """ Update the source

            path - path to modelfile
        """
        if self._dataset_updated:
            # FIXME: If no name set use the model name. Set that when setting dataset to input!
            datapath = self.dataset.pharmpy.write_csv(force=force)
            self.path = datapath

            data_record = self.control_stream.get_records('DATA')[0]

            label = self.dataset.columns[0]
            data_record.ignore_character_from_header(label)
            self._update_input(self.input.dataset.columns)

            # Remove IGNORE/ACCEPT. Could do diff between old dataset and find simple
            # IGNOREs to add i.e. for filter out certain ID.
            del(data_record.ignore)
            del(data_record.accept)
            self._dataset_updated = False

        self._update_parameters()
        self._update_initial_individual_etimates(path)

        super().update_source()

    def _update_parameters(self):
        """ Update parameters
        """
        if not self._parameters_updated:
            return

        params = self.parameters
        next_theta = 1
        for theta_record in self.control_stream.get_records('THETA'):
            theta_record.update(params, next_theta)
            next_theta += len(theta_record)
        next_omega = 1
        previous_size = None
        for omega_record in self.control_stream.get_records('OMEGA'):
            next_omega, previous_size = omega_record.update(params, next_omega, previous_size)
        next_sigma = 1
        previous_size = None
        for sigma_record in self.control_stream.get_records('SIGMA'):
            next_sigma, previous_size = sigma_record.update(params, next_sigma, previous_size)

        self._parameters_updated = False

    def _update_initial_individual_etimates(self, path):
        """ Update $ETAS

            Could have 0 FIX in model. Need to readd these
        """
        if path is None:        # What to do here?
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
        """Get the ParameterSet of all parameters
        """
        try:
            return self._parameters
        except AttributeError:
            pass

        next_theta = 1
        params = ParameterSet()
        for theta_record in self.control_stream.get_records('THETA'):
            thetas = theta_record.parameters(next_theta)
            params.update(thetas)
            next_theta += len(thetas)
        next_omega = 1
        previous_size = None
        for omega_record in self.control_stream.get_records('OMEGA'):
            omegas, next_omega, previous_size = omega_record.parameters(next_omega, previous_size)
            params.update(omegas)
        next_sigma = 1
        previous_size = None
        for sigma_record in self.control_stream.get_records('SIGMA'):
            sigmas, next_sigma, previous_size = sigma_record.parameters(next_sigma, previous_size)
            params.update(sigmas)
        self._parameters = params
        return params

    @parameters.setter
    def parameters(self, params):
        """params can be a ParameterSet or a dict-like with name: value

           Current restricions:
            * ParameterSet only supported for new initial estimates and fix
            * Only set the exact same parameters. No additions and no removing of parameters
        """
        if isinstance(params, ParameterSet):
            inits = params.inits
        else:
            inits = params
        if not self.random_variables.validate_parameters(inits):
            raise ValueError("New parameter inits are not valid")

        parameters = self.parameters
        if not isinstance(params, ParameterSet):
            parameters.inits = params
        else:
            if len(self.parameters) != len(params):
                raise NotImplementedError("Current number of parameters in model and parameters to "
                                          "set are not the same. Not yet supported")

            for p in self.parameters:
                name = p.name
                if name not in params:
                    raise NotImplementedError(f"Parameter '{name}' not found in input. Currently "
                                              f"not supported")
            self._parameters = params
        self._parameters_updated = True

    @property
    def initial_individual_estimates(self):
        '''Initial individual estimates

           These are taken from the $ETAS FILE. 0 FIX ETAs are removed.
           If no $ETAS is present None will be returned.

           Setter assumes that all IDs are present
        '''
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
            raise ValueError(f'Cannot set initial estimate for random variable not in the model:'
                             f' {rv_names - columns}')
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
        """Update initial individual estimates from another model
        """
        self._initial_individual_estimates_updated = False
        try:
            del self._initial_individual_estimates
        except AttributeError:
            pass
        self._updated_etas_file = self.source.path.with_suffix('.phi')

    def _sort_eta_columns(self, df):
        colnames = df.columns

        def keyfunc(name):
            m = re.match(r'ETA\((\d+)\)', name)
            return int(m.group(1))
        sorted_colnames = colnames.sort(key=keyfunc)
        return df.reindex(sorted_colnames, axis=1)

    @property
    def statements(self):
        pred = self.control_stream.get_records('PRED')
        if len(pred) == 0:
            pk = self.control_stream.get_records('PK')
            rec = pk[0]
        else:
            rec = pred[0]
        return rec.statements

    @statements.setter
    def statements(self, statements_new):
        pred = self.control_stream.get_records('PRED')
        if len(pred) == 0:
            pk = self.control_stream.get_records('PK')
            rec = pk[0]
        else:
            rec = pred[0]

        rec.statements = statements_new

    def _zero_fix_rvs(self, eta=True):
        zero_fix = []
        if eta:
            prev_cov = None
            next_omega = 1
            for omega_record in self.control_stream.get_records('OMEGA'):
                _, next_omega, prev_cov, new_zero_fix = \
                        omega_record.random_variables(next_omega, prev_cov)
                zero_fix += new_zero_fix
        else:
            prev_cov = None
            next_sigma = 1
            for sigma_record in self.control_stream.get_records('SIGMA'):
                _, next_sigma, prev_cov, new_zero_fix = \
                        sigma_record.random_variables(next_sigma, prev_cov)
                zero_fix += new_zero_fix
        return zero_fix

    @property
    def random_variables(self):
        rvs = RandomVariables()
        next_omega = 1
        prev_cov = None
        for omega_record in self.control_stream.get_records('OMEGA'):
            etas, next_omega, prev_cov, _ = omega_record.random_variables(next_omega, prev_cov)
            rvs.update(etas)
        next_sigma = 1
        prev_cov = None
        for sigma_record in self.control_stream.get_records('SIGMA'):
            epsilons, next_sigma, prev_cov, _ = sigma_record.random_variables(next_sigma, prev_cov)
            rvs.update(epsilons)
        return rvs

    def __str__(self):
        return str(self.control_stream)

    @property
    def dataset_path(self):
        """This property is NONMEM specific
        """
        record = self.control_stream.get_records('DATA')[0]
        path = Path(record.filename)
        if not path.is_absolute():
            path = self.source.path.parent / path     # Relative model source file.
        try:
            return path.resolve()
        except FileNotFoundError:
            return path

    @dataset_path.setter
    def dataset_path(self, path):
        path = Path(path)
        assert not path.exists() or path.is_file(), ('input path change, but non-file exists at '
                                                     'target (%s)' % str(path))
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
        """Return a tuple reserved name and synonym
        """
        _reserved_column_names = [
            'ID', 'L1', 'L2', 'DV', 'MDV', 'RAW_', 'MRG_', 'RPT_',
            'TIME', 'DATE', 'DAT1', 'DAT2', 'DAT3', 'EVID', 'AMT', 'RATE', 'SS', 'II', 'ADDL',
            'CMT', 'PCMT', 'CALL', 'CONT'
        ]
        if key in _reserved_column_names:
            return (key, value)
        elif value in _reserved_column_names:
            return (value, key)
        else:
            raise DatasetError(f'A column name "{key}" in $INPUT has a synonym to a non-reserved '
                               f'column name "{value}"')

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

           currently supporting append columns at end
        """
        colnames, _, _, _ = self._column_info()
        appended_names = new_names[len(colnames):]
        input_records = self.control_stream.get_records("INPUT")
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

        df = pharmpy.data.read_nonmem_dataset(self.dataset_path, raw, ignore_character, colnames,
                                              coltypes, drop, null_value=null_value,
                                              parse_columns=parse_columns, ignore=ignore,
                                              accept=accept)
        df.name = self.dataset_path.stem
        return df
