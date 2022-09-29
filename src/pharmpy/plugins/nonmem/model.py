# The NONMEM Model class

import shutil
import warnings
from io import StringIO
from pathlib import Path

import pharmpy.model
import pharmpy.plugins.nonmem
import pharmpy.plugins.nonmem.dataset
from pharmpy.deps import sympy
from pharmpy.expressions import subs
from pharmpy.model import (
    Assignment,
    ColumnInfo,
    DataInfo,
    DatasetError,
    NormalDistribution,
    Parameter,
    Parameters,
)
from pharmpy.modeling.write_csv import write_csv
from pharmpy.plugins.nonmem.results import NONMEMChainedModelfitResults
from pharmpy.plugins.nonmem.table import NONMEMTableFile, PhiTable
from pharmpy.workflows import NullModelDatabase, default_model_database

from .nmtran_parser import NMTranParser
from .parsing import (
    create_name_trans,
    parameter_translation,
    parse_description,
    parse_estimation_steps,
    parse_parameters,
    parse_random_variables,
    parse_statements,
    parse_value_type,
)
from .update import (
    update_abbr_record,
    update_ccontra,
    update_description,
    update_estimation,
    update_input,
    update_name_of_tables,
    update_parameters,
    update_random_variables,
    update_sizes,
    update_statements,
)


class NONMEMModelInternals:
    pass


def convert_model(model):
    """Convert any model into a NONMEM model"""
    if isinstance(model, Model):
        return model.copy()
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
    nm_model.internals._old_parameters = Parameters()
    nm_model.statements = model.statements
    if hasattr(model, 'name'):
        nm_model.name = model.name
    # FIXME: No handling of other DVs
    nm_model.dependent_variable = sympy.Symbol('Y')
    nm_model.value_type = model.value_type
    nm_model._data_frame = model.dataset
    nm_model._estimation_steps = model.estimation_steps
    nm_model._initial_individual_estimates = model.initial_individual_estimates
    nm_model.observation_transformation = subs(
        model.observation_transformation,
        {model.dependent_variable: nm_model.dependent_variable},
        simultaneous=True,
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
        self.internals = NONMEMModelInternals()
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
        self.internals.old_name = self._name
        self.internals.control_stream = parser.parse(code)
        self._create_datainfo()
        self._initial_individual_estimates_updated = False
        self._updated_etas_file = None
        self._dataset_updated = False
        self._parent_model = None

        dv = sympy.Symbol('Y')
        self.dependent_variable = dv
        self.observation_transformation = dv
        self.internals._old_observation_transformation = dv

        parameters = parse_parameters(self.internals.control_stream)

        statements = parse_statements(self)

        rvs = parse_random_variables(self.internals.control_stream)

        trans_statements, trans_params = create_name_trans(
            self.internals.control_stream, rvs, statements
        )
        for theta in self.internals.control_stream.get_records('THETA'):
            theta.update_name_map(trans_params)
        for omega in self.internals.control_stream.get_records('OMEGA'):
            omega.update_name_map(trans_params)
        for sigma in self.internals.control_stream.get_records('SIGMA'):
            sigma.update_name_map(trans_params)

        d_par = dict()
        d_rv = dict()
        for key, value in trans_params.items():
            if key in parameters:
                d_par[key] = value
            else:
                d_rv[sympy.Symbol(key)] = value

        self.internals._old_random_variables = rvs  # FIXME: This has to stay here
        rvs = rvs.subs(d_rv)

        new = []
        for p in parameters:
            if p.name in d_par:
                newparam = Parameter(
                    name=d_par[p.name], init=p.init, lower=p.lower, upper=p.upper, fix=p.fix
                )
            else:
                newparam = p
            new.append(newparam)
        parameters = Parameters(new)

        statements = statements.subs(trans_statements)

        if not rvs.validate_parameters(parameters.inits):
            nearest = rvs.nearest_valid_parameters(parameters.inits)
            before, after = self._compare_before_after_params(parameters.inits, nearest)
            warnings.warn(
                f"Adjusting initial estimates to create positive semidefinite "
                f"omega/sigma matrices.\nBefore adjusting:  {before}.\n"
                f"After adjusting: {after}"
            )
            parameters = parameters.set_initial_estimates(nearest)

        self._random_variables = rvs

        self._parameters = parameters
        self.internals._old_parameters = parameters

        if path is None:
            self._modelfit_results = None
        else:
            self.read_modelfit_results(path.parent)

        description = parse_description(self.internals.control_stream)
        self.description = description
        self.internals._old_description = description

        steps = parse_estimation_steps(self.internals.control_stream, self._random_variables)
        self._estimation_steps = steps
        self.internals._old_estimation_steps = steps

        vt = parse_value_type(self.internals.control_stream, statements)
        self._value_type = vt

        self._statements = statements
        self.internals._old_statements = statements

        init_etas = parse_initial_individual_estimates(
            self.internals.control_stream, rvs, None if path is None else path.parent
        )
        self._initial_individual_estimates = init_etas

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
                eta = NormalDistribution.create('eta_dummy', 'iiv', 0, omega.symbol)
                statement = Assignment(sympy.Symbol('DUMMYETA'), sympy.Symbol(eta.names[0]))
                self.statements = statement + self.statements
                self.random_variables = self.random_variables + eta
                self.parameters = Parameters([p for p in self.parameters] + [omega])
            update_random_variables(
                self, self.internals._old_random_variables, self._random_variables
            )
            self.internals._old_random_variables = self._random_variables
        if hasattr(self, '_parameters'):
            update_parameters(self, self.internals._old_parameters, self._parameters)
            self.internals._old_parameters = self._parameters
        trans = parameter_translation(
            self.internals.control_stream, reverse=True, remove_idempotent=True, as_symbols=True
        )
        rv_trans = self.rv_translation(reverse=True, remove_idempotent=True, as_symbols=True)
        trans.update(rv_trans)
        if pharmpy.plugins.nonmem.conf.write_etas_in_abbr:
            abbr_trans = self._abbr_translation(rv_trans)
            trans.update(abbr_trans)
        if hasattr(self, '_statements'):
            update_statements(self, self.internals._old_statements, self._statements, trans)
            self.internals._old_statements = self._statements

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
            data_record = self.internals.control_stream.get_records('DATA')[0]

            label = self.datainfo.names[0]
            data_record.ignore_character_from_header(label)
            update_input(self)

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
                data_record = self.internals.control_stream.get_records('DATA')[0]
                data_record.filename = str(path)

        update_sizes(self)
        update_estimation(self)

        if self.observation_transformation != self.internals._old_observation_transformation:
            if not nofiles:
                update_ccontra(self, path, force)
        update_description(self)

        if self._name != self.internals.old_name:
            update_name_of_tables(self.internals.control_stream, self._name)

    def _abbr_translation(self, rv_trans):
        abbr_pharmpy = self.internals.control_stream.abbreviated.translate_to_pharmpy_names()
        abbr_replace = self.internals.control_stream.abbreviated.replace
        abbr_trans = update_abbr_record(self, rv_trans)
        abbr_recs = {
            sympy.Symbol(abbr_pharmpy[value]): sympy.Symbol(key)
            for key, value in abbr_replace.items()
            if value in abbr_pharmpy.keys()
        }
        abbr_trans.update(abbr_recs)
        return abbr_trans

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
            eta_records = self.internals.control_stream.get_records('ETAS')
            if eta_records:
                record = eta_records[0]
            else:
                record = self.internals.control_stream.append_record('$ETAS ')
            record.path = phi_path
        elif self._updated_etas_file:
            eta_records = self.internals.control_stream.get_records('ETAS')
            if eta_records:
                record = eta_records[0]
            else:
                record = self.internals.control_stream.append_record('$ETAS')
            shutil.copy(self._updated_etas_file, phi_path)
            record.path = phi_path

        if self._initial_individual_estimates_updated or self._updated_etas_file:
            first_est_record = self.internals.control_stream.get_records('ESTIMATION')[0]
            try:
                first_est_record.option_pairs['MCETA']
            except KeyError:
                first_est_record.set_option('MCETA', 1)
            self._updated_etas_file = None
            self._initial_individual_estimates_updated = False

    def validate(self):
        """Validates NONMEM model (records) syntactically."""
        self.internals.control_stream.validate()

    @property
    def initial_individual_estimates(self):
        return self._initial_individual_estimates

    @initial_individual_estimates.setter
    def initial_individual_estimates(self, estimates):
        rv_names = {rv for rv in self.random_variables.names if rv.startswith('ETA')}
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
            for omega_record in self.internals.control_stream.get_records('OMEGA'):
                _, next_omega, prev_cov, new_zero_fix = omega_record.random_variables(
                    next_omega, prev_cov
                )
                zero_fix += new_zero_fix
        else:
            prev_cov = None
            next_sigma = 1
            for sigma_record in self.internals.control_stream.get_records('SIGMA'):
                _, next_sigma, prev_cov, new_zero_fix = sigma_record.random_variables(
                    next_sigma, prev_cov
                )
                zero_fix += new_zero_fix
        return zero_fix

    @property
    def model_code(self):
        self.update_source(nofiles=True)
        return str(self.internals.control_stream)

    def _read_dataset_path(self):
        record = next(iter(self.internals.control_stream.get_records('DATA')), None)
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
        input_records = self.internals.control_stream.get_records("INPUT")
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
        have_pk = self.internals.control_stream.get_pk_record()
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
            elif colname == 'RATE' and have_pk:
                info = ColumnInfo(colname, drop=coldrop, type='rate')
            else:
                info = ColumnInfo(colname, drop=coldrop)
            column_info.append(info)

        di = DataInfo(column_info, path=dataset_path)
        self.datainfo = di
        self._old_datainfo = di

    def _read_dataset(self, raw=False, parse_columns=tuple()):
        data_records = self.internals.control_stream.get_records('DATA')
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
        d = dict()
        for record in self.internals.control_stream.get_records('OMEGA'):
            for key, value in record.eta_map.items():
                nonmem_name = f'ETA({value})'
                d[nonmem_name] = key
        for record in self.internals.control_stream.get_records('SIGMA'):
            for key, value in record.eta_map.items():
                nonmem_name = f'EPS({value})'
                d[nonmem_name] = key
        if remove_idempotent:
            d = {key: val for key, val in d.items() if key != val}
        if reverse:
            d = {val: key for key, val in d.items()}
        if as_symbols:
            d = {sympy.Symbol(key): sympy.Symbol(val) for key, val in d.items()}
        return d

    def read_modelfit_results(self, path: Path):
        try:
            ext_path = path / (self.name + '.ext')
            self._modelfit_results = NONMEMChainedModelfitResults(ext_path, model=self)
            return self._modelfit_results
        except (FileNotFoundError, OSError):
            self._modelfit_results = None
            return None


def parse_initial_individual_estimates(control_stream, rvs, basepath):
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
            path = basepath / path
            path = path.resolve()
        phi_tables = NONMEMTableFile(path)
        rv_names = [rv for rv in rvs.names if rv.startswith('ETA')]
        phitab = next(phi_tables)
        names = [name for name in rv_names if name in phitab.etas.columns]
        etas = phitab.etas[names]
    else:
        etas = None
    return etas
