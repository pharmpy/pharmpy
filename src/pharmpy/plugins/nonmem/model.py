# The NONMEM Model class

import warnings
from io import StringIO
from pathlib import Path

import pharmpy.model
import pharmpy.plugins.nonmem
import pharmpy.plugins.nonmem.dataset
from pharmpy.deps import sympy
from pharmpy.expressions import subs
from pharmpy.model import Assignment, DatasetError, NormalDistribution, Parameter, Parameters
from pharmpy.modeling.write_csv import write_csv

from .nmtran_parser import NMTranParser
from .parsing import (
    create_name_trans,
    parameter_translation,
    parse_column_info,
    parse_datainfo,
    parse_description,
    parse_estimation_steps,
    parse_initial_individual_estimates,
    parse_parameters,
    parse_random_variables,
    parse_statements,
    parse_value_type,
    replace_synonym_in_filters,
)
from .results import parse_modelfit_results
from .update import (
    abbr_translation,
    rv_translation,
    update_ccontra,
    update_description,
    update_estimation,
    update_initial_individual_estimates,
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
            self.filename_extension = '.ctl'
        else:
            self._name = path.stem
            self.filename_extension = path.suffix
        self.internals.old_name = self._name
        self.internals.control_stream = parser.parse(code)
        di = parse_datainfo(self.internals.control_stream, path)
        self.datainfo = di
        self._old_datainfo = di
        self._initial_individual_estimates_updated = False
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

        steps = parse_estimation_steps(self.internals.control_stream, self._random_variables)
        self._estimation_steps = steps
        self.internals._old_estimation_steps = steps

        if path is None:
            self._modelfit_results = None
        else:
            self._modelfit_results = parse_modelfit_results(self, path.parent)

        description = parse_description(self.internals.control_stream)
        self.description = description
        self.internals._old_description = description

        vt = parse_value_type(self.internals.control_stream, statements)
        self._value_type = vt

        self._statements = statements
        self.internals._old_statements = statements

        init_etas = parse_initial_individual_estimates(
            self.internals.control_stream, rvs, None if path is None else path.parent
        )
        self._initial_individual_estimates = init_etas
        self.internals._old_initial_individual_estimates = init_etas

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
        update_initial_individual_estimates(self, path, nofiles)
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
        rv_trans = rv_translation(
            self.internals.control_stream, reverse=True, remove_idempotent=True, as_symbols=True
        )
        trans.update(rv_trans)
        if pharmpy.plugins.nonmem.conf.write_etas_in_abbr:
            abbr = abbr_translation(self, rv_trans)
            trans.update(abbr)
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

    def validate(self):
        """Validates NONMEM model (records) syntactically."""
        self.internals.control_stream.validate()

    @property
    def model_code(self):
        self.update_source(nofiles=True)
        return str(self.internals.control_stream)

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

    def _read_dataset(self, raw=False, parse_columns=tuple()):
        data_records = self.internals.control_stream.get_records('DATA')
        ignore_character = data_records[0].ignore_character
        null_value = data_records[0].null_value
        (colnames, drop, replacements, _) = parse_column_info(self.internals.control_stream)

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
