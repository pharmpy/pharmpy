# The NONMEM Model class
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.internals.expr.subs import subs
from pharmpy.internals.fs.path import path_relative_to
from pharmpy.model import Assignment, DataInfo, EstimationSteps
from pharmpy.model import Model as BaseModel
from pharmpy.model import NormalDistribution, Parameter, Parameters, RandomVariables, Statements
from pharmpy.model.model import compare_before_after_params, update_datainfo
from pharmpy.modeling.write_csv import create_dataset_path, write_csv

from .config import conf
from .nmtran_parser import NMTranControlStream, NMTranParser
from .parameters import parameter_translation
from .parsing import (
    create_name_trans,
    parse_datainfo,
    parse_dataset,
    parse_description,
    parse_estimation_steps,
    parse_initial_individual_estimates,
    parse_parameters,
    parse_random_variables,
    parse_statements,
    parse_value_type,
)
from .random_variables import rv_translation
from .results import _parse_modelfit_results
from .update import (
    abbr_translation,
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


@dataclass
class NONMEMModelInternals:
    control_stream: NMTranControlStream
    old_name: str
    _old_description: str
    _old_estimation_steps: EstimationSteps
    _old_observation_transformation: sympy.Symbol
    _old_parameters: Parameters
    _old_random_variables: RandomVariables
    _old_statements: Statements
    _old_initial_individual_estimates: Optional[pd.DataFrame]
    _old_datainfo: DataInfo
    _dataset_updated: bool
    _compartment_map: Optional[Dict[str, int]]


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
        code += '$ERROR\nA=B\n'
    nm_model = parse_code(code, dataset=model.dataset)
    assert isinstance(nm_model, Model)
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
    nm_model._dataset = model.dataset
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
        nm_model.internals._compartment_map = {
            name: i for i, name in enumerate(model.statements.ode_system.compartment_names, start=1)
        }
    return nm_model


def _dataset(control_stream: NMTranControlStream, di: DataInfo, df: Optional[pd.DataFrame]):
    # FIXME Use lambda: model.dataset instead
    if df is None:
        return lambda: parse_dataset(di, control_stream, raw=False)
    else:
        return lambda: df


class Model(BaseModel):
    def __init__(
        self,
        internals: NONMEMModelInternals,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        assert isinstance(internals, NONMEMModelInternals)
        self.internals = internals

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
                self.parameters = Parameters(list(self.parameters) + [omega])
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
        if conf.write_etas_in_abbr:
            abbr = abbr_translation(self, rv_trans)
            trans.update(abbr)
        if hasattr(self, '_statements'):
            update_statements(self, self.internals._old_statements, self._statements, trans)
            self.internals._old_statements = self._statements

        if (
            self.internals._dataset_updated
            or self.datainfo != self.internals._old_datainfo
            or self.datainfo.path != self.internals._old_datainfo.path
        ):
            # FIXME: If no name set use the model name. Set that when setting dataset to input!
            if (
                self.datainfo.path is None
            ):  # or self.datainfo.path == self.internals._old_datainfo.path:
                dir_path = Path(self.name + ".csv") if path is None else path.parent

                if nofiles:
                    datapath = create_dataset_path(self, path=dir_path)
                else:
                    datapath = write_csv(self, path=dir_path, force=force)

                self.datainfo = self.datainfo.derive(path=datapath)

            data_record = self.internals.control_stream.get_records('DATA')[0]

            label = self.datainfo.names[0]
            data_record.ignore_character_from_header(label)
            update_input(self)

            # Remove IGNORE/ACCEPT. Could do diff between old dataset and find simple
            # IGNOREs to add i.e. for filter out certain ID.
            del data_record.ignore
            del data_record.accept
            self.internals._dataset_updated = False
            self.internals._old_datainfo = self.datainfo

            datapath = self.datainfo.path
            if datapath is not None:
                assert (
                    not datapath.exists() or datapath.is_file()
                ), f'input path change, but no file exists at target {str(datapath)}'
                data_record = self.internals.control_stream.get_records('DATA')[0]
                parent_path = Path.cwd() if path is None else path.parent
                data_record.filename = str(path_relative_to(parent_path, datapath))

        update_sizes(self)
        update_estimation(self)

        if self.observation_transformation != self.internals._old_observation_transformation:
            if not nofiles:
                update_ccontra(self, path, force)
        update_description(self)

        if self._name != self.internals.old_name:
            update_name_of_tables(self.internals.control_stream, self._name)

    @property
    def model_code(self):
        self.update_source(nofiles=True)
        return str(self.internals.control_stream)

    @property
    def dataset(self):
        if not hasattr(self, '_dataset'):
            self._dataset = parse_dataset(self.datainfo, self.internals.control_stream, raw=False)
        return self._dataset

    @dataset.setter
    def dataset(self, df):
        self.internals._dataset_updated = True
        self._dataset = df
        self.datainfo = self.datainfo.derive(path=None)
        self.update_datainfo()

    def read_raw_dataset(self, parse_columns: Tuple[str, ...] = ()):
        return parse_dataset(
            self.datainfo, self.internals.control_stream, raw=True, parse_columns=parse_columns
        )


def parse_code(code: str, path: Optional[Path] = None, dataset: Optional[pd.DataFrame] = None, **_):
    parser = NMTranParser()
    if path is None:
        name = 'run1'
        filename_extension = '.ctl'
    else:
        name = path.stem
        filename_extension = path.suffix

    control_stream = parser.parse(code)
    di = parse_datainfo(control_stream, path)
    _old_datainfo = di
    # FIXME temporary workaround remove when changing constructor
    # NOTE inputting the dataset is needed for IV models when using convert_model, since it needs to read the
    # RATE column to decide dosing, meaning it needs the dataset before parsing statements
    if dataset is not None:
        di = update_datainfo(di.derive(path=None), dataset)

    dependent_variable = sympy.Symbol('Y')

    parameters = parse_parameters(control_stream)

    statements, comp_map = parse_statements(
        di, _dataset(control_stream, di, dataset), control_stream
    )

    rvs = parse_random_variables(control_stream)

    trans_statements, trans_params = create_name_trans(control_stream, rvs, statements)
    for theta in control_stream.get_records('THETA'):
        theta.update_name_map(trans_params)
    for omega in control_stream.get_records('OMEGA'):
        omega.update_name_map(trans_params)
    for sigma in control_stream.get_records('SIGMA'):
        sigma.update_name_map(trans_params)

    d_par = {key: value for key, value in trans_params.items() if key in parameters}

    d_rv = {
        sympy.Symbol(key): value for key, value in trans_params.items() if key not in parameters
    }

    _old_random_variables = rvs  # FIXME: This has to stay here
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
        before, after = compare_before_after_params(parameters.inits, nearest)
        warnings.warn(
            f"Adjusting initial estimates to create positive semidefinite "
            f"omega/sigma matrices.\nBefore adjusting:  {before}.\n"
            f"After adjusting: {after}"
        )
        parameters = parameters.set_initial_estimates(nearest)

    estimation_steps = parse_estimation_steps(control_stream, rvs)

    description = parse_description(control_stream)

    init_etas = parse_initial_individual_estimates(
        control_stream, rvs, None if path is None else path.parent
    )

    value_type = parse_value_type(control_stream, statements)

    modelfit_results = _parse_modelfit_results(
        path,
        control_stream,
        BaseModel(  # FIXME This should not be necessary
            name=name,
            description=description,
            parameters=parameters,
            random_variables=rvs,
            statements=statements,
            dependent_variable=dependent_variable,
            estimation_steps=estimation_steps,
        ),
    )

    internals = NONMEMModelInternals(
        control_stream=control_stream,
        old_name=name,
        _old_description=description,
        _old_estimation_steps=estimation_steps,
        _old_observation_transformation=dependent_variable,
        _old_parameters=parameters,
        _old_random_variables=_old_random_variables,
        _old_statements=statements,
        _old_initial_individual_estimates=init_etas,
        _old_datainfo=_old_datainfo,
        _dataset_updated=False,
        _compartment_map=comp_map,
    )

    return Model(
        name=name,
        parameters=parameters,
        random_variables=rvs,
        statements=statements,
        dataset=dataset,
        datainfo=di,
        dependent_variable=dependent_variable,
        estimation_steps=estimation_steps,
        modelfit_results=modelfit_results,
        parent_model=None,
        initial_individual_estimates=init_etas,
        filename_extension=filename_extension,
        value_type=value_type,
        description=description,
        internals=internals,
    )
