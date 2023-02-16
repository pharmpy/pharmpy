# The NONMEM Model class
from __future__ import annotations

import dataclasses
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from pharmpy.deps import pandas as pd
from pharmpy.deps import sympy
from pharmpy.internals.expr.subs import subs
from pharmpy.internals.fs.path import path_absolute, path_relative_to
from pharmpy.model import Assignment, DataInfo, EstimationSteps
from pharmpy.model import Model as BaseModel
from pharmpy.model import NormalDistribution, Parameter, Parameters, RandomVariables, Statements
from pharmpy.model.model import compare_before_after_params, update_datainfo
from pharmpy.modeling.write_csv import write_csv

from .nmtran_parser import NMTranControlStream, NMTranParser
from .parsing import (
    parse_datainfo,
    parse_dataset,
    parse_description,
    parse_estimation_steps,
    parse_initial_individual_estimates,
    parse_parameters,
    parse_statements,
    parse_value_type,
)
from .results import _parse_modelfit_results
from .update import (
    abbr_translation,
    create_name_map,
    update_ccontra,
    update_description,
    update_estimation,
    update_initial_individual_estimates,
    update_input,
    update_name_of_tables,
    update_random_variables,
    update_sizes,
    update_statements,
    update_thetas,
)


@dataclass(frozen=True)
class NONMEMModelInternals:
    control_stream: NMTranControlStream
    old_name: str
    old_description: str
    old_estimation_steps: EstimationSteps
    old_observation_transformation: sympy.Symbol
    old_parameters: Parameters
    old_random_variables: RandomVariables
    old_statements: Statements
    old_initial_individual_estimates: Optional[pd.DataFrame]
    old_datainfo: DataInfo
    dataset_updated: bool
    compartment_map: Optional[Dict[str, int]]
    name_map: Dict[str, str]

    def replace(self, **kwargs):
        return dataclasses.replace(self, **kwargs)


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
    nm_model._parameters = model.parameters
    if hasattr(model, 'name'):
        nm_model.name = model.name
    nm_model._dataset = model.dataset
    nm_model._estimation_steps = model.estimation_steps
    nm_model._initial_individual_estimates = model.initial_individual_estimates
    new_obs_trans = subs(
        model.observation_transformation,
        {model.dependent_variable: nm_model.dependent_variable},
        simultaneous=True,
    )
    internals = nm_model.internals.replace(old_parameters=Parameters())
    nm_model = nm_model.replace(
        value_type=model.value_type,
        observation_transformation=new_obs_trans,
        dependent_variable=sympy.Symbol('Y'),
        random_variables=model.random_variables,
        statements=model.statements,
        description=model.description,
        internals=internals,
    )
    nm_model = nm_model.update_source()
    if model.statements.ode_system:
        nm_model = nm_model.replace(
            internals=nm_model.internals.replace(
                compartment_map={
                    name: i
                    for i, name in enumerate(model.statements.ode_system.compartment_names, start=1)
                }
            )
        )
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
        self._internals = internals

    def update_source(self):
        """Update the source

        path - path to modelfile
        nofiles - Set to not write any files (i.e. dataset, phi input etc)
        """
        model = self
        if (
            model.initial_individual_estimates
            is not model.internals.old_initial_individual_estimates
        ):
            cs = update_initial_individual_estimates(model, path='DUMMYPATH', nofiles=True)
            model = model.replace(
                internals=model.internals.replace(
                    control_stream=cs,
                    old_initial_individual_estimates=model.initial_individual_estimates,
                )
            )

        if not model.random_variables.etas:
            omega = Parameter('DUMMYOMEGA', init=0, fix=True)
            eta = NormalDistribution.create('eta_dummy', 'iiv', 0, omega.symbol)
            statement = Assignment(sympy.Symbol('DUMMYETA'), sympy.Symbol(eta.names[0]))
            model = model.replace(
                statements=statement + model.statements,
                random_variables=model.random_variables + eta,
                parameters=Parameters.create(list(model.parameters) + [omega]),
            )

        control_stream = update_random_variables(
            model, model.internals.old_random_variables, model._random_variables
        )

        control_stream = update_thetas(
            model, control_stream, model.internals.old_parameters, model._parameters
        )

        model = model.replace(
            internals=model.internals.replace(
                old_parameters=model._parameters,
                old_random_variables=model._random_variables,
                control_stream=control_stream,
            )
        )

        trans = create_name_map(model)

        rv_trans = {}
        i = 1
        for dist in model._random_variables.etas:
            for name in dist.names:
                rv_trans[name] = f'ETA({i})'
                i += 1
        model, _ = abbr_translation(model, rv_trans)

        trans = {sympy.Symbol(key): sympy.Symbol(value) for key, value in trans.items()}
        model, updated_dataset = update_statements(
            model, model.internals.old_statements, model._statements, trans
        )

        cs = model.internals.control_stream
        if (
            updated_dataset
            or model.internals.dataset_updated
            or model.datainfo != model.internals.old_datainfo
            or model.datainfo.path != model.internals.old_datainfo.path
        ):
            data_record = control_stream.get_records('DATA')[0]
            label = model.datainfo.names[0]
            newdata = data_record.set_ignore_character_from_header(label)
            cs = update_input(cs, model)

            # Remove IGNORE/ACCEPT. Could do diff between old dataset and find simple
            # IGNOREs to add i.e. for filter out certain ID.
            newdata = newdata.remove_ignore().remove_accept()
            if model.datainfo.path is None or model.internals.dataset_updated or updated_dataset:
                newdata = newdata.set_filename('DUMMYPATH')

            cs = cs.replace_records([data_record], [newdata])

        cs = update_sizes(cs, model)
        cs = update_estimation(cs, model)
        cs = update_description(cs, model.internals.old_description, model.description)

        if model._name != model.internals.old_name:
            cs = update_name_of_tables(control_stream, model._name)

        new_internals = model.internals.replace(
            control_stream=cs,
            old_description=model._description,
            old_estimation_steps=model._estimation_steps,
            old_datainfo=model._datainfo,
            old_statements=model._statements,
            dataset_updated=False,
        )
        model = model.replace(internals=new_internals)

        return model

    def write_files(self, path=None, force=False):
        model = self.update_source()

        etas_record = model.internals.control_stream.get_records('ETAS')
        if etas_record and str(etas_record[0].path) == 'DUMMYPATH':
            cs = update_initial_individual_estimates(model, path)
            model = model.replace(
                internals=model.internals.replace(
                    control_stream=cs,
                )
            )

        replace_dict = {}

        data_record = model.internals.control_stream.get_records('DATA')[0]
        if data_record.filename == 'DUMMYPATH' or force:
            datapath = model.datainfo.path
            if datapath is None:
                dir_path = Path(model.name + ".csv") if path is None else path.parent
                datapath = write_csv(model, path=dir_path, force=force)
                replace_dict['datainfo'] = model.datainfo.replace(path=datapath)
            assert (
                not datapath.exists() or datapath.is_file()
            ), f'input path change, but no file exists at target {str(datapath)}'
            parent_path = Path.cwd() if path is None else path.parent
            try:
                filename = str(path_relative_to(parent_path, datapath))
            except ValueError:
                # NOTE if parent path and datapath are in different drives absolute path
                # needs to be used
                filename = str(path_absolute(datapath))
                warnings.warn('Cannot resolve relative path, falling back to absolute path')
            newdata = data_record.set_filename(filename)
            newcs = model.internals.control_stream.replace_records([data_record], [newdata])
            replace_dict['internals'] = model.internals.replace(control_stream=newcs)

        if model.observation_transformation != model.internals.old_observation_transformation:
            update_ccontra(model, path, force)

        if replace_dict:
            return model.replace(**replace_dict)
        else:
            return model

    @property
    def model_code(self):
        return str(self.internals.control_stream)

    @property
    def dataset(self):
        if not hasattr(self, '_dataset'):
            self._dataset = parse_dataset(self.datainfo, self.internals.control_stream, raw=False)
        return self._dataset

    @dataset.setter
    def dataset(self, df):
        if hasattr(self, '_dataset'):
            # FIXME
            self._internals = self.internals.replace(dataset_updated=True)
        self._dataset = df
        self.datainfo = self.datainfo.replace(path=None)
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
    old_datainfo = di
    # FIXME temporary workaround remove when changing constructor
    # NOTE inputting the dataset is needed for IV models when using convert_model, since it needs to read the
    # RATE column to decide dosing, meaning it needs the dataset before parsing statements
    if dataset is not None:
        di = update_datainfo(di.replace(path=None), dataset)

    dependent_variable = sympy.Symbol('Y')

    statements, comp_map = parse_statements(
        di, _dataset(control_stream, di, dataset), control_stream
    )

    parameters, rvs, name_map = parse_parameters(control_stream, statements)

    subs_map = {sympy.Symbol(key): sympy.Symbol(val) for key, val in name_map.items()}
    statements = statements.subs(subs_map)

    # FIXME: Handle by creation of new model object
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
        control_stream, name_map, None if path is None else path.parent
    )

    value_type = parse_value_type(control_stream, statements)

    modelfit_results = _parse_modelfit_results(
        path,
        control_stream,
        name_map,
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
        old_description=description,
        old_estimation_steps=estimation_steps,
        old_observation_transformation=dependent_variable,
        old_parameters=parameters,
        old_random_variables=rvs,
        old_statements=statements,
        old_initial_individual_estimates=init_etas,
        old_datainfo=old_datainfo,
        dataset_updated=False,
        compartment_map=comp_map,
        name_map=name_map,
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
