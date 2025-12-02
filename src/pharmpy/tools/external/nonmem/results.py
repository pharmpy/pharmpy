from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Union, overload

import pharmpy.modeling as modeling
from pharmpy.basic import Expr
from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.internals.immutable import cache_method_no_args
from pharmpy.internals.math import nearest_positive_semidefinite
from pharmpy.model import ExecutionSteps, Model, Parameters, RandomVariables
from pharmpy.model.external.nonmem.nmtran_parser import NMTranControlStream
from pharmpy.model.external.nonmem.parsing import extract_verbatim_derivatives, parse_table_columns
from pharmpy.model.external.nonmem.records.table_record import DEFAULT_TABLE_RECORD_FORMAT
from pharmpy.model.external.nonmem.table import ExtTable, NONMEMTableFile, PhiTable
from pharmpy.model.external.nonmem.update import create_name_map
from pharmpy.workflows.log import Log
from pharmpy.workflows.results import ModelfitResults, SimulationResults

from .results_file import NONMEMResultsFile


@dataclass(frozen=True)
class Tables:
    residuals: pd.DataFrame
    predictions: pd.DataFrame
    derivatives: pd.DataFrame


@dataclass(frozen=True)
class Individuals:
    ofv: Optional[pd.Series]
    estimates: Optional[pd.DataFrame]
    estimates_covariance: Optional[pd.Series]


@dataclass(frozen=True)
class Iterations:
    table_numbers: list[int | None]
    final_ofv: float
    ofv: pd.Series
    final_pe: pd.Series
    sdcorr: pd.Series
    pe: pd.DataFrame
    ses: pd.Series
    ses_sdcorr: pd.Series
    cov_abort: bool
    condition_number: float


@dataclass(frozen=True)
class Status:
    runtime_total: float | None
    log_likelihood: float
    covstatus: bool | None
    covstatus_table_number: int | None
    minimization_successful: list[bool]
    function_evaluations: list[int]
    significant_digits: list[int]
    termination_cause: list[Any]
    estimation_runtime: list[float]
    estimate_near_boundary: list[bool]
    log: Log | None
    est_table_numbers: list[int | None]


@dataclass(frozen=True)
class Gradient:
    all_iterations: Optional[pd.DataFrame]
    final_is_zero: Optional[bool]
    final_iteration: Optional[pd.Series]


@dataclass(frozen=True)
class Covariance:
    cov: pd.DataFrame
    cor: pd.DataFrame
    coi: pd.DataFrame


@dataclass
class ModelfitResultsProxy:
    path: Path
    control_stream: NMTranControlStream
    name_map: dict[str, str]
    model: Model
    strict: bool = False
    subproblem: Optional[int] = None
    log: Optional[Log] = None

    @property
    @cache_method_no_args
    def iterations(self):
        path = self.path

        ext_path = path.with_suffix('.ext')

        try:
            ext_tables = NONMEMTableFile(ext_path)
        except FileNotFoundError:
            msg = f"Couldn't find NONMEM .ext-file at {ext_path}"
            if self.log is not None:
                self.log = self.log.log_error(msg)
            raise FileNotFoundError(msg)
        except ValueError:
            if self.log is not None:
                self.log = self.log.log_error(f"Broken ext-file {path.with_suffix('.ext')}")
            raise

        for table in ext_tables:
            try:
                table.data_frame
            except ValueError:
                if self.log is not None:
                    self.log = self.log.log_error(
                        f"Broken table in ext-file {path.with_suffix('.ext')}, "
                        f"table no. {table.number}"
                    )

        control_stream = self.control_stream
        name_map = self.name_map
        subproblem = self.subproblem
        model = self.model
        parameters = model.parameters
        return _parse_ext(control_stream, name_map, ext_tables, subproblem, parameters)

    @property
    @cache_method_no_args
    def status(self):
        model = self.model
        execution_steps = model.execution_steps
        path = self.path
        log = self.log
        status = _parse_lst(len(execution_steps), path, self.iterations.table_numbers, log)
        self.log = status.log
        return status

    @property
    @cache_method_no_args
    def gradient(self):
        path = self.path
        control_stream = self.control_stream
        name_map = self.name_map
        model = self.model
        parameters = model.parameters
        subproblem = self.subproblem

        return _parse_grd(path, control_stream, name_map, parameters, subproblem)

    @property
    @cache_method_no_args
    def last_estimation_index(self):
        model = self.model
        execution_steps = model.execution_steps
        return _get_last_est(execution_steps)

    @property
    @cache_method_no_args
    def evaluation(self):
        model = self.model
        execution_steps = model.execution_steps
        return _parse_evaluation(execution_steps)

    @property
    @cache_method_no_args
    def relative_standard_errors(self) -> Optional[pd.Series]:
        return _calculate_relative_standard_errors(self.iterations.final_pe, self.iterations.ses)

    @property
    @cache_method_no_args
    def _minimization_successful(self) -> list[bool]:
        return override_minimization_successful(
            self.status.minimization_successful, self.iterations.pe
        )

    @property
    @cache_method_no_args
    def minimization_successful(self) -> bool:
        return self._minimization_successful[self.last_estimation_index]

    @property
    @cache_method_no_args
    def estimation_steps(self):
        if self.status.est_table_numbers:
            return self.status.est_table_numbers
        else:
            model = self.model
            execution_steps = model.execution_steps
            return list(range(1, len(execution_steps) + 1))

    @property
    @cache_method_no_args
    def minimization_successful_iterations(self) -> pd.Series:
        return pd.Series(
            self._minimization_successful,
            index=self.estimation_steps,
            name='minimization_successful',
        )

    @property
    @cache_method_no_args
    def estimation_runtime_iterations(self) -> pd.Series:
        estimation_runtime = self.status.estimation_runtime
        return pd.Series(estimation_runtime, index=self.estimation_steps, name='estimation_runtime')

    @property
    @cache_method_no_args
    def individuals(self):
        path = self.path
        control_stream = self.control_stream
        name_map = self.name_map
        model = self.model
        etas = model.random_variables.etas
        final_pe = self.iterations.final_pe
        subproblem = self.subproblem
        return _parse_phi(path, control_stream, name_map, etas, model, final_pe, subproblem)

    @property
    @cache_method_no_args
    def tables(self):
        model = self.model
        path = self.path
        control_stream = self.control_stream
        table_df = _parse_tables(
            model, path, control_stream, netas=len(model.random_variables.etas.names)
        )  # $TABLEs
        residuals = _parse_residuals(table_df)
        predictions = _parse_predictions(table_df)
        derivatives = _parse_derivatives(table_df, model)
        return Tables(residuals=residuals, predictions=predictions, derivatives=derivatives)

    @property
    @cache_method_no_args
    def covariance(self):
        path = self.path
        control_stream = self.control_stream
        name_map = self.name_map

        if (
            self.status.covstatus
            and self.iterations.ses is not None
            and not self.iterations.cov_abort
        ):
            cov = _parse_matrix(
                path.with_suffix(".cov"), control_stream, name_map, self.iterations.table_numbers
            )
            cor = _parse_matrix(
                path.with_suffix(".cor"), control_stream, name_map, self.iterations.table_numbers
            )
            if cor is not None:
                np.fill_diagonal(cor.values, 1)
            coi = _parse_matrix(
                path.with_suffix(".coi"), control_stream, name_map, self.iterations.table_numbers
            )
        else:
            cov, cor, coi = None, None, None

        cov, cor, coi = calculate_cov_cor_coi(cov, cor, coi, self.iterations.ses)
        if cov is not None:
            cov = nearest_positive_semidefinite(cov)

        return Covariance(cov, cor, coi)

    @property
    @cache_method_no_args
    def standard_errors(self) -> Optional[pd.Series]:
        ses = self.iterations.ses

        if ses is None:
            if self.covariance.cov is not None:
                ses = modeling.calculate_se_from_cov(self.covariance.cov)
            elif self.covariance.coi is not None:
                ses = modeling.calculate_se_from_prec(self.covariance.coi)

        return ses

    @property
    @cache_method_no_args
    def termination_cause_iterations(self) -> Optional[pd.Series]:
        return pd.Series(
            self.status.termination_cause, index=self.estimation_steps, name='termination_cause'
        )

    @property
    @cache_method_no_args
    def function_evaluations_iterations(self) -> Optional[pd.Series]:
        return pd.Series(
            self.status.function_evaluations,
            index=self.estimation_steps,
            name='function_evaluations',
        )

    @property
    @cache_method_no_args
    def significant_digits_iterations(self) -> Optional[pd.Series]:
        return pd.Series(
            self.status.significant_digits, index=self.estimation_steps, name='significant_digits'
        )

    @property
    @cache_method_no_args
    def covstep_successful(self) -> Optional[bool]:
        model = self.model
        if (
            not model.execution_steps
            or model.execution_steps[-1].parameter_uncertainty_method is None
        ):
            return None
        elif self.status.covstatus:
            return True
        else:
            return False

    @property
    @cache_method_no_args
    def warnings(self) -> Optional[list[str]]:
        warnings = []
        if any(self.status.estimate_near_boundary):
            warnings.append('estimate_near_boundary')
        if self.gradient.final_is_zero:
            warnings.append('final_zero_gradient')

        return warnings

    @property
    @cache_method_no_args
    def individual_eta_samples(self) -> Optional[pd.DataFrame]:
        path = self.path
        model = self.model
        etas = model.random_variables.etas
        subproblem = self.subproblem

        return _parse_ets(path, etas, subproblem)


@overload
def _parse_modelfit_results(
    path: Optional[Union[str, Path]],
    control_stream: NMTranControlStream,
    name_map,
    model: Model,
    strict: bool,
    subproblem: Optional[int],
    with_log: bool,
    lazy: Literal[False],
) -> ModelfitResults | None: ...


@overload
def _parse_modelfit_results(
    path: Optional[Union[str, Path]],
    control_stream: NMTranControlStream,
    name_map,
    model: Model,
    strict: bool,
    subproblem: Optional[int],
    with_log: bool,
    lazy: Literal[True],
) -> LazyModelfitResults | None: ...


def _parse_modelfit_results(
    path: Optional[Union[str, Path]],
    control_stream: NMTranControlStream,
    name_map,
    model: Model,
    strict: bool = False,
    subproblem: Optional[int] = None,
    with_log: bool = True,
    lazy: bool = False,
):
    # Path to model file or results file
    if path is None:
        return None

    path = Path(path)

    proxy = ModelfitResultsProxy(
        path=path,
        control_stream=control_stream,
        name_map=name_map,
        model=model,
        strict=strict,
        subproblem=subproblem,
        log=Log() if with_log else None,
    )

    try:
        proxy.iterations
    except FileNotFoundError:
        if strict:
            raise
        else:
            return create_failed_results(model, proxy.log)
    except OSError:
        return None
    except ValueError:
        return create_failed_results(model, proxy.log)

    _lazy = LazyModelfitResults(proxy)
    if lazy:
        return _lazy

    res = ModelfitResults(
        minimization_successful=_lazy.minimization_successful,
        minimization_successful_iterations=_lazy.minimization_successful_iterations,
        estimation_runtime=_lazy.estimation_runtime,
        estimation_runtime_iterations=_lazy.estimation_runtime_iterations,
        function_evaluations=_lazy.function_evaluations,
        function_evaluations_iterations=_lazy.function_evaluations_iterations,
        termination_cause=_lazy.termination_cause,
        termination_cause_iterations=_lazy.termination_cause_iterations,
        significant_digits=_lazy.significant_digits,
        significant_digits_iterations=_lazy.significant_digits_iterations,
        relative_standard_errors=_lazy.relative_standard_errors,
        individual_estimates=_lazy.individual_estimates,
        individual_estimates_covariance=_lazy.individual_estimates_covariance,
        runtime_total=_lazy.runtime_total,
        log_likelihood=_lazy.log_likelihood,
        covariance_matrix=_lazy.covariance_matrix,
        correlation_matrix=_lazy.correlation_matrix,
        precision_matrix=_lazy.precision_matrix,
        standard_errors=_lazy.standard_errors,
        standard_errors_sdcorr=_lazy.standard_errors_sdcorr,
        individual_ofv=_lazy.individual_ofv,
        parameter_estimates=_lazy.parameter_estimates,
        parameter_estimates_sdcorr=_lazy.parameter_estimates_sdcorr,
        parameter_estimates_iterations=_lazy.parameter_estimates_iterations,
        ofv=_lazy.ofv,
        ofv_iterations=_lazy.ofv_iterations,
        predictions=_lazy.predictions,
        residuals=_lazy.residuals,
        derivatives=_lazy.derivatives,
        evaluation=_lazy.evaluation,
        covstep_successful=_lazy.covstep_successful,
        gradients=_lazy.gradients,
        gradients_iterations=_lazy.gradients_iterations,
        warnings=_lazy.warnings,
        individual_eta_samples=_lazy.individual_eta_samples,
        condition_number=_lazy.condition_number,
        # NOTE: `proxy.log` is extracted last because other
        #       property extractions can update `proxy.log`.
        log=_lazy.log,
    )
    return res


@dataclass(frozen=True)
class LazyModelfitResults:
    _proxy: ModelfitResultsProxy

    @property
    def minimization_successful(self):
        return self._proxy.minimization_successful

    @property
    def minimization_successful_iterations(self):
        return self._proxy.minimization_successful_iterations

    @property
    def estimation_runtime(self):
        return self._proxy.status.estimation_runtime[self._proxy.last_estimation_index]

    @property
    def estimation_runtime_iterations(self):
        return self._proxy.estimation_runtime_iterations

    @property
    def function_evaluations(self):
        return self._proxy.status.function_evaluations[self._proxy.last_estimation_index]

    @property
    def function_evaluations_iterations(self):
        return self._proxy.function_evaluations_iterations

    @property
    def termination_cause(self):
        return self._proxy.status.termination_cause[self._proxy.last_estimation_index]

    @property
    def termination_cause_iterations(self):
        return self._proxy.termination_cause_iterations

    @property
    def significant_digits(self):
        return self._proxy.status.significant_digits[-1]

    @property
    def significant_digits_iterations(self):
        return self._proxy.significant_digits_iterations

    @property
    def relative_standard_errors(self):
        return self._proxy.relative_standard_errors

    @property
    def individual_estimates(self):
        return self._proxy.individuals.estimates

    @property
    def individual_estimates_covariance(self):
        return self._proxy.individuals.estimates_covariance

    @property
    def runtime_total(self):
        return self._proxy.status.runtime_total

    @property
    def log_likelihood(self):
        return self._proxy.status.log_likelihood

    @property
    def covariance_matrix(self):
        return self._proxy.covariance.cov

    @property
    def correlation_matrix(self):
        return self._proxy.covariance.cor

    @property
    def precision_matrix(self):
        return self._proxy.covariance.coi

    @property
    def standard_errors(self):
        return self._proxy.standard_errors

    @property
    def standard_errors_sdcorr(self):
        return self._proxy.iterations.ses_sdcorr

    @property
    def individual_ofv(self):
        return self._proxy.individuals.ofv

    @property
    def parameter_estimates(self):
        return self._proxy.iterations.final_pe

    @property
    def parameter_estimates_sdcorr(self):
        return self._proxy.iterations.sdcorr

    @property
    def parameter_estimates_iterations(self):
        return self._proxy.iterations.pe

    @property
    def ofv(self) -> float | None:
        return self._proxy.iterations.final_ofv

    @property
    def ofv_iterations(self):
        return self._proxy.iterations.ofv

    @property
    def predictions(self):
        return self._proxy.tables.predictions

    @property
    def residuals(self):
        return self._proxy.tables.residuals

    @property
    def derivatives(self):
        return self._proxy.tables.derivatives

    @property
    def evaluation(self):
        return self._proxy.evaluation

    @property
    def covstep_successful(self):
        return self._proxy.covstep_successful

    @property
    def gradients(self):
        return self._proxy.gradient.final_iteration

    @property
    def gradients_iterations(self):
        return self._proxy.gradient.all_iterations

    @property
    def warnings(self):
        return self._proxy.warnings

    @property
    def individual_eta_samples(self):
        return self._proxy.individual_eta_samples

    @property
    def condition_number(self):
        return self._proxy.iterations.condition_number

    @property
    def log(self):
        return self._proxy.log


def calculate_cov_cor_coi(cov, cor, coi, ses):
    if cov is None:
        if cor is not None:
            cov = modeling.calculate_cov_from_corrse(cor, ses)
        elif coi is not None:
            cov = modeling.calculate_cov_from_prec(coi)
    if cor is None:
        if cov is not None:
            cor = modeling.calculate_corr_from_cov(cov)
        elif coi is not None:
            cor = modeling.calculate_corr_from_prec(coi)
    if coi is None:
        if cov is not None:
            coi = modeling.calculate_prec_from_cov(cov)
        elif cor is not None:
            coi = modeling.calculate_prec_from_corrse(cor, ses)
    return cov, cor, coi


def _parse_matrix(
    path: Path,
    control_stream: NMTranControlStream,
    name_map,
    table_numbers,
):
    try:
        tables = NONMEMTableFile(path)
    except OSError:
        return None

    last_table = tables.table_no(table_numbers[-1])
    assert last_table is not None
    df = last_table.data_frame
    if df is not None:
        df = df.rename(index=name_map)
        df.columns = df.index
    return df


def _empty_lst_results(n: int, log):
    false_vec = [False] * n
    nan_vec = [np.nan] * n
    none_vec = [None] * n
    return Status(
        None,
        np.nan,
        False,
        None,
        false_vec,
        nan_vec,
        nan_vec,
        none_vec,
        nan_vec,
        false_vec,
        log,
        nan_vec,
    )


def _parse_lst(n: int, path: Path, table_numbers: list[int | None], log: Log | None):
    try:
        rfile = NONMEMResultsFile(path.with_suffix('.lst'), log=log)
    except OSError:
        return _empty_lst_results(n, log)

    if not table_numbers:
        return _empty_lst_results(n, log)

    runtime_total = rfile.runtime_total

    covstatus_table_number = table_numbers[-1]

    try:
        for table_number in reversed(table_numbers):
            if "OPTIMALITY" not in rfile.table[table_number]['METH']:
                log_likelihood_table_number = table_number
                break
        else:
            log_likelihood_table_number = None
    except (KeyError, FileNotFoundError):
        log_likelihood_table_number = None

    try:
        log_likelihood = rfile.table[log_likelihood_table_number]['ofv_with_constant']
    except (KeyError, FileNotFoundError):
        log_likelihood = np.nan

    covstatus = rfile.covariance_status(covstatus_table_number)['covariance_step_ok']
    assert covstatus is None or isinstance(covstatus, bool)

    if log_likelihood_table_number is not None:
        est_table_numbers: list[int | None] = [
            table_number
            for table_number in table_numbers
            if table_number <= log_likelihood_table_number
        ]
    else:
        est_table_numbers: list[int | None] = [table_number for table_number in table_numbers]

    (
        minimization_successful,
        function_evaluations,
        significant_digits,
        termination_cause,
        estimation_runtime,
        estimate_near_boundary,
    ) = parse_estimation_status(rfile, est_table_numbers)

    return Status(
        runtime_total,
        log_likelihood,
        covstatus,
        covstatus_table_number,
        minimization_successful,
        function_evaluations,
        significant_digits,
        termination_cause,
        estimation_runtime,
        estimate_near_boundary,
        rfile.log,
        est_table_numbers,
    )


def parse_estimation_status(results_file: NONMEMResultsFile, table_numbers: list[int | None]):
    minimization_successful = []
    function_evaluations = []
    significant_digits = []
    termination_cause = []
    estimation_runtime = []
    estimate_near_boundary = []
    for tabno in table_numbers:
        estimation_status = results_file.estimation_status(tabno)
        minimization_successful.append(estimation_status['minimization_successful'])
        function_evaluations.append(estimation_status['function_evaluations'])
        significant_digits.append(estimation_status['significant_digits'])
        estimate_near_boundary.append(estimation_status['estimate_near_boundary'])
        if estimation_status['maxevals_exceeded'] is True:
            tc = 'maxevals_exceeded'
        elif estimation_status['rounding_errors'] is True:
            tc = 'rounding_errors'
        else:
            tc = None
        termination_cause.append(tc)
        try:
            er = results_file.table[tabno]['estimation_runtime']
        except (KeyError, FileNotFoundError):
            er = np.nan
        estimation_runtime.append(er)

    return (
        minimization_successful,
        function_evaluations,
        significant_digits,
        termination_cause,
        estimation_runtime,
        estimate_near_boundary,
    )


def _get_last_est(execution_steps: ExecutionSteps):
    # Find last estimation
    for i in range(len(execution_steps) - 1, -1, -1):
        step = execution_steps[i]
        if not step.evaluation:
            return i
    # If all steps were evaluation the last evaluation step is relevant
    return len(execution_steps) - 1


def _parse_phi(
    path: Path,
    control_stream: NMTranControlStream,
    name_map,
    etas: RandomVariables,
    model,
    pe,
    subproblem=None,
):
    try:
        phi_tables = NONMEMTableFile(path.with_suffix('.phi'))
    except FileNotFoundError:
        return Individuals(None, None, None)
    if subproblem is None:
        table = None

        for t in reversed(phi_tables.tables):
            if t.design_optimality is None:
                table = t
                break
    else:
        table = phi_tables.tables[subproblem - 1]

    if table is None:
        return Individuals(None, None, None)

    assert isinstance(table, PhiTable)

    eta_names = set(name_map.values())
    rv_names = list(filter(eta_names.__contains__, etas.names))
    try:
        individual_ofv = table.iofv
        prefix, individual_estimates = _parse_individual_estimates(model, pe, table, rv_names)
        ids, eta_col_names, matrix_array = table.etc_data()
        index = {name_map[x]: i for i, x in enumerate(eta_col_names)}
        indices = tuple(map(index.__getitem__, rv_names))
        selector = np.ix_(indices, indices)
        etc_frames = [
            pd.DataFrame(matrix[selector], columns=rv_names, index=rv_names)
            for matrix in matrix_array
        ]
        covs = pd.Series(etc_frames, index=ids, dtype='object')
        return Individuals(individual_ofv, individual_estimates, covs)
    except KeyError:
        return Individuals(None, None, None)


def _parse_individual_estimates(model, pe, table, rv_names):
    df = table.etas
    prefix = df.columns[0][0:3]  # PHI or ETA
    d = {f"{prefix}({i})": name for i, name in enumerate(rv_names, start=1)}
    df = df.rename(columns=d)  # [rv_names] needed?
    if prefix == "PHI":
        for i in range(1, len(rv_names) + 1):
            mu = Expr.symbol(f"MU_{i}")
            expr = model.statements.before_odes.full_expression(mu)
            if expr != mu:  # MU is defined in model code
                value = expr.subs(dict(pe)).subs(model.parameters.inits)
                columns = value.free_symbols
                if columns:
                    # MU depends on covariates
                    # NONMEM do not support time varying covariates so we can use baselines
                    colnames = [str(name) for name in columns]
                    baseline_data = modeling.get_baselines(model)[colnames]
                    for j, (_, row) in enumerate(baseline_data.iterrows()):
                        rowcovs = row.to_dict()
                        rowvalue = value.subs(rowcovs)
                        df.iloc[j, i - 1] -= float(rowvalue)
                else:
                    df.iloc[:, i - 1] -= float(value)
    return prefix, df


def _parse_grd(
    path: Path,
    control_stream: NMTranControlStream,
    name_map,
    parameters: Parameters,
    subproblem=None,
):
    try:
        grd_tables = NONMEMTableFile(path.with_suffix('.grd'))
    except FileNotFoundError:
        return Gradient(None, None, None)
    if subproblem is None:
        table = grd_tables.tables[-1]
    else:
        table = grd_tables.tables[subproblem - 1]

    if table is None:
        return Gradient(None, None, None)

    gradients_table = table.data_frame
    old_col_names = table.data_frame.columns.to_list()[1::]
    param_names = [name for name in list(name_map.values()) if name in parameters.names]
    new_col_names = {old_col_names[i]: param_names[i] for i in range(len(old_col_names))}
    gradients_table = gradients_table.rename(columns=new_col_names)
    _last_row = gradients_table.tail(1)
    _last_row = _last_row.drop(columns=['ITERATION'])
    last_row = _last_row.squeeze(axis=0)
    assert isinstance(last_row, pd.Series)
    last_row = last_row.rename('gradients')
    final_zero_gradient = (last_row == 0).any() or last_row.isnull().any()
    assert isinstance(final_zero_gradient, np.bool)
    return Gradient(gradients_table, final_zero_gradient, last_row)


def _parse_ets(path, etas, subproblem):
    try:
        ets_tables = NONMEMTableFile(path.with_suffix('.ets'))
    except FileNotFoundError:
        return None
    if subproblem is None:
        table = ets_tables.tables[-1]
    else:
        table = ets_tables.tables[subproblem - 1]

    if table is None:
        return None

    df = table.data_frame
    df = df.drop(columns=['SUBJECT_NO'])
    d = {f'ETA({i})': name for i, name in enumerate(etas.names, start=1)}
    d['SAMPLE'] = 'sample'
    df = df.rename(columns=d).set_index(['ID', 'sample'])
    return df


def _parse_tables(
    model: Model, path: Path, control_stream: NMTranControlStream, netas
) -> pd.DataFrame:
    """Parse $TABLE and table files into one large dataframe of useful columns"""
    interesting_columns = {
        'ID',
        'TIME',
        'PRED',
        'CIPREDI',
        'CPRED',
        'IPRED',
        'RES',
        'WRES',
        'CWRES',
        'MDV',
    }

    derivative_regex = r'[HG]\d+'
    verbatim_derivatives = extract_verbatim_derivatives(control_stream, model.random_variables)
    verbatim_derivativates_names = set(verbatim_derivatives.keys())
    interesting_columns = interesting_columns.union(verbatim_derivativates_names)

    table_recs = control_stream.get_records('TABLE')
    colnames_list = parse_table_columns(control_stream, netas)
    found = set()
    df = pd.DataFrame()
    for table_rec, colnames in zip(table_recs, colnames_list):
        if (
            table_rec.has_option("FIRSTONLY")
            or table_rec.has_option("LASTONLY")
            or table_rec.has_option("FIRSTLASTONLY")
        ):
            continue
        columns_in_table = []
        colnames_in_table = []
        for i, name in enumerate(colnames):
            if (
                name in interesting_columns or re.match(derivative_regex, name)
            ) and name not in found:
                found.add(name)
                colnames_in_table.append(name)
                columns_in_table.append(i)

        noheader = table_rec.has_option("NOHEADER")
        notitle = table_rec.has_option("NOTITLE") or noheader
        nolabel = table_rec.has_option("NOLABEL") or noheader
        format = table_rec.get_option("FORMAT") or DEFAULT_TABLE_RECORD_FORMAT
        table_path = path.parent / table_rec.path
        try:
            table_file = NONMEMTableFile(
                table_path, notitle=notitle, nolabel=nolabel, format=format
            )
        except IOError:
            continue
        table = table_file.tables[0]

        df[colnames_in_table] = table.data_frame.iloc[:, columns_in_table]

    if 'ID' in df.columns:
        df['ID'] = df['ID'].convert_dtypes()
    dataset = model.dataset
    if dataset is not None and not df.empty:
        if len(df) == len(dataset):
            # If dataset have been filtered after parsing and model has then been run by Pharmpy,
            # we can use the (filtered) index from the dataset to reindex.
            df = df.set_index(dataset.index)
        else:
            # Needed when rows have been dropped when parsing (e.g. when IDs with no observations
            # are filtered when parsing the dataset), but IDs still exist in results.
            # See https://github.com/pharmpy/pharmpy/pull/4106
            df = df.iloc[dataset.index]

    assert isinstance(df, pd.DataFrame)
    return df


def _extract_at_least_one_column(df: pd.DataFrame, cols):
    # Extract at least one column from df
    found_cols = [col for col in cols if col in df.columns]
    if not found_cols:
        return None
    return df[found_cols]


def _parse_residuals(df: pd.DataFrame):
    cols = ['RES', 'WRES', 'CWRES']
    df = _extract_at_least_one_column(df, cols)
    if df is not None:
        df = df.loc[(df != 0).any(axis=1)]  # Simple way of removing non-observations
    return df


def _parse_predictions(df: pd.DataFrame):
    cols = ['PRED', 'CIPREDI', 'CPRED', 'IPRED']
    df = _extract_at_least_one_column(df, cols)
    return df


def _parse_derivatives(df: pd.DataFrame, model: Model):
    cols = []
    derivative_regex = r'[HG]\d+'
    verbatim_derivatives = extract_verbatim_derivatives(
        model.internals.control_stream, model.random_variables
    )
    verbatim_derivatives_names = list(verbatim_derivatives.keys())
    cols += verbatim_derivatives_names

    rename_derivative_dict = {}  # Name is ";".join of names as stored in EstimationStep
    for col in df.columns:
        if re.match(derivative_regex, col):
            if match := re.match(r'H(\d+)1', col):
                param_name = model.random_variables.epsilons.names[int(match.group(1)) - 1]
                rename_derivative_dict[col] = param_name
            elif match := re.match(r'G(\d+)1', col):
                param_name = model.random_variables.etas.names[int(match.group(1)) - 1]
                rename_derivative_dict[col] = param_name
            cols.append(col)
        elif col in verbatim_derivatives_names:
            param_name = verbatim_derivatives[col]
            param_name = map(str, param_name)
            rename_derivative_dict[col] = ";".join(param_name)

    df = _extract_at_least_one_column(df, cols)
    if df is not None:
        df = df.rename(rename_derivative_dict, axis=1)
    return df


def _create_failed_ofv_iterations(n: int):
    steps = list(range(1, n + 1))
    iterations = [0] * n
    ofvs = [np.nan] * n
    ofv_iterations = create_ofv_iterations_series(ofvs, steps, iterations)
    return ofv_iterations


def create_ofv_iterations_series(ofv, steps, iterations):
    step_series = pd.Series(steps, dtype='int32', name='steps')
    iteration_series = pd.Series(iterations, dtype='int32', name='iteration')
    with warnings.catch_warnings():
        # Needed because pandas 2.1.1 uses the _data attribute that it
        # also has deprecated
        warnings.simplefilter("ignore")
        ofv_iterations = pd.Series(
            ofv, name='OFV', dtype='float64', index=[step_series, iteration_series]
        )
    return ofv_iterations


def _create_failed_parameter_estimates(parameters: Parameters):
    return pd.Series(np.nan, name='estimates', index=list(parameters.nonfixed.inits.keys()))


def _parse_ext(
    control_stream: NMTranControlStream,
    name_map,
    ext_tables: NONMEMTableFile,
    subproblem: Optional[int],
    parameters: Parameters,
):
    table_numbers = _parse_table_numbers(ext_tables, subproblem)

    final_ofv, ofv_iterations = _parse_ofv(ext_tables, subproblem)
    condition_number = _parse_condition_number(ext_tables, subproblem)
    final_pe, sdcorr, pe_iterations = _parse_parameter_estimates(
        control_stream, name_map, ext_tables, subproblem, parameters
    )
    ses, ses_sdcorr, cov_abort = _parse_standard_errors(
        control_stream, name_map, ext_tables, parameters, final_pe
    )
    return Iterations(
        table_numbers,
        final_ofv,
        ofv_iterations,
        final_pe,
        sdcorr,
        pe_iterations,
        ses,
        ses_sdcorr,
        cov_abort,
        condition_number,
    )


def _parse_table_numbers(ext_tables: NONMEMTableFile, subproblem: Optional[int]):
    table_numbers: list[int | None] = []
    for table in ext_tables.tables:
        if subproblem and table.subproblem != subproblem:
            continue
        table_numbers.append(table.number)
    return table_numbers


def _parse_condition_number(ext_tables: NONMEMTableFile, subproblem: Optional[int]):
    final_table = None
    for table in ext_tables.tables:
        if (subproblem and table.subproblem != subproblem) or table.design_optimality is not None:
            continue

        final_table = table

    assert isinstance(final_table, ExtTable)
    try:
        return final_table.condition_number
    except KeyError:
        return None


def _parse_ofv(ext_tables: NONMEMTableFile, subproblem: Optional[int]):
    step = []
    iteration = []
    ofv = []
    final_table = None
    for table_number, table in enumerate(ext_tables.tables, start=1):
        if (subproblem and table.subproblem != subproblem) or table.design_optimality is not None:
            continue
        df = _get_iter_df(table.data_frame)
        n = len(df)
        step += [table_number] * n
        iteration += list(df['ITERATION'])
        ofv += list(df['OBJ'])
        final_table = table

    assert isinstance(final_table, ExtTable)
    if np.isnan(ofv[-1]):
        final_ofv = ofv[-1]
    else:
        final_ofv = final_table.final_ofv
    ofv_iterations = create_ofv_iterations_series(ofv, step, iteration)
    return float(final_ofv), ofv_iterations


def _get_iter_df(df):
    final_iter = -(10**9)
    iters = df['ITERATION']
    if 0 not in iters.values and final_iter in iters.values:
        df = df[iters == -(10**9)]
        df.at[0, 'ITERATION'] = 0
    else:
        try:
            final_iter_ofv = df[iters == final_iter].iloc[-1].loc['OBJ']
        except IndexError:
            final_iter_ofv = np.nan
        last_iter_ofv = df[iters >= 0].iloc[-1].loc['OBJ']
        if final_iter_ofv != last_iter_ofv:
            new_iter_no = int(df[iters >= 0].iloc[-1].loc['ITERATION']) + 1
            if final_iter in iters.values:
                idx = df[iters == final_iter].index.values[0]
                df.at[idx, 'ITERATION'] = new_iter_no
                new_iter_no += 1
            nan_row_dict = {
                colname: (new_iter_no if colname == 'ITERATION' else np.nan)
                for colname in df.columns
            }
            nan_row = pd.DataFrame(nan_row_dict, index=[len(df)])
            df = pd.concat([df, nan_row])
        df = df[df['ITERATION'] >= 0]
    return df


def _calculate_relative_standard_errors(pe, se):
    if pe is None or se is None:
        ser = None
    else:
        ser = se / pe
        ser.name = 'RSE'
    return ser


def _parse_parameter_estimates(
    control_stream: NMTranControlStream,
    name_map,
    ext_tables: NONMEMTableFile,
    subproblem: Optional[int],
    parameters: Parameters,
):
    pe = pd.DataFrame()
    fixed_param_names = []
    final_table = None
    fix = None
    for table_number, table in enumerate(ext_tables.tables, start=1):
        if (subproblem and table.subproblem != subproblem) or table.design_optimality is not None:
            continue
        df = _get_iter_df(table.data_frame)
        assert isinstance(table, ExtTable)
        fix = _get_fixed_parameters(table, parameters, name_map)
        fixed_param_names = [name for name in list(df.columns)[1:-1] if fix[name]]
        df = df.drop(fixed_param_names + ['OBJ'], axis=1)
        df['step'] = table_number
        df = df.rename(columns=name_map)
        pe = pd.concat([pe, df])
        final_table = table

    assert fix is not None
    assert final_table is not None
    if pe.iloc[-1].drop(['ITERATION', 'step']).isnull().all():
        final_pe = pe.iloc[-1].drop(['ITERATION', 'step']).rename('estimates')
    else:
        final_pe = final_table.final_parameter_estimates
        final_pe = final_pe.drop(fixed_param_names)
        final_pe = final_pe.rename(index=name_map)
    pe = pe.rename(columns={'ITERATION': 'iteration'}).set_index(['step', 'iteration'])

    try:
        sdcorr = final_table.omega_sigma_stdcorr[~fix]
    except KeyError:
        sdcorr_ests = pd.Series(np.nan, index=pe.index)
    else:
        sdcorr = sdcorr.rename(index=name_map)
        sdcorr_ests = final_pe.copy()
        sdcorr_ests.update(sdcorr)
    return final_pe, sdcorr_ests, pe


def _parse_standard_errors(
    control_stream: NMTranControlStream,
    name_map,
    ext_tables: NONMEMTableFile,
    parameters: Parameters,
    pe: pd.Series,
):
    cov_abort = False
    table = ext_tables.tables[-1]
    assert isinstance(table, ExtTable)
    try:
        ses = table.standard_errors
    except KeyError:
        ses = pd.Series(np.nan, index=pe.index, name="SE")
        sdcorr_ses = pd.Series(np.nan, index=pe.index, name="SE_sdcorr")
        return ses, sdcorr_ses, cov_abort

    fix = _get_fixed_parameters(table, parameters, name_map)
    ses = ses[~fix]
    try:  # if line -1000000001 and -1000000005 exist in .ext file
        sdcorr = table.omega_sigma_se_stdcorr[~fix]
    except KeyError:  # if only line -1000000001 exists but not -1000000005
        ses = pd.Series(np.nan, index=pe.index, name="SE")
        sdcorr_ses = pd.Series(np.nan, index=pe.index, name="SE_sdcorr")
        cov_abort = True
        return ses, sdcorr_ses, cov_abort
    else:
        ses = ses.rename(index=name_map)
        sdcorr = sdcorr.rename(index=name_map)
        sdcorr_ses = ses.copy()
        sdcorr_ses.update(sdcorr)
        sdcorr_ses = sdcorr_ses.rename(index=name_map)
    return ses, sdcorr_ses, cov_abort


def _parse_evaluation(execution_steps: ExecutionSteps):
    index = list(range(1, len(execution_steps) + 1))
    evaluation = [est.evaluation for est in execution_steps]
    return pd.Series(evaluation, index=index, name='evaluation', dtype='float64')


def _get_fixed_parameters(table: ExtTable, parameters: Parameters, pe_translation: dict):
    try:
        return table.fixed
    except KeyError:
        # NM 7.2 does not have row -1000000006 indicating FIXED status
        ests = table.final_parameter_estimates
        fixed = pd.Series(parameters.fix)
        # NOTE: Parameters in result file haven't been renamed yet
        fixed = fixed.rename({value: key for key, value in pe_translation.items()})
        return pd.concat([fixed, pd.Series(True, index=ests.index.difference(fixed.index))])


def simfit_results(model, model_path):
    """Read in modelfit results from a simulation/estimation model"""
    nsubs = model.internals.control_stream.get_records('SIMULATION')[0].nsubs
    results = []
    for i in range(1, nsubs + 1):
        res = parse_modelfit_results(model, model_path, subproblem=i)
        results.append(res)
    return results


# def parse_ext(model, path, subproblem):
#     try:
#         ext_tables = NONMEMTableFile(path.with_suffix('.ext'))
#     except ValueError:
#         failed_pe = _create_failed_parameter_estimates(model.parameters)
#         n = len(model.execution_steps)
#         df = pd.concat([failed_pe] * n, axis=1).T
#         df['step'] = range(1, n + 1)
#         df['iteration'] = 0
#         df = df.set_index(['step', 'iteration'])
#         return (
#             [],
#             np.nan,
#             _create_failed_ofv_iterations(len(model.execution_steps)),
#             failed_pe,
#             failed_pe,
#             df,
#             None,
#             None,
#         )
#     return _parse_ext(model.internals.control_stream, ext_tables, subproblem, model.parameters)


def parse_modelfit_results(
    model,
    path: Optional[Union[str, Path]],
    strict=False,
    subproblem: Optional[int] = None,
    with_log: bool = True,
    lazy: bool = False,
):
    name_map = create_name_map(model)
    name_map = {value: key for key, value in name_map.items()}
    res = _parse_modelfit_results(
        path,
        model.internals.control_stream,
        name_map,
        model,
        strict=strict,
        subproblem=subproblem,
        with_log=with_log,
        lazy=lazy,
    )
    return res


def _parse_table_file(model, path: Optional[Union[str, Path]], subproblem: Optional[int] = None):
    table_recs = model.internals.control_stream.get_records('TABLE')
    df = pd.DataFrame()
    for table_rec in table_recs:
        noheader = table_rec.has_option("NOHEADER")
        notitle = table_rec.has_option("NOTITLE") or noheader
        nolabel = table_rec.has_option("NOLABEL") or noheader
        table_path = path.parent / table_rec.path
        try:
            table_file = NONMEMTableFile(table_path, notitle=notitle, nolabel=nolabel)
        except IOError:
            continue
        for i in range(len(table_file)):
            sim = table_file.tables[i].data_frame[['DV']].copy()
            assert 'DV' in sim.columns
            sim['SIM'] = i + 1
            sim['index'] = np.arange(len(model.dataset))
            df = pd.concat([df, sim], ignore_index=True)

    if not df.empty:
        df = df.set_index(['SIM', 'index'])
    return df


def parse_simulation_results(
    model, path: Optional[Union[str, Path]], subproblem: Optional[int] = None
):
    table = _parse_table_file(model, path=path, subproblem=subproblem)
    res = SimulationResults(table=table)
    return res


def override_minimization_successful(minimization_successful, pe_iterations):
    # NONMEM could return infinity as parameter estimate even if minimization
    # was successful. We set minimization successful to False in these cases.
    # This reduces the need for special cases further downstream.

    new_minsucc = []
    for i, minsucc in enumerate(minimization_successful):
        try:
            ests_for_iteration = pe_iterations.loc[i + 1].iloc[-1]
        except KeyError:
            new_minsucc.append(minsucc)
            continue
        have_inf = np.isinf(ests_for_iteration).any()
        if have_inf:
            new_minsucc.append(False)
        else:
            new_minsucc.append(minsucc)
    return new_minsucc


def create_failed_results(model, log):
    ests = _create_failed_parameter_estimates(model.parameters)
    return ModelfitResults(
        minimization_successful=False,
        ofv=float("NaN"),
        parameter_estimates=ests,
        parameter_estimates_sdcorr=ests,
        standard_errors=ests.rename("SE"),
        standard_errors_sdcorr=ests.rename("SE_sdcorr"),
        log=log,
        significant_digits=np.nan,
    )
