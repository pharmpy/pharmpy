"""

.. list-table:: Options for the results module
   :widths: 25 25 50 150
   :header-rows: 1

   * - Option name
     - Default value
     - Type
     - Description
   * - ``native_shrinkage``
     - ``True``
     - bool
     - Should shrinkage calculation of external tool be used.
       Otherwise pharmpy will calculate shrinkage
"""

import copy
import json
import lzma
import math
import warnings
from collections.abc import MutableSequence
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import sympy

import pharmpy.config as config
import pharmpy.visualization
from pharmpy.data import PharmDataFrame
from pharmpy.data_structures import OrderedSet
from pharmpy.math import cov2corr
from pharmpy.parameter_sampling import sample_from_covariance_matrix


class ResultsConfiguration(config.Configuration):
    module = 'pharmpy.results'
    """ FIXME: This setting should be moved to the NONMEM plugin
    """
    native_shrinkage = config.ConfigItem(True, 'Use shrinkage results from external tool')


conf = ResultsConfiguration()


class ResultsJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            d = json.loads(obj.to_json(orient='table'))
            if isinstance(obj, pd.DataFrame):
                d['__class__'] = 'DataFrame'
            elif isinstance(obj, pd.Series):
                d['__class__'] = 'Series'
            else:
                d['__class__'] = obj.__class__.__name__
            return d
        elif obj.__class__.__module__.startswith('altair.'):
            d = obj.to_dict()
            d['__class__'] = 'vega-lite'
            return d
        else:
            return json.JSONEncoder.encode(self, obj)


class ResultsJSONDecoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
        self.cls = None

    def object_hook(self, dct):
        if '__class__' in dct:
            cls = dct['__class__']
            del dct['__class__']
            if cls == 'DataFrame' or cls == 'Series':
                res = pd.read_json(json.dumps(dct), orient='table')
                return res
            elif cls == 'vega-lite':
                res = alt.Chart.from_dict(dct)
                return res
            else:
                self.cls = cls
        return dct


def read_results(path_or_buf):
    if '{' in str(path_or_buf):  # Heuristic to determine if path or buffer
        s = path_or_buf
    else:
        path = Path(path_or_buf)
        if path.is_dir():
            path /= 'results.json'
        if not path.is_file():
            raise FileNotFoundError(str(path))
        if path.name.endswith('.xz'):
            with lzma.open(path, 'r') as json_file:
                s = json_file.read().decode('utf-8')
        else:
            with open(path, 'r') as json_file:
                s = json_file.read()
    decoder = ResultsJSONDecoder()
    d = decoder.decode(s)
    if decoder.cls == 'FREMResults':
        from pharmpy.tools.frem import FREMResults

        res = FREMResults.from_dict(d)
    elif decoder.cls == 'BootstrapResults':
        from pharmpy.tools.bootstrap import BootstrapResults

        res = BootstrapResults.from_dict(d)
    elif decoder.cls == 'CDDResults':
        from pharmpy.tools.cdd import CDDResults

        res = CDDResults.from_dict(d)
    elif decoder.cls == 'SCMResults':
        from pharmpy.tools.scm import SCMResults

        res = SCMResults.from_dict(d)
    elif decoder.cls == 'QAResults':
        from pharmpy.tools.qa import QAResults

        res = QAResults.from_dict(d)
    elif decoder.cls == 'LinearizeResults':
        from pharmpy.tools.linearize import LinearizeResults

        res = LinearizeResults.from_dict(d)
    elif decoder.cls == 'ResmodResults':
        from pharmpy.tools.resmod import ResmodResults

        res = ResmodResults.from_dict(d)
    elif decoder.cls == 'SimevalResults':
        from pharmpy.tools.simeval import SimevalResults

        res = SimevalResults.from_dict(d)
    elif decoder.cls == 'CrossvalResults':
        from pharmpy.tools.crossval import CrossvalResults

        res = CrossvalResults.from_dict(d)

    return res


class Results:
    """Base class for all result classes"""

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_json(self, path=None, lzma=False):
        json_dict = self.to_dict()
        json_dict['__class__'] = self.__class__.__name__
        s = json.dumps(json_dict, cls=ResultsJSONEncoder)
        if path:
            if not lzma:
                with open(path, 'w') as fh:
                    fh.write(s)
            else:
                xz_path = path.parent / (path.name + '.xz')
                with lzma.open(xz_path, 'w') as fh:
                    fh.write(bytes(s, 'utf-8'))
        else:
            return s

    def get_and_reset_index(self, attr, **kwargs):
        """Wrapper to reset index of attribute or result from method.

        Used to facilitate importing multiindex dataframes into R
        """
        val = getattr(self, attr)
        if callable(val):
            df = val(**kwargs)
        else:
            df = val
        return df.reset_index()

    def to_dict(self):
        """Convert results object to a dictionary"""
        return vars(self).copy()

    def __str__(self):
        start = self.__class__.__name__
        s = f'{start}\n\n'
        d = self.to_dict()
        for key, value in d.items():
            if value.__class__.__module__.startswith('altair.'):
                continue
            s += f'{key}\n'
            if isinstance(value, pd.DataFrame):
                s += value.to_string()
            elif isinstance(value, list):  # Print list of lists as table
                if len(value) > 0 and isinstance(value[0], list):
                    df = pd.DataFrame(value)
                    df_str = df.to_string(index=False)
                    df_str = df_str.split('\n')[1:]
                    s += '\n'.join(df_str)
            else:
                s += str(value) + '\n'
            s += '\n\n'
        return s

    def to_csv(self, path):
        """Save results as a human readable csv file

        Index will not be printed if it is a basic range.
        """
        d = self.to_dict()
        s = ""
        for key, value in d.items():
            if value.__class__.__module__.startswith('altair.'):
                continue
            s += f'{key}\n'
            if isinstance(value, pd.DataFrame):
                if isinstance(value.index, pd.RangeIndex):
                    use_index = False
                else:
                    use_index = True
                s += value.to_csv(index=use_index)
            elif isinstance(value, pd.Series):
                s += value.to_csv()
            elif isinstance(value, list):  # Print list of lists as table
                if len(value) > 0 and isinstance(value[0], list):
                    for row in value:
                        s += f'{",".join(map(str, row))}\n'
            else:
                s += str(value) + '\n'
            s += '\n'
        with open(path, 'w', newline='') as fh:
            print(s, file=fh)

    def create_report(self, path):
        import pharmpy.reporting.reporting as reporting

        reporting.generate_report(self.rst_path, path)

    def add_plots(self):
        """Create and add all plots to results object"""
        raise NotImplementedError()


class ModelfitResults(Results):
    """Base class for results from a modelfit operation

    properties: individual_OFV is a df with currently ID and iOFV columns
        model_name - name of model that generated the results
        model
    """

    def __init__(
        self,
        ofv=None,
        parameter_estimates=None,
        covariance_matrix=None,
        standard_errors=None,
        minimization_successful=None,
        individual_ofv=None,
        individual_estimates=None,
    ):
        self._ofv = ofv
        self._parameter_estimates = parameter_estimates
        self._covariance_matrix = covariance_matrix
        self._standard_errors = standard_errors
        self._minimization_successful = minimization_successful
        self._individual_estimates = individual_estimates
        self._individual_ofv = individual_ofv

    def to_dict(self):
        """Convert results object to a dictionary"""
        return {'parameter_estimates': self.parameter_estimates}

    @property
    def ofv(self):
        """Final objective function value"""
        return self._ofv

    @property
    def aic(self):
        """Final AIC value assuming the OFV to be -2LL"""
        parameters = self.model.parameters.copy()
        parameters.remove_fixed()
        return self.ofv + 2 * len(parameters)

    @property
    def bic(self):
        """Final BIC value assuming the OFV to be -2LL"""
        parameters = self.model.parameters.copy()
        parameters.remove_fixed()
        return self.ofv + len(parameters) * math.log(len(self.model.dataset.pharmpy.observations))

    @property
    def evaluation_ofv(self):
        """The ofv as if the model was evaulated

        Currently works for classical estimation methods by taking the OFV of the
        first iteration.
        """
        return self._evaluation_ofv

    @property
    def minimization_successful(self):
        """Was the minimization successful"""
        return self._minimization_successful

    @property
    def parameter_estimates(self):
        """Parameter estimates as series"""
        return self._parameter_estimates

    @parameter_estimates.setter
    def parameter_estimates(self, value):
        self._parameter_estimates = value

    @property
    def covariance_matrix(self):
        """The covariance matrix of the population parameter estimates"""
        return self._covariance_matrix

    def _cov_from_inf(self):
        Im = self.information_matrix
        cov = pd.DataFrame(np.linalg.inv(Im.values), index=Im.index, columns=Im.columns)
        return cov

    def _cov_from_corrse(self):
        se = self.standard_errors
        corr = self.correlation_matrix
        x = se.values @ se.values.T
        cov = x * corr.values
        cov_df = pd.DataFrame(cov, index=corr.index, columns=corr.columns)
        return cov_df

    @property
    def information_matrix(self):
        """The Fischer information matrix of the population parameter estimates"""
        raise NotImplementedError()

    def _inf_from_cov(self):
        C = self.covariance_matrix
        Im = pd.DataFrame(np.linalg.inv(C.values), index=C.index, columns=C.columns)
        return Im

    def _inf_from_corrse(self):
        se = self.standard_errors
        corr = self.correlation_matrix
        x = se.values @ se.values.T
        cov = x * corr.values
        Im = pd.DataFrame(np.linalg.inv(cov), index=corr.index, columns=corr.columns)
        return Im

    @property
    def correlation_matrix(self):
        """The correlation matrix of the population parameter estimates"""
        raise NotImplementedError()

    def _corr_from_cov(self):
        C = self.covariance_matrix
        corr = pd.DataFrame(cov2corr(C.values), index=C.index, columns=C.columns)
        return corr

    def _corr_from_inf(self):
        Im = self.information_matrix
        corr = pd.DataFrame(cov2corr(np.linalg.inv(Im.values)), index=Im.index, columns=Im.columns)
        return corr

    @property
    def standard_errors(self):
        """Standard errors of population parameter estimates"""
        return self._standard_errors

    @property
    def relative_standard_errors(self):
        """Relative standard errors of population parameter estimates"""
        if self.standard_errors is not None:
            ser = self.standard_errors / self.parameter_estimates
            ser.name = 'RSE'
            return ser

    def _se_from_cov(self):
        """Calculate the standard errors from the covariance matrix
        can be used by subclasses
        """
        cov = self.covariance_matrix
        se = pd.Series(np.sqrt(np.diag(cov.values)), index=cov.index)
        return se

    def _se_from_inf(self):
        """Calculate the standard errors from the information matrix"""
        Im = self.information_matrix
        se = pd.Series(np.sqrt(np.linalg.inv(Im.values)), index=Im.index)
        return se

    @property
    def individual_ofv(self):
        """A Series with individual estimates indexed over ID"""
        return self._individual_ofv

    @property
    def individual_estimates(self):
        """Individual parameter estimates

        A DataFrame with ID as index one column for each individual parameter
        """
        return self._individual_estimates

    @property
    def individual_estimates_covariance(self):
        """The covariance matrix of the individual estimates"""
        raise NotImplementedError("Not implemented")

    def eta_shrinkage(self, sd=False):
        """Eta shrinkage for each eta

        Variance = False to get sd scale
        """
        pe = self.parameter_estimates
        # Want parameter estimates combined with fixed parameter values
        param_inits = self.model.parameters.to_dataframe()['value']
        pe = pe.combine_first(param_inits)

        ie = self.individual_estimates
        param_names = self.model.random_variables.iiv.variance_parameters
        diag_ests = pe[param_names]
        diag_ests.index = ie.columns
        if not sd:
            shrinkage = 1 - (ie.var() / diag_ests)
        else:
            shrinkage = 1 - (ie.std() / (diag_ests ** 0.5))
        return shrinkage

    @property
    def individual_shrinkage(self):
        """The individual eta-shrinkage

        Definition: ieta_shr = (var(eta) / omega)
        """
        cov = self.individual_estimates_covariance
        pe = self.parameter_estimates
        # Want parameter estimates combined with fixed parameter values
        param_inits = self.model.parameters.to_dataframe()['value']
        pe = pe.combine_first(param_inits)

        # Get all iiv variance parameters
        param_names = self.model.random_variables.etas.variance_parameters
        param_names = list(OrderedSet(param_names))  # Only unique in order

        diag_ests = pe[param_names]

        def fn(row, ests):
            names = row[0].index
            ser = pd.Series(np.diag(row[0].values) / ests, index=names)
            return ser

        ish = pd.DataFrame(cov).apply(fn, axis=1, ests=diag_ests.values)
        return ish

    def near_bounds(self, zero_limit=0.001, significant_digits=2):
        return self.model.parameters.is_close_to_bound(
            values=self.parameter_estimates,
            zero_limit=zero_limit,
            significant_digits=significant_digits,
        )

    def parameter_summary(self):
        """Summary of parameter estimates and uncertainty"""
        pe = self.parameter_estimates
        ses = self.standard_errors
        rses = self.relative_standard_errors
        df = pd.DataFrame({'estimate': pe, 'SE': ses, 'RSE': rses})
        return df

    def plot_iofv_vs_iofv(self, other):
        x_label = f'{self.model_name} iOFV'
        y_label = f'{other.model_name} iOFV'
        df = PharmDataFrame({x_label: self.individual_ofv, y_label: other.individual_ofv})
        plot = pharmpy.visualization.scatter_plot_correlation(
            df, x_label, y_label, title='iOFV vs iOFV'
        )
        return plot

    def plot_individual_predictions(self, predictions=None, individuals=None):
        """Plot DV and predictions grouped on individuals

        Parameters
        ----------
        predictions : list
            A list of names of predictions to plot. None for all available
        individuals: list
            A list of individuals to include. None for all individuals
        """
        pred = self.predictions
        obs = self.model.dataset.pharmpy.observations
        indexcols = pred.index.names
        idcol = indexcols[0]
        idvcol = indexcols[1]

        data = pred.join(obs).reset_index()
        data = data.melt(id_vars=indexcols)

        if individuals is not None:
            data = data[data[idcol].isin(individuals)]
        if predictions is not None:
            dvcol = obs.name
            data = data[data['variable'].isin(predictions + [dvcol])]

        plot = (
            alt.Chart(data)
            .mark_line(point=True)
            .encode(x=idvcol, y='value', color='variable')
            .facet(f'{idcol}:N', columns=5)
        )
        return plot

    def individual_parameter_statistics(self, exprs, seed=None):
        """Calculate statistics for individual parameters

        exprs - is one string or an iterable of strings

        The parameter does not have to be in the model, but can be an
        expression of other parameters from the model.
        Does not support parameters that relies on the solution of the ODE-system
        """
        if isinstance(exprs, str) or isinstance(exprs, sympy.Basic):
            exprs = [_split_equation(exprs)]
        else:
            exprs = [_split_equation(expr) for expr in exprs]
        model = self.model
        dataset = model.dataset
        cols = set(dataset.columns)
        i = 0
        table = pd.DataFrame(columns=['parameter', 'covariates', 'mean', 'variance', 'stderr'])
        for name, expr in exprs:
            full_expr = model.statements.full_expression_from_odes(expr)
            covariates = {symb.name for symb in full_expr.free_symbols if symb.name in cols}
            if not covariates:
                cases = {'median': dict()}
            else:
                q5 = dataset[{'ID'} | covariates].groupby('ID').median().quantile(0.05)
                q95 = dataset[{'ID'} | covariates].groupby('ID').median().quantile(0.95)
                median = dataset[{'ID'} | covariates].groupby('ID').median().median()
                cases = {'p5': dict(q5), 'median': dict(median), 'p95': dict(q95)}

            df = pd.DataFrame(index=list(cases.keys()), columns=['mean', 'variance', 'stderr'])
            for case, cov_values in cases.items():
                pe = dict(model.modelfit_results.parameter_estimates)
                cov_expr = full_expr.subs(cov_values)
                expr = cov_expr.subs(pe)
                samples = model.random_variables.sample(
                    expr, parameters=pe, samples=1000000, seed=seed
                )

                mean = np.mean(samples)
                variance = np.var(samples)

                parameters = sample_from_covariance_matrix(
                    model,
                    n=100,
                    force_posdef_covmatrix=True,
                    seed=seed,
                )
                samples = []
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    for _, row in parameters.iterrows():
                        batch = model.random_variables.sample(
                            cov_expr.subs(dict(row)),
                            parameters=dict(row),
                            samples=10,
                            seed=seed,
                        )
                        samples.extend(list(batch))
                stderr = pd.Series(samples).std()
                df.loc[case] = [mean, variance, stderr]
                df.index.name = 'covariates'

            df.reset_index(inplace=True)
            if not name:
                name = f'unknown{i}'
                i += 1
            df['parameter'] = name
            table = pd.concat([table, df])
        table.set_index(['parameter', 'covariates'], inplace=True)
        return table

    def pk_parameters(self, seed=None):
        statements = self.model.statements
        odes = statements.ode_system
        central = odes.find_central()
        output = odes.find_output()
        depot = odes.find_depot(statements)
        peripherals = odes.find_peripherals()
        elimination_rate = odes.get_flow(central, output)

        expressions = []  # Eq(name, expr)
        # FO abs + 1comp + FO elimination
        if len(odes) == 3 and depot and odes.t not in elimination_rate.free_symbols:
            ode_list, ics = odes.to_explicit_odes(skip_output=True)
            sols = sympy.dsolve(ode_list, ics=ics)
            expr = sols[1].rhs
            d = sympy.diff(expr, odes.t)
            tmax_closed_form = sympy.solve(d, odes.t)[0]
            expressions.append(sympy.Eq(sympy.Symbol('t_max'), tmax_closed_form))
            e2 = sympy.simplify(expr / depot.dose.amount / sympy.denom(elimination_rate))
            cmax_dose_closed_form = sympy.simplify(e2.subs({odes.t: tmax_closed_form}))
            expressions.append(sympy.Eq(sympy.Symbol('C_max_dose'), cmax_dose_closed_form))

        # Any abs + 1comp + FO elimination
        if not peripherals and odes.t not in elimination_rate.free_symbols:
            elimination_system = statements.copy().ode_system
            # keep central and output
            for name in elimination_system.names:
                if name not in [central.name, output.name]:
                    elimination_system.remove_compartment(elimination_system.find_compartment(name))
                    ode_list, ics = elimination_system.to_explicit_odes(skip_output=True)
                    A0 = sympy.Symbol('A0')
                    ic = ics.popitem()
                    ics = {ic[0]: A0}
                    sols = sympy.dsolve(ode_list[0], ics=ics)
                    eq = sympy.Eq(sympy.Rational(1, 2) * A0, sols.rhs)
                    thalf_elim = sympy.solve(eq, odes.t)[0]
                    expressions.append(sympy.Eq(sympy.Symbol('t_half_elim'), thalf_elim))

        # Any abs + any comp + FO elimination
        if odes.t not in elimination_rate.free_symbols:
            expressions.append(sympy.Eq(sympy.Symbol('k_e'), elimination_rate))

        df = self.individual_parameter_statistics(expressions, seed=seed)
        return df

    def __repr__(self):
        df = self.parameter_summary()
        return df.to_string()


class ChainedModelfitResults(MutableSequence, ModelfitResults):
    """A sequence of modelfit results given in order from first to final
    inherits from both list and ModelfitResults. Each method from ModelfitResults
    will be performed on the final modelfit object
    """

    def __init__(self, results=None):
        if isinstance(results, ChainedModelfitResults):
            self._results = copy.deepcopy(results._results)
        elif results is None:
            self._results = []
        else:
            self._results = list(results)

    def __getitem__(self, ind):
        return self._results[ind]

    def __setitem__(self, ind, value):
        self._results[ind] = value

    def __delitem__(self, ind):
        del self._results[ind]

    def __len__(self):
        return len(self._results)

    def insert(self, ind, value):
        self._results.insert(ind, value)

    @property
    def ofv(self):
        return self[-1].ofv

    @property
    def evaluation_ofv(self):
        return self[0].evaluation_ofv

    @property
    def minimization_successful(self):
        return self[-1].minimization_successful

    @property
    def parameter_estimates(self):
        return self[-1].parameter_estimates

    @parameter_estimates.setter
    def parameter_estimates(self, value):
        self[-1].parameter_estimates = value

    @property
    def parameter_estimates_sdcorr(self):
        return self[-1].parameter_estimates_sdcorr

    @property
    def covariance_matrix(self):
        return self[-1].covariance_matrix

    @property
    def information_matrix(self):
        return self[-1].information_matrix

    @property
    def correlation_matrix(self):
        return self[-1].correlation_matrix

    @property
    def standard_errors(self):
        return self[-1].standard_errors

    @property
    def standard_errors_sdcorr(self):
        return self[-1].standard_errors_sdcorr

    @property
    def individual_ofv(self):
        return self[-1].individual_ofv

    @property
    def individual_estimates(self):
        return self[-1].individual_estimates

    @property
    def individual_estimates_covariance(self):
        return self[-1].individual_estimates_covariance

    @property
    def individual_shrinkage(self):
        return self[-1].individual_shrinkage

    @property
    def residuals(self):
        return self[-1].residuals

    @property
    def predictions(self):
        return self[-1].predictions

    def plot_iofv_vs_iofv(self, other):
        return self[-1].plot_iofv_vs_iofv(other)

    @property
    def model_name(self):
        return self[-1].model_name

    def individual_parameter_statistics(self, expr, seed=None):
        return self[-1].individual_parameter_statistics(expr, seed=seed)

    def pk_parameters(self, seed=None):
        return self[-1].pk_parameters(seed=seed)

    def __repr__(self):
        return repr(self._results[-1])

    # FIXME: To not have to manually intercept everything here. Could do it in a general way.


def _split_equation(s):
    if isinstance(s, str):
        a = s.split('=')
        if len(a) == 1:
            name = None
            expr = sympy.sympify(s)
        else:
            name = a[0].strip()
            expr = sympy.sympify(a[1])
    elif isinstance(s, sympy.Eq):
        name = s.lhs.name
        expr = s.rhs
    else:  # sympy expr
        name = None
        expr = s
    if name is None and isinstance(expr, sympy.Symbol):
        name = expr.name
    return name, expr
