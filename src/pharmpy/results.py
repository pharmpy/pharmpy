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

import json
from pathlib import Path

import numpy as np
import pandas as pd

import pharmpy.config as config
import pharmpy.visualization
from pharmpy.data import PharmDataFrame
from pharmpy.math import cov2corr


class ResultsConfiguration(config.Configuration):
    native_shrinkage = config.ConfigItem(True, 'Use shrinkage results from external tool')


conf = ResultsConfiguration()


class ResultsJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame) or isinstance(obj, pd.Series):
            d = json.loads(obj.to_json(orient='split'))
            d['__class__'] = obj.__class__.__name__
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
                res = pd.read_json(json.dumps(dct), orient='split')
                return res
            else:
                self.cls = cls
        return dct


def read_results(path_or_buf):
    try:
        path = Path(path_or_buf)
        if not path.is_file():
            raise FileNotFoundError
    except (FileNotFoundError, OSError, TypeError, ValueError):
        s = path_or_buf
    else:
        with open(path, 'r') as json_file:
            s = json_file.read()
    decoder = ResultsJSONDecoder()
    d = decoder.decode(s)
    if decoder.cls == 'FREMResults':
        from pharmpy.methods.frem.results import FREMResults
        res = FREMResults.from_dict(d)
    elif decoder.cls == 'BootstrapResults':
        res = BootstrapResults(original_model=None, bootstrap_models=None)
        res._statistics = d['statistics']

    return res


class Results:
    """ Base class for all result classes
    """
    def to_json(self, path=None):
        json_dict = self.to_dict()
        json_dict['__class__'] = self.__class__.__name__
        s = json.dumps(json_dict, cls=ResultsJSONEncoder)
        if path:
            with open(path, 'w') as fh:
                fh.write(s)
        else:
            return s

    def to_dict(self):
        raise NotImplementedError()

    def to_csv(self, path):
        """Save results as a human readable csv file

           Index will not be printed if it is a basic range.
        """
        d = self.to_dict()
        s = ""
        for key, value in d.items():
            s += f'{key}\n'
            if isinstance(value, pd.DataFrame):
                if isinstance(value.index, pd.RangeIndex):
                    use_index = False
                else:
                    use_index = True
                s += value.to_csv(index=use_index)
            else:
                s += str(value)
            s += '\n'
        with open(path, 'w') as fh:
            print(s, file=fh)


class ModelfitResults:
    """ Base class for results from a modelfit operation

    properties: individual_OFV is a df with currently ID and iOFV columns
        model_name - name of model that generated the results
        model
    """
    def __init__(self, ofv=None, parameter_estimates=None, covariance_matrix=None,
                 standard_errors=None):
        self._ofv = ofv
        self._parameter_estimates = parameter_estimates
        self._covariance_matrix = covariance_matrix
        self._standard_errors = standard_errors

    def reparameterize(self, parameterizations):
        """Reparametrize all parameters given a list of parametrization object
           will change the parameter_estimates and standard_errors to be for
           the transformed parameter
        """
        raise NotImplementedError("Not implemented")

    @property
    def ofv(self):
        """Final objective function value
        """
        return self._ofv

    @property
    def parameter_estimates(self):
        """Parameter estimates as series
        """
        return self._parameter_estimates

    @property
    def covariance_matrix(self):
        """The covariance matrix of the population parameter estimates
        """
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
        """The Fischer information matrix of the population parameter estimates
        """
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
        """The correlation matrix of the population parameter estimates
        """
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
        """Standard errors of population parameter estimates
        """
        return self._standard_errors

    @property
    def relative_standard_errors(self):
        """Relative standard errors of popilation parameter estimates
        """
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
        """Calculate the standard errors from the information matrix
        """
        Im = self.information_matrix
        se = pd.Series(np.sqrt(np.linalg.inv(Im.values)), index=Im.index)
        return se

    @property
    def individual_ofv(self):
        """A Series with individual estimates indexed over ID
        """
        raise NotImplementedError("Not implemented")

    @property
    def individual_estimates(self):
        """Individual parameter estimates

           A DataFrame with ID as index one column for each individual parameter
        """
        raise NotImplementedError("Not implemented")

    @property
    def individual_estimates_covariance(self):
        """The covariance matrix of the individual estimates
        """
        raise NotImplementedError("Not implemented")

    def eta_shrinkage(self, sd=False):
        """Eta shrinkage for each eta

           Variance = False to get sd scale
        """
        pe = self.parameter_estimates
        # Want parameter estimates combined with fixed parameter values
        param_inits = self.model.parameters.summary()['value']
        pe = pe.combine_first(param_inits)

        ie = self.individual_estimates
        parameters = self.model.random_variables.iiv_variance_parameters()
        param_names = [param.name for param in parameters]
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
        param_inits = self.model.parameters.summary()['value']
        pe = pe.combine_first(param_inits)

        # Get all iiv variance parameters
        parameters = self.model.random_variables.iiv_variance_parameters()
        param_names = [param.name for param in parameters]
        diag_ests = pe[param_names]

        def fn(row, ests):
            names = row[0].index
            ser = pd.Series(np.diag(row[0].values) / ests, index=names)
            return ser

        ish = pd.DataFrame(cov).apply(fn, axis=1, ests=diag_ests.values)
        return ish

    def parameter_summary(self):
        """Summary of parameter estimates and uncertainty
        """
        pe = self.parameter_estimates
        ses = self.standard_errors
        rses = self.relative_standard_errors
        df = pd.DataFrame({'estimate': pe, 'SE': ses, 'RSE': rses})
        return df

    def plot_iofv_vs_iofv(self, other):
        x_label = f'{self.model_name} iOFV'
        y_label = f'{other.model_name} iOFV'
        df = PharmDataFrame({x_label: self.individual_ofv, y_label: other.individual_ofv})
        plot = pharmpy.visualization.scatter_plot_correlation(df, x_label, y_label,
                                                              title='iOFV vs iOFV')
        return plot


class ChainedModelfitResults(list, ModelfitResults):
    """A list of modelfit results given in order from first to final
       inherits from both list and ModelfitResults. Each method from ModelfitResults
       will be performed on the final modelfit object
    """
    def reparameterize(self, parameterizations):
        return self[-1].reparameterize(parameterizations)

    @property
    def ofv(self):
        return self[-1].ofv

    @property
    def parameter_estimates(self):
        return self[-1].parameter_estimates

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

    def plot_iofv_vs_iofv(self, other):
        return self[-1].plot_iofv_vs_iofv(other)

    @property
    def model_name(self):
        return self[-1].model_name

    # FIXME: To not have to manually intercept everything here. Could do it in a general way.


class CaseDeletionResults:
    def __init__(self, original_fit, case_deleted_fits):
        # Would also need group numbers. For dOFV can only do ID as group
        pass

    @property
    def cook_scores(self):
        """ Calculate a series of cook scores. One for each group
        """
        pass


class BootstrapResults(Results):
    # FIXME: Could inherit from results that take multiple runs like bootstrap, cdd etc.
    def __init__(self, original_model, bootstrap_models):
        if original_model is None and bootstrap_models is None:
            # FIXME: this is a special case for now for json handling before we have modelfit json
            return
        self._original_results = original_model.modelfit_results
        self._bootstrap_results = [m.modelfit_results for m in bootstrap_models
                                   if m.modelfit_results is not None]
        self._total_number_of_models = len(bootstrap_models)

    @property
    def ofv(self):
        boot_ofvs = [x.ofv for x in self._bootstrap_results]
        return pd.Series(boot_ofvs, name='ofv')

    @property
    def parameter_estimates(self):
        df = pd.DataFrame()
        for res in self._bootstrap_results:
            df = df.append(res.parameter_estimates, ignore_index=True, sort=False)
        df = df.reindex(self._bootstrap_results[0].parameter_estimates.index, axis=1)
        df = df.reset_index(drop=True)
        return df

    @property
    def standard_errors(self):
        pass
        # FIXME: Continue here

    @property
    def statistics(self):
        try:
            return self._statistics
        except AttributeError:
            pass
        df = self.parameter_estimates
        ofvs = self.ofv
        df.insert(0, 'OFV', ofvs)
        orig = self._original_results.parameter_estimates
        ofv_ser = pd.Series({'OFV': self._original_results.ofv})
        orig = pd.concat([ofv_ser, orig])
        mean = df.mean()
        bias = mean - orig
        summary = pd.DataFrame({'mean': mean, 'bias': bias, 'stderr': df.std()})
        self._statistics = summary
        return summary

    @property
    def distribution(self):
        try:
            return self._distribution
        except AttributeError:
            pass
        df = self.parameter_estimates
        ofvs = self.ofv
        df.insert(0, 'OFV', ofvs)
        summary = pd.DataFrame({'min': df.min(), '0.05%': df.quantile(0.0005),
                                '0.5%': df.quantile(0.005), '2.5%': df.quantile(0.025),
                                '5%': df.quantile(0.05), 'median': df.median(),
                                '95%': df.quantile(0.95), '97.5%': df.quantile(0.975),
                                '99.5%': df.quantile(0.995), '99.95%': df.quantile(0.9995),
                                'max': df.max()})
        self._distribution = summary
        return summary

    def __repr__(self):
        inclusions = f'Inclusion\n\nTotal number of models: {self._total_number_of_models}\n' \
                     f'Models not included because of failure: ' \
                     f'{self._total_number_of_models - len(self._bootstrap_results)}\n' \
                     f'Models included in analysis: {len(self._bootstrap_results)}'
        statistics = f'Statistics\n{repr(self.statistics)}'
        distribution = f'Distribution\n{repr(self.distribution)}'
        return f'{inclusions}\n\n{statistics}\n\n{distribution}'

    def to_dict(self):
        return {'statistics': self.statistics}

    def plot_ofv(self):
        plot = pharmpy.visualization.histogram(self.ofv, title='Bootstrap OFV')
        return plot
