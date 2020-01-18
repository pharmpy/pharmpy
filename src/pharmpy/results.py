# Classes for method results

import altair as alt
import pandas as pd

import pharmpy.visualization
from pharmpy.data import PharmDataFrame


class ModelfitResults:
    """ Results from a modelfit operation

    properties: individual_OFV is a df with currently ID and iOFV columns
        model_name - name of model that generated the results
    """
    @property
    def ofv(self):
        """Final objective function value
        """
        raise NotImplementedError("Not implemented")

    @property
    def parameter_estimates(self):
        """Parameter estimates as series
        """
        raise NotImplementedError("Not implemented")

    @property
    def covariance_matrix(self):
        """The covariance matrix of the population parameter estimates
        """
        raise NotImplementedError("Not implemented")

    @property
    def information_matrix(self):
        """The Fischer information matrix of the population parameter estimates
        """
        raise NotImplementedError("Not implemented")

    @property
    def individual_OFV(self):
        """A Series with individual estimates indexed over ID
        """
        raise NotImplementedError("Not implemented")

    def plot_iofv_vs_iofv(self, other):
        x_label = f'{self.model_name} iOFV'
        y_label = f'{other.model_name} iOFV'
        df = PharmDataFrame({x_label: self.individual_OFV, y_label: other.individual_OFV})
        plot = pharmpy.visualization.scatter_plot_correlation(df, x_label, y_label,
                                                              title='iOFV vs iOFV')
        return plot


class ChainedModelfitResults(list, ModelfitResults):
    """A list of modelfit results given in order from first to final
       inherits from both list and ModelfitResults. Each method from ModelfitResults
       will be performed on the final modelfit object
    """
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
    def individual_OFV(self):
        return self[-1].individual_OFV

    def plot_iOFV_vs_iOFV(self, other):
        return self[-1].plot_iOFV_vs_iOFV(other)

    @property
    def model_name(self):
        return self[-1].model_name

    # FIXME: To not have to manually intercept everything here. Should do it in a general way.


class CaseDeletionResults:
    def __init__(self, original_fit, case_deleted_fits):
        # Would also need group numbers. For dOFV can only do ID as group
        pass

    @property
    def cook_scores(self):
        """ Calculate a series of cook scores. One for each group
        """
        pass


class BootstrapResults:
    def __init__(self, original_result, bootstrap_results):
        self._original_result = original_result
        self._bootstrap_results = bootstrap_results

    def plot_ofv(self):
        boot_ofvs = [x.ofv for x in self._bootstrap_results]
        df = pd.DataFrame({'ofv': boot_ofvs, 'num': list(range(1, len(boot_ofvs) + 1))})

        slider = alt.binding_range(min=1, max=len(boot_ofvs), step=1, name='Number of samples: ')
        selection = alt.selection_single(bind=slider, fields=['num'], name="num",
                                         init={'num': len(boot_ofvs)})

        base = alt.Chart(df).transform_filter('datum.num <= num_num')

        plot = base.transform_joinaggregate(
            total='count(*)'
        ).transform_calculate(
            pct='1 / datum.total'
        ).mark_bar().encode(
            alt.X("ofv:Q", bin=True),
            alt.Y('sum(pct):Q', axis=alt.Axis(format='%'))
        ).add_selection(
            selection
        ).properties(
            title="Bootstrap OFV"
        )

        rule = base.mark_rule(color='red').encode(
            x='mean(ofv):Q',
            size=alt.value(5)
        )

        return plot + rule
