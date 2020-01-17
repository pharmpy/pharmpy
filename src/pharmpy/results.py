# Classes for method results

import pharmpy.visualization
from pharmpy.data import PharmDataFrame


class ModelfitResults:
    """ Results from a modelfit operation

    properties: individual_OFV is a df with currently ID and iOFV columns
        model_name - name of model that generated the results
    """
    @property
    def covariance_matrix(self):
        """The covariance matrix of the population parameter estimates
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
    def covariance_matrix(self):
        return self[-1].covariance_matrix

    @property
    def individual_OFV(self):
        return self[-1].individual_OFV

    def plot_iOFV_vs_iOFV(self, other):
        return self[-1].plot_iOFV_vs_iOFV(other)

    @property
    def model_name(self):
        return self[-1].model_name

    # FIXME: To not have to manually intercept everything here. Should do it in a general way.
