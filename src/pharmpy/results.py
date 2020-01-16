# Classes for method results


class ModelfitResults:
    """ Results from a modelfit operation

    properties: individual_estimates is a df with currently ID and iOFV columns
    """
    @property
    def individual_OFV(self):
        """A Series with individual estimates indexed over ID
        """
        raise NotImplementedError("Not implemented")


class ChainedModelfitResults(list, ModelfitResults):
    """A list of modelfit results given in order from first to final
       inherits from both list and ModelfitResults. Each method from ModelfitResults
       will be performed on the final modelfit object
    """
    @property
    def individual_OFV(self):
        return self[-1].individual_OFV
