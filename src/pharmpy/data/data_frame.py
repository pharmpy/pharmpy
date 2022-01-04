import pandas as pd


class DatasetError(Exception):
    pass


class DatasetWarning(Warning):
    pass


class PharmDataFrame(pd.DataFrame):
    """A DataFrame with additional metadata.

    ============  =============
    ColumnType    Description
    ============  =============
    ID            Individual identifier. Max one per DataFrame. All values have to be unique
    IDV           Independent variable. Max one per DataFrame.
    DV            Dependent variable
    COVARIATE     Covariate
    DOSE          Dose amount
    EVENT         0 = observation
    UNKOWN        Unkown type. This will be the default for columns that hasn't been assigned a type
    ============  =============

    """

    @property
    def _constructor(self):
        return PharmDataFrame

    @property
    def _constructor_sliced(self):
        return pd.Series

    def __deepcopy__(self, memo):
        return self.copy()

    def to_json(self, **kwargs):
        """ """
        # FIXME: Same docstring issue as for copy
        # FIXME: Directly using to_json on PharmDataFrame doesn't work
        return pd.DataFrame(self).to_json(**kwargs)
