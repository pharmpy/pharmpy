from .filters import InputFilters


class ModelInput(object):
    """Implements API for :attr:`Model.input`, the model dataset"""
    def __init__(self, model):
        self.model = model

    @property
    def path(self):
        """The path to the dataset
        """
        return self._path

    @path.setter
    def path(self, p):
        self._path = p

    @property
    def data_frame(self):
        """Gets the pandas DataFrame object representing the dataset"""
        raise NotImplementedError

    @property
    def filters(self):
        """Gets an InputFilters object representing
        all data filters of the model
        """
        raise NotImplementedError

    @filters.setter
    def filters(self, new):
        """Sets all data filters
        """
        raise NotImplementedError

    def apply_and_remove_filters(self):
        """A convenience method to apply all filters on the dataset
        and remove them from the model.
        """
        self.filters.apply(self.data_frame)
        self.filters = InputFilters([])
