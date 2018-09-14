class ModelInput(object):
    """Implements API for :attr:`Model.input`, the model dataset"""
    def __init__(self, model):
        self.model = model

    @property
    def path(self):
        """Gets the path of the dataset"""
        raise NotImplementedError

    @path.setter
    def path(self):
        """Sets the path of the dataset"""
        raise NotImplementedError

    @property
    def data_frame(self, p):
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
