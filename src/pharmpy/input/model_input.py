"""
===========
Model Input
===========

API for model input (e.g. dataset). Bound at :attr:`Model.input`).

Definitions
===========
"""

import logging


class ModelInput:
    """Implements API for :attr:`Model.input`, the model dataset"""
    def __init__(self, model):
        self.model = model

    @property
    def path(self):
        """Resolved (absolute) path to the dataset."""
        raise NotImplementedError

    @path.setter
    def path(self, path):
        self.logger.info('Setting %r.path to %r', repr(self), str(path))

    @property
    def dataset(self):
        """Retrieve the dataset as a PharmDataFrame
        """
        raise NotImplementedError

    @dataset.setter
    def dataset(self):
        """Replace the dataset
        """
        raise NotImplementedError

    @property
    def raw_dataset(self):
        """Retrieve the dataset as a PharmDataFrame with no processing. Keeping the raw strings
        """
        raise NotImplementedError

    def write_dataset(self):
        """Write the dataset at the dataset path
        """
        self.data_frame.to_csv(str(self.path), index=False)

    @property
    def logger(self):
        return logging.getLogger('%s.%s' % (self.model.logger.name, self.__class__.__name__))
