#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
===========
Model Input
===========

API for model input (e.g. dataset). Bound at :attr:`Model.input`).

Definitions
===========
"""

import logging

from .filters import InputFilters


class ModelInput(object):
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

    @property
    def id_column(self):
        """The name of the id_column
        """
        raise NotImplementedError

    def write_dataset(self):
        """Write the dataset at the dataset path
        """
        self.data_frame.to_csv(str(self.path), index=False)

    def apply_and_remove_filters(self):
        """A convenience method to apply all filters on the dataset
        and remove them from the model.
        """
        if self.filters:
            self.logger.debug('Filtering through %r', self.data_frame, self.filters)
            self.filters.apply(self.data_frame)
            self.filters = InputFilters([])
            self.logger.info('Data %r filtered: %d records (was %d)', self.data_frame, 0, 0)

    @property
    def logger(self):
        return logging.getLogger('%s.%s' % (self.model.logger.name, self.__class__.__name__))
