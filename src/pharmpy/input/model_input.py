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


class DatasetError(Exception):
    pass

class DatasetWarning(Warning):
    pass

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
        """Gets the DataFrame object representing the dataset 
        """
        raise NotImplementedError

    def write_dataset(self):
        """Write the dataset at the dataset path
        """
        self.data_frame.to_csv(str(self.path), index=False)

    @property
    def logger(self):
        return logging.getLogger('%s.%s' % (self.model.logger.name, self.__class__.__name__))
