# -*- encoding: utf-8 -*-

from pathlib import Path

from pysn import generic

from .model_estimation import ModelEstimation


class ModelOutput(generic.ModelOutput):
    """A NONMEM 7.x model output class."""

    @property
    def estimation(self):
        table_records = self.model.get_records("TABLE")
        table_paths = [Path(record.path).resolve() for record in table_records]
        return ModelEstimation(tables=table_paths)
