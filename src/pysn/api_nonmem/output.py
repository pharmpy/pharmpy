# -*- encoding: utf-8 -*-

# from pathlib import Path

from pysn import generic


class ModelOutput(generic.ModelOutput):
    """A NONMEM 7.x model output class."""

    def __init__(self, model):
        self.model = model
        # table_records = model.get_records("TABLE")
        # table_paths = [Path(record.path) for record in table_records]
