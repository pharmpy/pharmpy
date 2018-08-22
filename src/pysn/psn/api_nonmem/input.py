# -*- encoding: utf-8 -*-

import re
from io import StringIO
import pandas as pd

from . import generic


class NMTRANDataIO(StringIO):
    """ An IO class that is a prefilter for pandas.read_table.
        Things that cannot be handled directly by pandas will be taken care of here and the
        rest will be taken care of by pandas.
    """
    def __init__(self, filename, ignore_character):
        with open(str(filename), 'r') as datafile:
            contents = datafile.read()      # All variations of newlines are converted into \n

        if ignore_character:
            if ignore_character == '@':
                comment_regexp = re.compile(r'^[A-Za-z].*\n', re.MULTILINE)
            else:
                comment_regexp = re.compile('^[' + ignore_character + '].*\n', re.MULTILINE)
            contents = re.sub(comment_regexp, '', contents)

        contents = re.sub(r'\s\.\s', '0', contents)        # Replace dot surrounded by space with 0 as explained in the NM-TRAN manual
        super().__init__(contents)


class ModelInput(generic.ModelInput):
    """A NONMEM 7.x model input class. Covers at least $INPUT and $DATA."""

    def __init__(self, model):
        self.model = model
        data_records = model.get_records("DATA")
        filename = data_records[0].first_key
        self._path = model.path.parent / filename
        print(self.path)
        self.ignore_character = '@'     # FIXME: Read from model!

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, p):
        self._path = p

    @property
    def data_frame(self):
        try:
            return self._data_frame
        except:
            file_io = NMTRANDataIO(self.path, self.ignore_character) 
            self._data_frame = pd.read_table(file_io, sep='\s+|,', header=None, engine='python')
        return self._data_frame
