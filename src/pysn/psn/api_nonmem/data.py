import re
from io import StringIO
import pandas as pd

# This class is an IO class that is a prefilter for pandas.read_table.
# Only things that cannot be handled by pandas will be taken care of here and the rest will be taken care of by pandas.

class NMTRANDataIO(StringIO):
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

# A data set

class Data:
    def __init__(self, filename):       # FIXME: Should init directly from model in future
        self.filename = filename
        self.ignore_character = '@'

    #@property
    #def data_frame(self):
    #    return self._data_frame

    def read(self):
        file_io = NMTRANDataIO(self.filename, self.ignore_character) 
        self.data_frame = pd.read_table(file_io, sep='\s+|,', header=None, engine='python')
