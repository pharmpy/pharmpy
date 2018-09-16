import re
from io import StringIO
from pathlib import Path
import pandas as pd



class NONMEMTableIO(StringIO):
    """ An IO class for NONMEM table files that is a prefilter for pandas.read_table.
        Things that cannot be handled directly by pandas will be taken care of here and the
        rest will be taken care of by pandas.
    """
    def __init__(self, filename):
        with open(str(filename), 'r') as tablefile:
            contents = tablefile.read()      # All variations of newlines are converted into \n

        # We might want to read out the data from the ext header here
        # and also add replicate (Table) column if applicable
        contents = re.sub(r'TABLE NO.\s+\d+.*\n', '', contents, flags=re.MULTILINE)

        super().__init__(contents)






class NONMEMTable:
    '''A NONMEM output table.
    '''
    def __init__(self, filename):
        file_io = NONMEMTableIO(filename)
        df = pd.read_table(file_io, sep='\s+|,', engine='python')
