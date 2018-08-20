import re

# This class is an IO class that is a prefilter for pandas.read_table. 
# Only things that cannot be handled by pandas will be taken care of here and the rest will be taken care of by pandas.

class NMTRANDataIO:
    def __init__(self, filename, ignore_character):
        if ignore_character == '@':
            comment_regexp = re.compile(r'^[A-Za-z].*\n', re.MULTILINE)
        else:
            comment_regexp = re.compile('^[' + ignore_character + '].*\n', re.MULTILINE)

        with open(filename, "r") as datafile:
            self.contents = datafile.read()      # All variations of newlines are converted into \n

        self.contents = re.sub(comment_regexp, '', self.contents)
        self.contents = re.sub(r'\s\.\s', '0', self.contents)        # Replace dot surrounded by space with 0 as explained in the NM-TRAN manual

        self.length = len(self.contents)
        self.index = 0

    def read(self, numbytes):
        index = self.index
        self.index += numbytes
        return self.contents[index:index + numbytes]
