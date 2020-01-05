# import dateutil.parser
import re

# Warning! Bad and experimental code below.
# Should ideally be able to parse on demand or at least be able to skip sections


class NONMEMResultsFile:
    '''Representing and parsing a NONMEM results file (aka lst-file)
       This is not a generic output object and will be combined by other classes
       into new structures. For example if ext file exists we would not use the
       estimates from the lst-file.
    '''
    def __init__(self, path):
        with open(str(path), 'r') as res_file:
            content = res_file.readlines()
        parts = self._split_content(content)
        self._parse_datestamps(parts['start_timestamp'])
        self.model_code = parts['model_code']
        self.nmtran_messages = parts['nmtran_messages']
        self._parse_version(parts['about_text'])

    def _split_content(self, content):
        '''First coarse parser. Splitting the file in parts that will be parsed separately
        '''
        parts = {}
        parts['start_timestamp'] = content.pop(0)

        (parts['model_code'], content) = self._split_until_regexp(
                content, ['NM-TRAN MESSAGES', 'License Registered to:'], remove_lines=1)

        if content[0].startswith('NM-TRAN MESSAGES'):
            (parts['nmtran_messages'], content) = \
                    self._split_until_regexp(content, ['License Registered to:'], remove_lines=1)

        (_, content) = self._split_until_regexp(content, ['1NONLINEAR MIXED EFFECTS MODEL PROGRAM'])
        (parts['about_text'], content) = self._split_until_regexp(content, [' PROBLEM NO.:'])
        return parts

    def _split_until_regexp(self, content, regexps, remove_lines=0):
        '''Split an array of strings into two parts. One before the match of any of the regexps and one
        after and including the match.
        remove_lines is the number of lines to remove at the end of the before part
        '''
        match = False
        for i, line in enumerate(content):
            for regexp in regexps:
                if re.match(regexp, line):
                    match = True
            if match:
                break
        else:
            raise EOFError
        return (content[0:(i - remove_lines)], content[i:])

    def _parse_version(self, about_text):
        m = re.match(r'1NONLINEAR MIXED EFFECTS MODEL PROGRAM \(NONMEM\) VERSION (.*)\n',
                     about_text[0])
        if m:
            self.nonmem_version = m.group(1)

    def _parse_datestamps(self, raw_string):
        pass  # Crap! This dateutil doesn't seem to support Swedish
        # Had a look at dateparser that could support Swedish, but it needs patching for that
        # self.start_timestamp = dateutil.parser.parse(raw_string)
