# import dateutil.parser
import re

from numpy import nan
from packaging import version

# Should ideally be able to parse on demand or at least be able to skip sections


class NONMEMResultsFile:
    '''Representing and parsing a NONMEM results file (aka lst-file)
       This is not a generic output object and will be combined by other classes
       into new structures.
       We rely on tags in NONMEM >= 7.2.0
    '''

    def __init__(self, path):
        self.table = dict()
        self.nonmem_version = None
        for name, content in lst_blocks(path):
            if name == 'INIT':
                self.nonmem_version = content.pop('nonmem_version', None)
            else:
                self.table[name] = content

    def estimation_status(self, table_number):
        result = unknown_termination()
        if version.parse(self.nonmem_version) >= version.parse('7.2.0'):
            if (table_number in self.table.keys()):
                for key in result.keys():
                    result[key] = self.table[table_number].get(key)
            else:
                result['minimization_successful'] = False
        return result

    def covariance_status(self, table_number):
        result = {'covariance_step_ok': None}
        if version.parse(self.nonmem_version) >= version.parse('7.2.0'):
            if (table_number in self.table.keys()):
                result['covariance_step_ok'] = self.table[table_number].get('covariance_step_ok')
            else:
                result['covariance_step_ok'] = False
        return result

    def ofv(self, table_number):
        ofv = None
        if version.parse(self.nonmem_version) >= version.parse('7.2.0'):
            if (table_number in self.table.keys()):
                ofv = self.table[table_number].get('OBJV')
        if ofv is not None:
            ofv = float(ofv)
        else:
            ofv = nan
        return ofv


def _parse_datestamps(self, raw_string):
    pass  # Crap! This dateutil doesn't seem to support Swedish
    # Had a look at dateparser that could support Swedish, but it needs patching for that
    # self.start_timestamp = dateutil.parser.parse(raw_string)


def cleanup_version(v):
    if v == 'V':
        v = '5.0'
    elif v == 'VI':
        v = '6.0'
    return v


def parse_covariance(rows):
    if len(rows) < 1:
        return {'covariance_step_ok': False}
    result = {'covariance_step_ok': False}

    ok = re.compile(r' Elapsed covariance\s+time in seconds: ')  # need variable whitespace

    for row in rows:
        if ok.match(row):
            result['covariance_step_ok'] = True
            break
    return result


def unknown_termination():
    return {'minimization_successful': None,
            'estimate_near_boundary': None,
            'rounding_errors': None,
            'maxevals_exceeded': None,
            'significant_digits': nan,
            'function_evaluations': nan,
            'estimation_warning': None}


def parse_termination(rows):
    result = unknown_termination()
    if len(rows) < 1:  # will happen if e.g. TERMINATED BY OBJ during estimation
        result['minimization_successful'] = False
        return result
    result['estimate_near_boundary'] = False
    result['rounding_errors'] = False
    result['maxevals_exceeded'] = False
    result['estimation_warning'] = False

    success = [re.compile(r'0MINIMIZATION SUCCESSFUL'),
               re.compile(r' (REDUCED STOCHASTIC PORTION|OPTIMIZATION|' +
                          r'BURN-IN|EXPECTATION ONLY PROCESS)' +
                          r'( STATISTICALLY | WAS | )(COMPLETED|NOT TESTED)'),
               re.compile(r'1OBJECTIVE FUNCTION IS TO BE EVALUATED')]
    failure = [re.compile(r'0MINIMIZATION TERMINATED'),
               re.compile(r'0SEARCH WITH ESTIMATION STEP WILL NOT PROCEED')]
    maybe = re.compile(r' (REDUCED STOCHASTIC PORTION|OPTIMIZATION|' +
                       r'BURN-IN|EXPECTATION ONLY PROCESS)' +
                       r'( WAS | )NOT COMPLETED')  # success only if next line USER INTERRUPT
    misc = {'estimate_near_boundary': re.compile(r'0(ESTIMATE OF THETA IS NEAR THE BOUNDARY AND|' +
                                                 r'PARAMETER ESTIMATE IS NEAR ITS BOUNDARY)'),
            'rounding_errors': re.compile(r' DUE TO ROUNDING ERRORS'),
            'maxevals_exceeded': re.compile(r' DUE TO MAX. NO. OF FUNCTION EVALUATIONS EXCEEDED'),
            'estimation_warning': re.compile(r' HOWEVER, PROBLEMS OCCURRED WITH THE MINIMIZATION.')}
    sig_digits = re.compile(r' NO. OF SIG. DIGITS IN FINAL EST.:\s*(\S+)')  # only classical est
    feval = re.compile(r' NO. OF FUNCTION EVALUATIONS USED:\s*(\S+)')  # only classical est

    maybe_success = False
    for row in rows:
        if maybe_success:
            result['minimization_successful'] = bool(re.search(r'USER INTERRUPT', row))
            break
        for p in success:
            if p.match(row):
                result['minimization_successful'] = True
                break
        if result['minimization_successful'] is not None:
            break
        for p in failure:
            if p.match(row):
                result['minimization_successful'] = False
                break
        if result['minimization_successful'] is not None:
            break
        maybe_success = bool(maybe.match(row))
    for row in rows:
        if sig_digits.match(row):
            result['significant_digits'] = float(sig_digits.match(row).group(1))
            continue
        if feval.match(row):
            result['function_evaluations'] = int(feval.match(row).group(1))
            continue
        for name, p in misc.items():
            if p.match(row):
                result[name] = True
                break
    return result


def lst_items(path):
    nmversion = \
        re.compile(r'1NONLINEAR MIXED EFFECTS MODEL PROGRAM \(NONMEM\) VERSION\s+(\S+)')
    tag = re.compile(r'\s*#([A-Z]{4}):\s*(.*)')
    end_TERE = re.compile(r'(0|1)')  # The part we need after #TERE: will preceed next ^1 or ^0
    cleanup = re.compile(r'\*+\s*')
    TERM = list()
    TERE = list()
    found_TERM = False
    found_TERE = False

    with open(path) as file:
        version_number = 0
        for row in file:
            m = nmversion.match(row)
            if m:
                version_number = cleanup_version(m.group(1))
                yield ('nonmem_version', version_number)
                break  # we will stay at current file position
        if version.parse(version_number) >= version.parse('7.2.0'):
            for row in file:
                m = tag.match(row)
                if m:
                    if m.group(1) == 'TERM':
                        if found_TERM:
                            raise NotImplementedError('Two TERM tags without TERE in between')
                        found_TERM = True
                        TERM = list()
                    elif m.group(1) == 'TERE':
                        if not found_TERM:
                            raise NotImplementedError('TERE tag without TERM tag')
                        found_TERE = True
                        yield ('TERM', parse_termination(TERM))
                        found_TERM = False
                        TERM = list()
                    elif found_TERE:
                        raise NotImplementedError('TERE tag without ^1 or ^0 before next tag')
                    else:
                        v = cleanup.sub('', m.group(2))
                        yield (m.group(1), v.strip())
                elif found_TERE:
                    if end_TERE.match(row):
                        yield('TERE', parse_covariance(TERE))
                        found_TERE = False
                        TERE = list()
                    else:
                        TERE.append(row)
                elif found_TERM:
                    TERM.append(row)
            if found_TERM:
                yield ('TERM', parse_termination(TERM))
            if found_TERE:
                yield ('TERE', parse_covariance(TERE))


def lst_blocks(path):
    block = dict()
    table_number = 'INIT'
    for name, content in lst_items(path):
        if name == 'TERM' or name == 'TERE':
            for k, v in content.items():
                block[k] = v
        elif name == 'TBLN':
            if bool(block):
                yield (table_number, block)
            block = dict()
            table_number = int(content)
        else:
            # If already set then it means TBLN was missing, probably $SIM, skip
            if name not in block.keys():
                block[name] = content
    if bool(block):
        yield (table_number, block)
