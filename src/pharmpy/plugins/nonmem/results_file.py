# import dateutil.parser
import re

from numpy import nan
from packaging import version


class NONMEMResultsFile:
    """Representing and parsing a NONMEM results file (aka lst-file)
    This is not a generic output object and will be combined by other classes
    into new structures.
    We rely on tags in NONMEM >= 7.2.0
    """

    def __init__(self, path=None):
        self.table = dict()
        self.nonmem_version = None
        if path is not None:
            for name, content in NONMEMResultsFile.table_blocks(path):
                if name == 'INIT':
                    self.nonmem_version = content.pop('nonmem_version', None)
                else:
                    self.table[name] = content

    @property
    def _supported_nonmem_version(self):
        return NONMEMResultsFile.supported_version(self.nonmem_version)

    def estimation_status(self, table_number):
        result = NONMEMResultsFile.unknown_termination()
        if self._supported_nonmem_version:
            if table_number in self.table.keys():
                for key in result.keys():
                    result[key] = self.table[table_number].get(key)
            else:
                result['minimization_successful'] = False
        return result

    def covariance_status(self, table_number):
        result = NONMEMResultsFile.unknown_covariance()
        if self._supported_nonmem_version:
            if table_number in self.table.keys():
                for key in result.keys():
                    result[key] = self.table[table_number].get(key)
            else:
                result['covariance_step_ok'] = False
        return result

    def ofv(self, table_number):
        ofv = None
        if self._supported_nonmem_version:
            if table_number in self.table.keys():
                ofv = self.table[table_number].get('OBJV')
        if ofv is not None:
            ofv = float(ofv)
        else:
            ofv = nan
        return ofv

    @staticmethod
    def supported_version(nonmem_version):
        return nonmem_version is not None and version.parse(nonmem_version) >= version.parse(
            '7.2.0'
        )

    @staticmethod
    def unknown_covariance():
        return {'covariance_step_ok': None}

    @staticmethod
    def unknown_termination():
        return {
            'minimization_successful': None,
            'estimate_near_boundary': None,
            'rounding_errors': None,
            'maxevals_exceeded': None,
            'significant_digits': nan,
            'function_evaluations': nan,
            'warning': None,
        }

    @staticmethod
    def cleanup_version(v):
        if v == 'V':
            v = '5.0'
        elif v == 'VI':
            v = '6.0'
        return v

    @staticmethod
    def parse_covariance(rows):
        result = NONMEMResultsFile.unknown_covariance()
        result['covariance_step_ok'] = False
        if len(rows) < 1:
            return result

        not_ok = re.compile(r' INTERPRET VARIANCE-COVARIANCE OF ESTIMATES WITH CARE')
        not_ok_either = re.compile(r'(R|S) MATRIX ALGORITHMICALLY SINGULAR')
        ok = re.compile(r' Elapsed covariance\s+time in seconds: ')  # need variable whitespace

        for row in rows:
            if not_ok.match(row):
                result['covariance_step_ok'] = False
                break
            if not_ok_either.search(row):
                result['covariance_step_ok'] = False
                break
            if ok.match(row):
                result['covariance_step_ok'] = True
                break
        return result

    @staticmethod
    def parse_termination(rows):
        result = NONMEMResultsFile.unknown_termination()
        if len(rows) < 1:  # will happen if e.g. TERMINATED BY OBJ during estimation
            result['minimization_successful'] = False
            return result
        result['estimate_near_boundary'] = False
        result['rounding_errors'] = False
        result['maxevals_exceeded'] = False
        result['warning'] = False

        success = [
            re.compile(r'0MINIMIZATION SUCCESSFUL'),
            re.compile(
                r' (REDUCED STOCHASTIC PORTION|OPTIMIZATION|'
                + r'BURN-IN|EXPECTATION ONLY PROCESS)'
                + r'( STATISTICALLY | WAS | )(COMPLETED|NOT TESTED)'
            ),
            re.compile(r'1OBJECTIVE FUNCTION IS TO BE EVALUATED'),
        ]
        failure = [
            re.compile(r'0MINIMIZATION TERMINATED'),
            re.compile(r'0SEARCH WITH ESTIMATION STEP WILL NOT PROCEED'),
        ]
        maybe = re.compile(
            r' (REDUCED STOCHASTIC PORTION|OPTIMIZATION|'
            + r'BURN-IN|EXPECTATION ONLY PROCESS)'
            + r'( WAS | )NOT COMPLETED'
        )  # success only if next line USER INTERRUPT
        misc = {
            'estimate_near_boundary': re.compile(
                r'0(ESTIMATE OF THETA IS NEAR THE BOUNDARY AND|'
                + r'PARAMETER ESTIMATE IS NEAR ITS BOUNDARY)'
            ),
            'rounding_errors': re.compile(r' DUE TO ROUNDING ERRORS'),
            'maxevals_exceeded': re.compile(r' DUE TO MAX. NO. OF FUNCTION EVALUATIONS EXCEEDED'),
            'warning': re.compile(r' HOWEVER, PROBLEMS OCCURRED WITH THE MINIMIZATION.'),
        }
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

    @staticmethod
    def tag_items(path):
        nmversion = re.compile(r'1NONLINEAR MIXED EFFECTS MODEL PROGRAM \(NONMEM\) VERSION\s+(\S+)')
        tag = re.compile(r'\s*#([A-Z]{4}):\s*(.*)')
        end_TERE = re.compile(r'(0|1)')  # The part we need after #TERE: will preceed next ^1 or ^0
        cleanup = re.compile(r'\*+\s*')
        TERM = list()
        TERE = list()
        found_TERM = False
        found_TERE = False

        with open(path) as file:
            version_number = None
            for row in file:
                m = nmversion.match(row)
                if m:
                    version_number = NONMEMResultsFile.cleanup_version(m.group(1))
                    yield ('nonmem_version', version_number)
                    break  # we will stay at current file position
            if NONMEMResultsFile.supported_version(version_number):
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
                            yield ('TERM', NONMEMResultsFile.parse_termination(TERM))
                            found_TERM = False
                            TERM = list()
                        elif found_TERE:
                            raise NotImplementedError('TERE tag without ^1 or ^0 before next tag')
                        else:
                            v = cleanup.sub('', m.group(2))
                            yield (m.group(1), v.strip())
                    elif found_TERE:
                        if end_TERE.match(row):
                            yield ('TERE', NONMEMResultsFile.parse_covariance(TERE))
                            found_TERE = False
                            TERE = list()
                        else:
                            TERE.append(row)
                    elif found_TERM:
                        TERM.append(row)
                if found_TERM:
                    yield ('TERM', NONMEMResultsFile.parse_termination(TERM))
                if found_TERE:
                    yield ('TERE', NONMEMResultsFile.parse_covariance(TERE))

    @staticmethod
    def table_blocks(path):
        block = dict()
        table_number = 'INIT'
        for name, content in NONMEMResultsFile.tag_items(path):
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
