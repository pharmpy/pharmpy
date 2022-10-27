import re
from datetime import datetime
from typing import Dict, Optional, Union

import dateutil.parser
from packaging import version

from pharmpy.deps import numpy as np


class NONMEMResultsFile:
    """Representing and parsing a NONMEM results file (aka lst-file)
    This is not a generic output object and will be combined by other classes
    into new structures.
    We rely on tags in NONMEM >= 7.2.0
    """

    def __init__(self, path=None, log=None):
        self.log = log
        self.table = {}
        self.nonmem_version = None
        self.runtime_total = None
        if path is not None:
            for name, content in self.table_blocks(path):
                if name == 'INIT':
                    self.nonmem_version = content.pop('nonmem_version', None)
                elif name == 'runtime':
                    self.runtime_total = content.pop('total', None)
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
            ofv = np.nan
        return ofv

    @staticmethod
    def supported_version(nonmem_version):
        return nonmem_version is not None and version.parse(nonmem_version) >= version.parse(
            '7.2.0'
        )

    @staticmethod
    def unknown_covariance() -> Dict[str, Optional[Union[bool, float]]]:
        return {'covariance_step_ok': None}

    @staticmethod
    def unknown_termination() -> Dict[str, Optional[Union[bool, float]]]:
        return {
            'minimization_successful': None,
            'estimate_near_boundary': None,
            'rounding_errors': None,
            'maxevals_exceeded': None,
            'significant_digits': np.nan,
            'function_evaluations': np.nan,
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
    def parse_tere(rows):
        result = NONMEMResultsFile.unknown_covariance()
        result['covariance_step_ok'] = False
        result['estimation_runtime'] = np.nan
        if len(rows) < 1:
            return result

        cov_not_ok = re.compile(
            r'( INTERPRET VARIANCE-COVARIANCE OF ESTIMATES WITH CARE)|'
            r'(R|S) MATRIX ALGORITHMICALLY SINGULAR'
        )
        cov_ok = re.compile(r' Elapsed covariance\s+time in seconds: ')  # need variable whitespace
        est_time = re.compile(r' Elapsed estimation\s+time in seconds:\s+(\d+\.*\d+)')

        for row in rows:
            if cov_not_ok.match(row):
                result['covariance_step_ok'] = False
                break
            if cov_ok.match(row):
                result['covariance_step_ok'] = True
                break
            m = est_time.match(row)
            if m:
                result['estimation_runtime'] = float(m.group(1))
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
            re.compile(r'\s*INDIVIDUAL OBJECTIVE FUNCTION VALUES ARE ALL ZERO\. PROBLEM ENDED'),
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
        sig_digits_unreport = re.compile(
            r'\w*(NO. OF SIG. DIGITS UNREPORTABLE)\w*\n'
        )  # only classical est
        feval = re.compile(r' NO. OF FUNCTION EVALUATIONS USED:\s*(\S+)')  # only classical est
        ofv_with_constant = re.compile(r' OBJECTIVE FUNCTION VALUE WITH CONSTANT:\s*(\S+)')

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
            m = sig_digits.match(row)
            if m:
                result['significant_digits'] = float(m.group(1))
                continue
            m = sig_digits_unreport.match(row)
            if m:
                result['significant_digits'] = np.nan
            m = ofv_with_constant.match(row)
            if m:
                result['ofv_with_constant'] = float(m.group(1))
                continue
            m = feval.match(row)
            if m:
                result['function_evaluations'] = int(m.group(1))
                continue
            for name, p in misc.items():
                if p.match(row):
                    result[name] = True
                    break
        return result

    @staticmethod
    def parse_runtime(row, row_next=None):
        # TODO: support AM/PM
        weekday_month_en = re.compile(
            r'^\s*(Sun|Mon|Tue|Wed|Thu|Fri|Sat)'
            r'\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'  # Month
            r'\s+(\d+)'  # Day
            r'\s+.*'
            r'\s+(\d{4})'  # Year
        )
        weekday_month_sv = re.compile(
            r'^\s*(mån|tis|ons|tor|fre|lör|sön)'
            r'\s+(\d+)'  # Day
            r'\s+(jan|feb|mar|apr|maj|jun|jul|aug|sep|okt|nov|dec)'  # Month
            r'\s+(\d+)'  # Year
        )
        day_month_year = re.compile(r'^(\d{2})/(\d{2})/(\d{4})\s*$')  # dd/mm/yyyy
        year_month_day = re.compile(r'^(\d{4})-\d{2}-\d{2}\s*$')  # yyyy/mm/dd
        timestamp = re.compile(r'([0-9]{2}:[0-9]{2}:[0-9]{2})')

        month_no = {
            'JAN': 1,
            'FEB': 2,
            'MAR': 3,
            'APR': 4,
            'MAY': 5,
            'JUN': 6,
            'JUL': 7,
            'AUG': 8,
            'SEP': 9,
            'OCT': 10,
            'NOV': 11,
            'DEC': 12,
        }
        month_trans = {'MAJ': 'MAY', 'OKT': 'OCT'}

        def _dmy(row):
            if match_en := weekday_month_en.match(row):
                _, month, day, year = match_en.groups()
                return day, month, year

            elif match_sv := weekday_month_sv.match(row):
                _, day, month, year = match_sv.groups()
                return day, month, year

            return None

        dmy = _dmy(row)
        if dmy is not None:
            day, month, year = dmy
            try:
                month = month_no[month.upper()]
            except KeyError:
                month_en = month_trans[month.upper()]
                month = month_no[month_en.upper()]

            date = datetime(int(year), int(month), int(day))

            match = timestamp.search(row)
            if match is None:
                return date

            time_str = match.groups()[0]
            time = dateutil.parser.parse(time_str).time()
            return datetime.combine(date, time)

        elif (match_day_first := day_month_year.match(row)) or year_month_day.match(row):
            date = dateutil.parser.parse(row, dayfirst=bool(match_day_first))

            if row_next is None:
                return date

            time = dateutil.parser.parse(row_next).time()
            return datetime.combine(date, time)

    def log_items(self, lines):
        fulltext = '\n'.join(lines)

        warnings = []
        errors = []

        warning_patterns = [
            re.compile(r'0WARNING:((\s+.+\n)+)'),
            re.compile(r'0(PARAMETER ESTIMATE IS NEAR ITS BOUNDARY)'),
            re.compile(r'0(MINIMIZATION SUCCESSFUL\n\s*HOWEVER.+\n)'),
        ]

        for pattern in warning_patterns:
            match = pattern.search(fulltext)
            if match:
                message = match.group(1)
                message_split = message.split('\n')
                message_trimmed = '\n'.join([m.strip() for m in message_split])
                warnings.append(message_trimmed.strip())

        error_patterns = [
            re.compile(
                r'(AN ERROR WAS FOUND IN THE CONTROL STATEMENTS\.(.*\n)+'
                r'.+UPPER OR LOWER BOUNDS\.)'
            ),
            re.compile(
                r'(INITIAL ESTIMATE OF OMEGA HAS A NONZERO BLOCK WHICH IS NUMERICALLY NOT '
                r'POSITIVE DEFINITE)'
            ),
            re.compile(r'0(UPPER BOUNDS INAPPROPRIATE)'),
            re.compile(r'0(PRED EXIT CODE = 1\n(.*\n)+.+MAY BE TOO LARGE\.)'),
            re.compile(
                r'0(PRED EXIT CODE = 1\n(.*\n)+\s*'
                r'NUMERICAL DIFFICULTIES OBTAINING THE SOLUTION\.)\s*\n'
            ),
            re.compile(r'0(PRED EXIT CODE = 1\n(.*\n)+\s+.+IS TOO CLOSE TO AN EIGENVALUE\s*)\n'),
            re.compile(r'0(PRED EXIT CODE = 1\n(.*\n)+\s+.+IS VERY LARGE\.\s*)\n'),
            re.compile(r'0(PROGRAM TERMINATED BY OBJ\n\s*MESSAGE ISSUED FROM ESTIMATION STEP)'),
            re.compile(
                r'0(PROGRAM TERMINATED BY OBJ\n(.*\n)*\s*'
                r'MESSAGE ISSUED FROM ESTIMATION STEP\n\s*'
                r'((AT 0TH ITERATION, UPON EVALUATION OF GRADIENT.*)|'
                r'(AT INITIAL OBJ. FUNCTION EVALUATION)))\n'
            ),
            re.compile(r'0(MINIMIZATION TERMINATED\n\s*DUE TO ROUNDING ERRORS.+)\n'),
            re.compile(r'0(MINIMIZATION TERMINATED\n\s*DUE TO ZERO GRADIENT)\n'),
            re.compile(
                r'0(MINIMIZATION TERMINATED\n\s*DUE TO MAX. NO. OF FUNCTION EVALUATIONS EXCEEDED)\n'
            ),
            re.compile(r'0(MINIMIZATION TERMINATED\n(.*\n)+\s*IS NON POSITIVE DEFINITE)\n'),
            re.compile(
                r'0(MINIMIZATION TERMINATED\n(.*\n)+\s*SUM OF "SQUARED" WEIGHTED INDIVIDUAL '
                r'RESIDUALS IS INFINITE)\n'
            ),
            re.compile(r'\s*(NO. OF SIG. DIGITS UNREPORTABLE)\s*\n'),
        ]

        for pattern in error_patterns:
            match = pattern.search(fulltext)
            if match:
                message = match.group(1)
                message_split = message.split('\n')
                message_trimmed = '\n'.join([m.strip() for m in message_split])
                errors.append(message_trimmed.strip())

        if self.log is not None:
            for message in warnings:
                self.log.log_warning(message)
            for message in errors:
                self.log.log_error(message)

    def tag_items(self, path):
        nmversion = re.compile(r'1NONLINEAR MIXED EFFECTS MODEL PROGRAM \(NONMEM\) VERSION\s+(\S+)')
        tag = re.compile(r'\s*#([A-Z]{4}):\s*(.*)')
        end_TERE = re.compile(r'(0|1)')  # The part we need after #TERE: will preceed next ^1 or ^0
        cleanup = re.compile(r'\*+\s*')
        TERM = []
        TERE = []
        found_TERM = False
        found_TERE = False
        runtime = None
        endtime_index = None

        with open(path, 'rb') as fp:
            binary = fp.readlines()
            # Since lst-files sometimes can have mixed encodings:
            # Allow for separate encodings of date strings and the rest of the content
            # Always try utf-8 first and fallback to latin-1 separately for each section
            try:
                line1 = binary[0].decode('utf-8')
            except UnicodeDecodeError:
                line1 = binary[0].decode('latin-1', errors='ignore')

            try:
                last_line = binary[-1].decode('utf-8')
            except UnicodeDecodeError:
                last_line = binary[-1].decode('latin-1', errors='ignore')

            chunk = b''.join(binary[1:-1])
            try:
                decoded_chunk = chunk.decode('utf-8')
            except UnicodeDecodeError:
                decoded_chunk = chunk.decode('latin-1', errors='ignore')

            lines = [line1]
            lines += decoded_chunk.replace('\r', '').split('\n')
            lines[-1] = last_line  # Replace since lst-files always end with \n

        version_number = None
        starttime = NONMEMResultsFile.parse_runtime(lines[0], lines[1])
        for row in lines:
            m = nmversion.match(row)
            if m:
                version_number = NONMEMResultsFile.cleanup_version(m.group(1))
                yield ('nonmem_version', version_number)
                break  # we will stay at current file position

        self.log_items(lines)

        if NONMEMResultsFile.supported_version(version_number):
            for i, row in enumerate(lines):
                row = row.rstrip()
                m = tag.match(row)
                if m:
                    if m.group(1) == 'TERM':
                        if found_TERM:
                            raise NotImplementedError('Two TERM tags without TERE in between')
                        found_TERM = True
                        TERM = []
                    elif m.group(1) == 'TERE':
                        if not found_TERM:
                            raise NotImplementedError('TERE tag without TERM tag')
                        found_TERE = True
                        yield ('TERM', NONMEMResultsFile.parse_termination(TERM))
                        found_TERM = False
                        TERM = []
                    elif found_TERE:
                        found_TERE = False
                        # raise NotImplementedError('TERE tag without ^1 or ^0 before next tag')
                    else:
                        v = cleanup.sub('', m.group(2))
                        yield (m.group(1), v.strip())
                elif found_TERE:
                    if end_TERE.match(row):
                        yield ('TERE', NONMEMResultsFile.parse_tere(TERE))
                        found_TERE = False
                        TERE = []
                    else:
                        TERE.append(row)
                elif found_TERM:
                    TERM.append(row)
                if row == 'Stop Time:':
                    endtime_index = i + 1

            if endtime_index is not None:
                second_line = lines[i] if (i := endtime_index + 1) < len(lines) else None
                endtime = NONMEMResultsFile.parse_runtime(lines[endtime_index], second_line)
                if starttime and endtime:
                    runtime = (endtime - starttime).total_seconds()

            if found_TERM:
                yield ('TERM', NONMEMResultsFile.parse_termination(TERM))
            if found_TERE:
                yield ('TERE', NONMEMResultsFile.parse_tere(TERE))
            if runtime is not None:
                yield ('runtime', runtime)

    def table_blocks(self, path):
        block = {}
        table_number = 'INIT'
        for name, content in self.tag_items(path):
            if name == 'TERM' or name == 'TERE':
                for k, v in content.items():
                    block[k] = v
            elif name == 'TBLN':
                if bool(block):
                    yield (table_number, block)
                block = {}
                table_number = int(content)
            elif name == 'runtime':
                yield ('runtime', {'total': content})
            else:
                # If already set then it means TBLN was missing, probably $SIM, skip
                if name not in block.keys():
                    block[name] = content
        if bool(block):
            yield (table_number, block)
