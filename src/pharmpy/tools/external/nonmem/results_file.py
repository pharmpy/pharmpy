import re
import sys
from datetime import datetime
from itertools import tee
from typing import Callable, Generator, Iterable, Iterator, Optional, TypeVar, Union

import dateutil.parser
from packaging import version

from pharmpy.deps import numpy as np
from pharmpy.deps import pandas as pd
from pharmpy.model.external.nonmem.dataset.nmtran import IOFromChunks

TAG = re.compile(r' #([A-Z]{4}):\s*(.*)')
END_TERE = re.compile(r'(0|1)')  # The part we need after #TERE: will precede the next ^1 or ^0
CLEANUP = re.compile(r'\*+\s*')


def _decode_lst(line: bytes):
    # Since lst-files sometimes can have mixed encodings:
    # Allow for separate encodings of date strings and the rest of the content
    # Always try utf-8 first and fallback to latin-1 separately for each section
    try:
        row = line.decode('utf-8')
    except UnicodeDecodeError:
        row = line.decode('latin-1', errors='ignore')

    pos = len(row) - 2
    if pos >= 0 and row[pos] == '\r':
        return row[:pos] + '\n'
    else:
        return row


T = TypeVar('T')

_version = sys.version_info[:3]
_minor = _version[:2]

if _minor == (3, 12) and _version >= (3, 12, 8):
    # SEE: https://github.com/python/cpython/commit/cf2532b39d099e004d1c07b2d0fcc46567b68e75
    _tee = tee

elif _minor == (3, 13) and _version >= (3, 13, 1):
    # SEE: https://github.com/python/cpython/commit/7bc99dd49ed4cebe4795cc7914c4231209b2aa4b
    _tee = tee

elif _minor >= (3, 14):
    # SEE: https://github.com/python/cpython/pull/124490
    _tee = tee

else:
    # SEE: https://github.com/python/cpython/issues/137597#issuecomment-3186240062
    def _tee(iterable: Iterable[T], n: int = 2, /):
        if hasattr(iterable, "__copy__"):
            return tee(iterable, n + 1)[1:]
        else:
            return tee(iterable, n)


def make_peekable(iterator: Iterator[T]):
    # NOTE: Adapted from https://docs.python.org/3/library/itertools.html#itertools.tee
    # NOTE: Extra copy required for Python 3.11
    (tee_iterator,) = _tee(iterator, 1)

    def lookahead(n: int):
        # NOTE: Extra copy required for Python 3.11
        # SEE: https://github.com/python/cpython/issues/137597#issuecomment-3186240062
        (forked_iterator,) = _tee(tee_iterator, 1)
        for _ in range(n):
            yield next(forked_iterator, None)

    return tee_iterator, lookahead


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
        self.runtime_total: Optional[float] = None
        if path is not None:
            for name, content in self.table_blocks(path):
                if name == 'INIT':
                    self.nonmem_version = content.pop('nonmem_version', None)
                elif name == 'runtime':
                    runtime_total = content.pop('total', None)
                    assert isinstance(runtime_total, float)
                    self.runtime_total = runtime_total
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
    def unknown_covariance() -> dict[str, Optional[Union[bool, float]]]:
        return {'covariance_step_ok': None}

    @staticmethod
    def unknown_termination() -> dict[str, Optional[Union[bool, float, pd.DataFrame]]]:
        return {
            'minimization_successful': None,
            'estimate_near_boundary': None,
            'rounding_errors': None,
            'maxevals_exceeded': None,
            'significant_digits': np.nan,
            'function_evaluations': np.nan,
            'warning': None,
            'eta_shrinkage': None,
            'ebv_shrinkage': None,
            'eps_shrinkage': None,
        }

    @staticmethod
    def cleanup_version(v):
        if v == 'V':
            v = '5.0'
        elif v == 'VI':
            v = '6.0'
        return v

    @staticmethod
    def read_tere(
        rows: Iterator[str], lookahead: Callable[[int], Generator[str | None, None, None]]
    ):
        read = []
        while True:
            preread = next(lookahead(1))
            if preread is None:
                break
            lead = preread[:2]
            if lead == ' #' and TAG.match(preread):
                # Raise NotImplementedError('TERE tag without ^1 or ^0 before next tag')
                return read
            elif END_TERE.match(preread.rstrip()):
                return read
            else:
                row = next(rows)
                assert row == preread
                read.append(row.rstrip())

        return read

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
        # Need variable whitespace
        cov_ok = re.compile(r' Elapsed (covariance|opt\. design)\s+time in seconds: ')
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
        if len(rows) < 1:  # Will happen if e.g. TERMINATED BY OBJ during estimation
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
            re.compile(r'0HESSIAN OF POSTERIOR DENSITY IS NON-POSITIVE-DEFINITE DURING SEARCH'),
        ]
        maybe = re.compile(
            r' (REDUCED STOCHASTIC PORTION|OPTIMIZATION|'
            + r'BURN-IN|EXPECTATION ONLY PROCESS)'
            + r'( WAS | )NOT COMPLETED'
        )  # Success only if next line USER INTERRUPT
        misc = {
            'estimate_near_boundary': re.compile(
                r'0(ESTIMATE OF THETA IS NEAR THE BOUNDARY AND|'
                + r'PARAMETER ESTIMATE IS NEAR ITS BOUNDARY)'
            ),
            'rounding_errors': re.compile(r' DUE TO ROUNDING ERRORS'),
            'maxevals_exceeded': re.compile(r' DUE TO MAX. NO. OF FUNCTION EVALUATIONS EXCEEDED'),
            'warning': re.compile(r' HOWEVER, PROBLEMS OCCURRED WITH THE MINIMIZATION.'),
        }
        sig_digits = re.compile(r' NO. OF SIG. DIGITS IN FINAL EST.:\s*(\S+)')  # Only classical est
        sig_digits_unreport = re.compile(
            r'\w*(NO. OF SIG. DIGITS UNREPORTABLE)\w*\n'
        )  # Only classical est
        feval = re.compile(r' NO. OF FUNCTION EVALUATIONS USED:\s*(\S+)')  # Only classical est
        ofv_with_constant = re.compile(r' OBJECTIVE FUNCTION VALUE WITH CONSTANT:\s*(\S+)')
        eta_shrinkage = re.compile(r'^ ETASHRINK(?:SD|VR)\(%\)  ')
        ebv_shrinkage = re.compile(r'^ EBVSHRINK(?:SD|VR)\(%\)  ')
        eps_shrinkage = re.compile(r'^ EPSSHRINK(?:SD|VR)\(%\)  ')

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

        result['eta_shrinkage'] = NONMEMResultsFile.parse_shrinkage(
            filter(eta_shrinkage.match, rows)
        )
        result['ebv_shrinkage'] = NONMEMResultsFile.parse_shrinkage(
            filter(ebv_shrinkage.match, rows)
        )
        result['eps_shrinkage'] = NONMEMResultsFile.parse_shrinkage(
            filter(eps_shrinkage.match, rows)
        )
        return result

    @staticmethod
    def parse_shrinkage(rows: Iterable[str]):
        try:
            return pd.read_table(
                IOFromChunks(map(lambda row: str.encode(row + '\n'), rows)),  # type: ignore
                header=None,
                index_col=0,
                sep=r'\s+',
                engine='c',
                float_precision="round_trip",
            )
        except pd.errors.EmptyDataError:
            return None

    @staticmethod
    def parse_runtime(row, row_next=None):
        # TODO: Support AM/PM
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
        if self.log is None:
            return

        fulltext = ''.join(lines)

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
            # This is duplicated in termination
            re.compile(r'0(HESSIAN OF POSTERIOR DENSITY IS NON-POSITIVE-DEFINITE DURING SEARCH)'),
        ]

        for pattern in error_patterns:
            match = pattern.search(fulltext)
            if match:
                message = match.group(1)
                message_split = message.split('\n')
                message_trimmed = '\n'.join([m.strip() for m in message_split])
                errors.append(message_trimmed.strip())

        for message in warnings:
            self.log = self.log.log_warning(message)
        for message in errors:
            self.log = self.log.log_error(message)

    def tag_items(self, path):
        nmversion = re.compile(r'1NONLINEAR MIXED EFFECTS MODEL PROGRAM \(NONMEM\) VERSION\s+(\S+)')

        version_number = None

        with open(path, 'rb') as fp:
            lines = map(_decode_lst, fp)
            if self.log is not None:
                lines, _lines = tee(lines, 2)
                self.log_items(_lines)

            it = lines
            starttime = NONMEMResultsFile.parse_runtime(next(it), next(it))

            for row in it:
                m = nmversion.match(row)
                if m:
                    version_number = NONMEMResultsFile.cleanup_version(m.group(1))
                    yield ('nonmem_version', version_number)
                    break  # We will stay at current file position

            if NONMEMResultsFile.supported_version(version_number):
                endtime = yield from NONMEMResultsFile.parse_rows(it)

                if starttime is not None and endtime is not None:
                    runtime = (endtime - starttime).total_seconds()
                    yield ('runtime', runtime)

    @staticmethod
    def parse_rows(it: Iterator[str]):
        endtime = None
        hessian = None

        it, lookahead = make_peekable(it)

        for row in it:
            lead = row[:2]
            if lead == ' #' and (m := TAG.match(row)):
                if m.group(1) == 'TERM':
                    # The hessian termination error is not in the TERM block
                    TERM = [] if hessian is None else [hessian]

                    hessian = None

                    while True:
                        preread = next(lookahead(1))
                        if preread is None:
                            break
                        lead = preread[:2]
                        if lead == ' #' and (m := TAG.match(preread)):
                            if m.group(1) == 'TERM':
                                raise NotImplementedError('Two TERM tags without TERE in between')
                            elif m.group(1) == 'TERE':
                                next(it)
                                TERE = NONMEMResultsFile.read_tere(it, lookahead)
                                yield ('TERE', NONMEMResultsFile.parse_tere(TERE))
                            break
                        else:
                            row = next(it)
                            assert row == preread
                            TERM.append(row.rstrip())

                    yield ('TERM', NONMEMResultsFile.parse_termination(TERM))
                elif m.group(1) == 'TERE':
                    raise NotImplementedError('TERE tag without TERM tag')
                else:
                    v = CLEANUP.sub('', m.group(2))
                    yield (m.group(1), v.strip())

                    if m.group(1) == 'CPUT':
                        _header, date, time = lookahead(3)
                        assert _header is not None and _header.startswith('Stop Time:')
                        endtime = NONMEMResultsFile.parse_runtime(date, time)

                    elif m.group(1) == 'METH':
                        while True:
                            preread = next(lookahead(1))
                            if preread is None:
                                break
                            lead = preread[:2]
                            if lead == ' #' and (m := TAG.match(preread)):
                                break
                            elif lead == '0H' and row.startswith('0HESSIAN OF POSTERIOR DENSITY'):
                                next(it)
                                _, maybe_term = lookahead(2)
                                if (
                                    maybe_term is not None
                                    and (m := TAG.match(maybe_term))
                                    and m.group(1) == 'TERM'
                                ):
                                    hessian = row
                            else:
                                next(it)

        return endtime

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
