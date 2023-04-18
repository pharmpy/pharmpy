import re
from pathlib import Path

from .option_record import EnumOpt, IntOpt, MxOpt, OptionRecord, Opts, SimpleOpt, StrOpt, WildOpt

table_options = Opts(
    MxOpt('PRINT', 'print', default=True),
    MxOpt('NOPRINT', 'print'),
    StrOpt('FILE'),
    MxOpt('NOHEADER', 'header'),
    MxOpt('ONEHEADER', 'header'),
    SimpleOpt('ONEHEADERALL', noabbrev=True),
    MxOpt('NOTITLE', 'title'),
    MxOpt('NOLABEL', 'title'),
    MxOpt('FIRSTONLY', 'only'),
    MxOpt('LASTONLY', 'only'),
    MxOpt('FIRSTLASTONLY', 'only'),
    MxOpt('NOFORWARD', 'forward'),
    MxOpt('FORWARD', 'forward'),
    MxOpt('APPEND', 'append', default=True),
    MxOpt('NOAPPEND', 'append'),
    StrOpt('FORMAT', default='s1PE11.4'),
    StrOpt('LFORMAT'),
    StrOpt('RFORMAT'),
    StrOpt('IDFORMAT'),
    EnumOpt('NOSUB', (0, 1), default=0),
    StrOpt('PARAFILE'),
    IntOpt('ESAMPLE', default=300),
    SimpleOpt('WRESCHOL'),
    IntOpt('SEED', default=11456),
    EnumOpt('CLOCKSEED', (0, 1)),
    StrOpt('RANMETHOD'),
    EnumOpt('VARCALC', (0, 1, 2, 3)),
    StrOpt('FIXEDETAS'),
    EnumOpt('NPDTYPE', (0, 1), default=0),
    MxOpt('UNCONDITIONAL', 'cond', default=True),
    MxOpt('CONDITIONAL', 'cond'),
    SimpleOpt('OMITTED'),
    WildOpt(),
    # BY and EXCLUDE_BY missing
)


class TableRecord(OptionRecord):
    option_defs = table_options

    def __init__(self, name, raw_name, root):
        # Overriding to not parse options at parse time.
        # Need other records to be parsed first.
        super(OptionRecord, self).__init__(name, raw_name, root)

    @property
    def path(self):
        file_option = self.option_pairs['FILE']
        assert file_option is not None
        return Path(file_option)

    def set_path(self, value):
        newrec = self.set_option('FILE', str(value))
        return newrec

    @property
    def eta_derivatives(self):
        """List of numbers for etas whose derivative are requested"""
        return self._find_derivatives('G')

    @property
    def epsilon_derivatives(self):
        """List of numbers for epsilons whose derivateives are requested"""
        return self._find_derivatives('H')

    def _find_derivatives(self, ch):
        derivs = []
        regexp = ch + r'(\d+)1$'
        for key, value in self.all_options:
            m = re.match(regexp, key)
            if not m and value is not None:
                m = re.match(regexp, value)
            if m:
                n = m.group(1)
                derivs.append(int(n))
        return derivs

    def parse_options(self, nonoptions, netas):
        return self.option_defs.parse_ast(self.root, nonoptions=nonoptions, netas=netas)


# FIXME: These situations are not handled
# No columns are allowed after first option
# Overridden names are ignored after first option (not error)
