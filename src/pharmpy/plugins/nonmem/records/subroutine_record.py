"""
The NONMEM $SUBROUTINES record
"""

from dataclasses import dataclass
from functools import cached_property

from .option_record import OptionRecord
from .record import with_parsed_and_generated


@with_parsed_and_generated
@dataclass(frozen=True)
class SubroutineRecord(OptionRecord):
    @cached_property
    def advan(self):
        advan = self.get_option('ADVAN')
        if advan is None:
            advan = self.get_option_startswith('ADVAN')
        return advan

    def replace_advan(self, value):
        # FIXME: Need replace_option
        return self.remove_option_startswith('ADVAN').append_option(value)

    @cached_property
    def trans(self):
        trans = self.get_option('TRANS')
        if trans is None:
            trans = self.get_option_startswith('TRANS')
            if trans is None:
                trans = 'TRANS1'
        return trans

    @cached_property
    def tol(self):
        tol = self.get_option('TOL')
        if tol is not None:
            tol = int(tol)
        return tol

    @cached_property
    def atol(self):
        atol = self.get_option('ATOL')
        if atol is None:
            atol = 1.0e-12
        else:
            atol = float(atol)
        return atol
