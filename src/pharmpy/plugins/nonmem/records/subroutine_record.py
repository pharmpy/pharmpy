"""
The NONMEM $SUBROUTINES record
"""

from .option_record import OptionRecord


class SubroutineRecord(OptionRecord):
    @property
    def advan(self):
        advan = self.get_option('ADVAN')
        if advan is None:
            advan = self.get_option_startswith('ADVAN')
        return advan

    def set_advan(self, value):
        newrec = self.remove_option_startswith('ADVAN').append_option(value)
        return newrec

    @property
    def trans(self):
        trans = self.get_option('TRANS')
        if trans is None:
            trans = self.get_option_startswith('TRANS')
        if trans is None:
            trans = 'TRANS1'
        return trans

    @property
    def tol(self):
        tol = self.get_option('TOL')
        if tol is not None:
            tol = int(tol)
        return tol

    @property
    def atol(self):
        atol = self.get_option('ATOL')
        if atol is None:
            atol = 1.0e-12
        else:
            atol = float(atol)
        return atol
