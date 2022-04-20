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

    @property
    def trans(self):
        trans = self.get_option('TRANS')
        if trans is None:
            trans = self.get_option_startswith('TRANS')
        if trans is None:
            trans = 'TRANS1'
        return trans
