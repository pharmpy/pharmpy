"""
The NONMEM $SIZES record
"""

from .option_record import OptionRecord


class SizesRecord(OptionRecord):
    @property
    def LTH(self):
        lth = self.option_pairs.get('LTH', 100)
        assert lth is not None
        return int(lth)

    def set_LTH(self, value):
        if value < 101:
            newrec = self.remove_option('LTH')
        else:
            newrec = self.set_option('LTH', str(value))
        return newrec

    @property
    def PC(self):
        pc = self.option_pairs.get('PC', 30)
        assert pc is not None
        return int(pc)

    def set_PC(self, value):
        if value > 99:
            raise ValueError(
                f'Model has {value} compartments, but NONMEM only support a maximum of 99 '
                f'compartments.'
            )
        if value > 30:
            newrec = self.set_option('PC', str(value))
        else:
            newrec = self.remove_option('PC')
        return newrec
