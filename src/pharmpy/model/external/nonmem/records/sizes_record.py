"""
The NONMEM $SIZES record
"""

from typing import Self

from .option_record import OptionRecord

DEFAULT_ISAMPLEMAX = 10


class SizesRecord(OptionRecord):
    @property
    def LTH(self) -> int:
        lth = self.option_pairs.get('LTH', 100)
        assert lth is not None
        return int(lth)

    def set_LTH(self, value: int) -> Self:
        curval = self.LTH
        if curval < 0 and abs(curval) >= value:
            return self
        elif value < 101:
            newrec = self.remove_option('LTH')
        else:
            newrec = self.set_option('LTH', str(value))
        return newrec

    @property
    def PC(self):
        pc = self.option_pairs.get('PC', 30)
        assert pc is not None
        return int(pc)

    def set_PC(self, value) -> Self:
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

    @property
    def ISAMPLEMAX(self) -> int:
        isamplemax = self.option_pairs.get('ISAMPLEMAX', DEFAULT_ISAMPLEMAX)
        assert isamplemax is not None
        return int(isamplemax)

    def set_ISAMPLEMAX(self, value: int) -> Self:
        if value > DEFAULT_ISAMPLEMAX:
            newrec = self.set_option('ISAMPLEMAX', str(value))
        else:
            newrec = self.remove_option('ISAMPLEMAX')
        return newrec
