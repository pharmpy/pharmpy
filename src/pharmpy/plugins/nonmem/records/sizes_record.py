"""
The NONMEM $SIZES record
"""

from dataclasses import dataclass

from .option_record import OptionRecord


@dataclass(frozen=True)
class SizesRecord(OptionRecord):
    @property
    def LTH(self):
        lth = self.option_pairs.get('LTH', 100)
        assert lth is not None
        return int(lth)

    def with_LTH(self, value):
        if value < 101:
            return self.remove_option('LTH')
        else:
            return self.set_option('LTH', str(value))

    @property
    def PC(self):
        pc = self.option_pairs.get('PC', 30)
        assert pc is not None
        return int(pc)

    def with_PC(self, value):
        if value > 99:
            raise ValueError(
                f'Model has {value} compartments, but NONMEM only support a maximum of 99 '
                f'compartments.'
            )
        if value > 30:
            return self.set_option('PC', str(value))
        else:
            return self.remove_option('PC')
