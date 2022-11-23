"""
The NONMEM $SIZES record
"""

from dataclasses import dataclass
from functools import cached_property

from .option_record import OptionRecord
from .record import with_parsed_and_generated


@with_parsed_and_generated
@dataclass(frozen=True)
class SizesRecord(OptionRecord):
    @cached_property
    def LTH(self):
        lth = self.option_pairs.get('LTH', 100)
        assert lth is not None
        return int(lth)

    def replace_LTH(self, value):
        if value < 101:
            return self.remove_option('LTH')
        else:
            return self.set_option('LTH', str(value))

    @cached_property
    def PC(self):
        pc = self.option_pairs.get('PC', 30)
        assert pc is not None
        return int(pc)

    def replace_PC(self, value):
        if value > 99:
            raise ValueError(
                f'Model has {value} compartments, but NONMEM only support a maximum of 99 '
                f'compartments.'
            )
        if value > 30:
            return self.set_option('PC', str(value))
        else:
            return self.remove_option('PC')
