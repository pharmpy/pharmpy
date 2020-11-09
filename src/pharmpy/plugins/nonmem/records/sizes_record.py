"""
The NONMEM $SIZES record
"""

from .option_record import OptionRecord


class SizesRecord(OptionRecord):
    @property
    def PC(self):
        return self.option_pairs.get('PC', 30)

    @PC.setter
    def PC(self, value):
        if value > 99:
            raise ValueError(f'Model need {value} compartments, but NONMEM only support a maximum of 99 compartments.')
        if value > 30:
            self.set_option('PC', value)
        else:
            self.remove_option('PC')
