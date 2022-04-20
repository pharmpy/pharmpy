"""
The NONMEM $SIZES record
"""

from .option_record import OptionRecord


class SizesRecord(OptionRecord):
    @property
    def LTH(self):
        return int(self.option_pairs.get('LTH', 100))

    @LTH.setter
    def LTH(self, value):
        if value < 101:
            self.remove_option('LTH')
        else:
            self.set_option('LTH', value)

    @property
    def PC(self):
        return int(self.option_pairs.get('PC', 30))

    @PC.setter
    def PC(self, value):
        if value > 99:
            raise ValueError(
                f'Model has {value} compartments, but NONMEM only support a maximum of 99 '
                f'compartments.'
            )
        if value > 30:
            self.set_option('PC', value)
        else:
            self.remove_option('PC')
