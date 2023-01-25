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

    @LTH.setter
    def LTH(self, value):
        if value < 101:
            newrec = self.remove_option('LTH')
            self.root = newrec.root  # FIXME!
        else:
            newrec = self.set_option('LTH', str(value))
            self.root = newrec.root  # FIXME!

    @property
    def PC(self):
        pc = self.option_pairs.get('PC', 30)
        assert pc is not None
        return int(pc)

    @PC.setter
    def PC(self, value):
        if value > 99:
            raise ValueError(
                f'Model has {value} compartments, but NONMEM only support a maximum of 99 '
                f'compartments.'
            )
        if value > 30:
            newrec = self.set_option('PC', str(value))
            self.root = newrec.root  # FIXME!
        else:
            newrec = self.remove_option('PC')
            self.root = newrec.root  # FIXME!
