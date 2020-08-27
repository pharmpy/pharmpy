"""
The NONMEM $MODEL record
"""

from .option_record import OptionRecord


class ModelRecord(OptionRecord):
    def add_compartment(self, name, dosing=False):
        options = [name]
        if dosing:
            options.append('DEFDOSE')
        self.append_option('COMPARTMENT', f'({" ".join(options)})')
