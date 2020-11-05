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

    def prepend_compartment(self, name, dosing=False):
        options = [name]
        if dosing:
            options.append('DEFDOSE')
        self.prepend_option('COMPARTMENT', f'({" ".join(options)})')

    def get_compartment_number(self, name):
        for i, (curname, _) in enumerate(self.compartments()):
            if name == curname:
                return i + 1
        return None

    def remove_compartment(self, name):
        n = self.get_compartment_number(name)
        self.remove_nth_option('COMPARTMENT', n - 1)

    def set_dosing(self, name):
        n = self.get_compartment_number(name)
        self.add_suboption_for_nth('COMPARTMENT', n - 1, 'DEFDOSE')

    def move_dosing_first(self):
        self.remove_suboption_for_all('COMPARTMENT', 'DEFDOSE')
        self.add_suboption_for_nth('COMPARTMENT', 0, 'DEFDOSE')

    def compartments(self):
        all_options = [
            'INITIALOFF',
            'NOOFF',
            'NODOSE',
            'EQUILIBRIUM',
            'EXCLUDE',
            'DEFOBSERVATION',
            'DEFDOSE',
        ]
        for n, opts in enumerate(self.get_option_lists('COMPARTMENT')):
            name = f'COMP{n + 1}'
            options = []
            for opt in opts:
                match = OptionRecord.match_option(all_options, opt)
                if match:
                    options.append(match)
                else:
                    name = opt
            yield name, options
