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

    def compartments(self):
        all_options = ['INITIALOFF', 'NOOFF', 'NODOSE', 'EQUILIBRIUM', 'EXCLUDE', 'DEFOBSERVATION',
                       'DEFDOSE']
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
