"""
The NONMEM $MODEL record
"""

from dataclasses import dataclass

from .option_record import OptionRecord


@dataclass(frozen=True)
class ModelRecord(OptionRecord):
    @property
    def ncomps(self):
        for option in ('NCOMPARTMENTS', 'NCOMPS', 'NCM'):
            nc = self.get_option(option)
            if nc is not None:
                return int(nc)

        return None

    def add_compartment(self, name, dosing=False):
        options = (name, 'DEFDOSE') if dosing else (name,)
        return self.append_option('COMPARTMENT', f'({" ".join(options)})')

    def prepend_compartment(self, name, dosing=False):
        options = (name, 'DEFDOSE') if dosing else (name,)
        return self.prepend_option('COMPARTMENT', f'({" ".join(options)})')

    def get_compartment_number(self, name):
        for i, (curname, _) in enumerate(self.compartments()):
            if name == curname:
                return i + 1
        return None

    def remove_compartment(self, name):
        n = self.get_compartment_number(name)
        assert n is not None
        return self.remove_nth_option('COMPARTMENT', n - 1)

    def set_dosing(self, name):
        n = self.get_compartment_number(name)
        assert n is not None
        return self.add_suboption_for_nth('COMPARTMENT', n - 1, 'DEFDOSE')

    def move_dosing_first(self):
        self.remove_suboption_for_all('COMPARTMENT', 'DEFDOSE')
        return self.add_suboption_for_nth('COMPARTMENT', 0, 'DEFDOSE')

    def compartments(self):
        ncomps = self.ncomps
        if (
            ncomps is not None
            and not self.has_option("COMPARTMENT")
            and not self.has_option("COMP")
        ):
            for i in range(1, ncomps + 1):
                yield f'COMP{i}', []
            return

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
