"""
The NONMEM $MODEL record
"""

from .option_record import OptionRecord


class ModelRecord(OptionRecord):
    @property
    def ncomps(self):
        nc = self.get_option("NCOMPARTMENTS")
        if nc is None:
            nc = self.get_option("NCOMPS")
            if nc is None:
                nc = self.get_option("NCM")
        if nc is not None:
            nc = int(nc)
        return nc

    def add_compartment(self, name, dosing=False):
        options = (name, 'DEFDOSE') if dosing else (name,)
        newrec = self.append_option('COMPARTMENT', f'({" ".join(options)})')
        return newrec

    def get_compartment_number(self, name):
        for i, (curname, _) in enumerate(self.compartments()):
            if name == curname:
                return i + 1
        return None

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
