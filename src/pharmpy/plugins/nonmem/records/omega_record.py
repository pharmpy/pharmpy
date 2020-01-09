import math

from pharmpy.parameter import Parameter, ParameterSet

from .record import Record


class OmegaRecord(Record):
    def parameters(self, start_omega):
        """Get a ParameterSet for this omega record
        """
        scalar_args, nreps, stdevs = [], [], []
        row, col = 0, 0
        corr_coords = []
        block = self.root.find('block')
        fixed = bool(self.root.find('FIX'))
        sd_matrix = bool(self.root.find('SD'))
        corr_matrix = bool(self.root.find('CORR'))
        for i, node in enumerate(self.root.all('omega')):
            init = node.init.NUMERIC
            if sd_matrix or node.find('SD'):
                init = math.pow(init, 2)
            fix = fixed or bool(node.find('FIX'))
            if corr_matrix or node.find('CORR'):
                corr_coords += [(i, row, col)]
            nreps += [node.n.INT if node.find('n') else 1]
            if row != col or fix:
                lower = None
            else:
                lower = 0
            scalar_args.append({'name': f'OMEGA({row + start_omega},{col + start_omega})',
                                'init': init, 'fix': fix, 'lower': lower})
            if row == col:
                stdevs += [init]
                row += 1
                col = 0 if block else row
            else:
                col += 1

        for i, row, col in corr_coords:
            scalar_args[i]['init'] = scalar_args[i]['init'] * stdevs[row] * stdevs[col]

        params = [Parameter(**a) for N, args in zip(nreps, scalar_args) for a in [args]*N]
        return(ParameterSet(params))

    def random_variables(self, start_omega):
        """Get a RandomVariableSet for this omega record
        """
        pass
