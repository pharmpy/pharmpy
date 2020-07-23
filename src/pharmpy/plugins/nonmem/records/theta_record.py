# from pharmpy.model import ModelSyntaxError
from pharmpy.parameter import Parameter, ParameterSet
from pharmpy.parse_utils.generic import AttrToken, remove_token_and_space

from .record import Record

max_upper_bound = 1000000
min_lower_bound = -1000000


class ThetaRecord(Record):
    def parameters(self, first_theta):
        """Get a parameter set for this theta record.
        first_theta is the number of the first theta in this record
        """
        pset = ParameterSet()
        current_theta = first_theta
        for theta in self.root.all('theta'):
            init = theta.init.tokens[0].eval
            fix = bool(theta.find('FIX'))
            if theta.find('low'):
                if theta.low.find('NEG_INF'):
                    lower = min_lower_bound
                else:
                    lower = theta.low.tokens[0].eval
                # FIXME: Ideally the following should be enforced
                # However PsN allows it and it is used to toggle FIX back
                # and forth without having to change or remove lower and upper bound
                # if fix and not init == lower:
                #    raise ModelSyntaxError(f'Fixed thetas must have equal lower limit and initial '
                #                           f'estimate: {theta}')
            else:
                lower = min_lower_bound
            if theta.find('up'):
                if theta.up.find('POS_INF'):
                    upper = max_upper_bound
                else:
                    upper = theta.up.tokens[0].eval
                # if fix and not init == upper:
                #    raise ModelSyntaxError(f'Fixed thetas must have equal upper limit and initial '
                #                           f'estimate: {theta}')
            else:
                upper = max_upper_bound
            multiple = theta.find('n')
            if multiple:
                n = multiple.INT
            else:
                n = 1
            for i in range(0, n):
                new_par = Parameter(f'THETA({current_theta})', init, lower, upper, fix)
                current_theta += 1
                pset.add(new_par)
        return pset

    def _multiple(self, theta):
        """Return the multiple (xn) of a theta or 1 if no multiple
        """
        multiple = theta.find('n')
        if multiple:
            n = multiple.INT
        else:
            n = 1
        return n

    def update(self, parameters, first_theta):
        """From a ParameterSet update the THETAs in this record

        Currently only updating initial estimates
        """
        i = first_theta
        for theta in self.root.all('theta'):
            name = f'THETA({i})'
            param = parameters[name]
            new_init = param.init
            if float(str(theta.init)) != new_init:
                theta.init.tokens[0].value = str(new_init)
            fix = bool(theta.find('FIX'))
            if fix != param.fix:
                if param.fix:
                    space = AttrToken('WS', ' ')
                    fix_token = AttrToken('FIX', 'FIX')
                    theta.children.extend([space, fix_token])
                else:
                    remove_token_and_space(theta, 'FIX')

            n = self._multiple(theta)
            i += n

    def __len__(self):
        """Number of thetas in this record
        """
        tot = 0
        for theta in self.root.all('theta'):
            tot += self._multiple(theta)
        return tot

    # @thetas.setter
    # def thetas(self, thetas):
    #    nodes = []
    #    nodes_new = self._new_theta_nodes(thetas)
    #    for child in self.root.children:
    #        if child.rule != 'theta':
    #            nodes += [child]
    #            continue
    #        try:
    #            nodes += [nodes_new.pop(0)]
    #        except IndexError:
    #            pass
    #    self.root = AttrTree.create('root', nodes + nodes_new)

    # def _new_theta_nodes(self, thetas):
    #    nodes = []
    #    for theta in thetas:
    #        if nodes:
    #            nodes += [dict(WS='\n  ')]
    #        new = [{'LPAR': '('}]
    #        if theta.lower is not None:
    #            new += [{'low': {'NUMERIC': theta.lower}}]
    #            new += [{'WS': ' '}]
    #        if theta.init is not None:
    #            new += [{'init': {'NUMERIC': theta.init}}]
    #            new += [{'WS': ' '}]
    #        if theta.upper is not None:
    #            new += [{'up': {'NUMERIC': theta.upper}}]
    #        if theta.fix:
    #            new += [{'WS': ' '}]
    #            new += [{'FIX': 'FIXED'}]
    #        new += [{'RPAR': ')'}]
    #        nodes += [AttrTree.create('theta', new)]
    #    return nodes
