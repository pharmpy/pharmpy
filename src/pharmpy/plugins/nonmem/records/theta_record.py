import re
import warnings

from pharmpy.parameter import Parameter, ParameterSet
from pharmpy.parse_utils.generic import AttrToken, remove_token_and_space

from .record import Record

max_upper_bound = 1000000
min_lower_bound = -1000000


class ThetaRecord(Record):
    def __init__(self, content, parser_class):
        super().__init__(content, parser_class)
        self.name_map = None

    def add_nonmem_name(self, name_original, theta_number):
        self.root.add_comment_node(name_original)
        self.root.add_newline_node()
        self.name_map = {name_original: theta_number}

    def parameters(self, first_theta, seen_labels=None):
        """Get a parameter set for this theta record.
        first_theta is the number of the first theta in this record
        """
        if seen_labels is None:
            seen_labels = set()
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
            else:
                lower = min_lower_bound
            if theta.find('up'):
                if theta.up.find('POS_INF'):
                    upper = max_upper_bound
                else:
                    upper = theta.up.tokens[0].eval
            else:
                upper = max_upper_bound
            multiple = theta.find('n')
            if multiple:
                n = multiple.INT
            else:
                n = 1
            for i in range(0, n):
                name = None
                import pharmpy.plugins.nonmem as nonmem

                if nonmem.conf.parameter_names == 'comment':
                    # needed to avoid circular import with Python 3.6
                    found = False
                    for subnode in self.root.tree_walk():
                        if id(subnode) == id(theta):
                            if found:
                                break
                            else:
                                found = True
                                continue
                        if found and subnode.rule == 'NEWLINE':
                            m = re.search(r';\s*([a-zA-Z_]\w*)', str(subnode))
                            if m:
                                name = m.group(1)
                                break
                    if name in seen_labels:
                        warnings.warn(
                            f'The parameter name {name} is duplicated. Falling back to basic '
                            f'NONMEM names.'
                        )
                        name = None

                if not name:
                    name = f'THETA({current_theta})'
                seen_labels.add(name)
                new_par = Parameter(name, init, lower, upper, fix)
                current_theta += 1
                pset.add(new_par)

        if not self.name_map:
            self.name_map = {name: first_theta + i for i, name in enumerate(pset.names)}
        return pset

    def _multiple(self, theta):
        """Return the multiple (xn) of a theta or 1 if no multiple"""
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
            name = {v: k for k, v in self.name_map.items()}[i]
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

    def renumber(self, new_start):
        old_start = min(self.name_map.values())
        if new_start != old_start:
            for name in self.name_map:
                self.name_map[name] += new_start - old_start

    def remove(self, names):
        first_theta = min(self.name_map.values())
        indices = {self.name_map[name] - first_theta for name in names}
        for name in names:
            del self.name_map[name]
        keep = []
        i = 0
        for node in self.root.children:
            if node.rule == 'theta':
                if i not in indices:
                    keep.append(node)
                i += 1
            else:
                keep.append(node)
        self.root.children = keep

    def __len__(self):
        """Number of thetas in this record"""
        tot = 0
        for theta in self.root.all('theta'):
            tot += self._multiple(theta)
        return tot
