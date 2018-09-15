# -*- encoding: utf-8 -*-


class ParameterList(list):
    def __str__(self):
        out = ['    %s (container of %d)' % (self.__class__.__name__, len(self))]
        for i, param in enumerate(self):
            prefix = '[%d]' % (i,)
            lines = str(param).splitlines()
            out += ['%s %s' % (prefix, lines[0])]
            out += ['%s %s' % (' '*len(prefix), line) for line in lines[1:]]
        return '\n'.join(out)
