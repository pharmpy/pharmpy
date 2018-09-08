# -*- encoding: utf-8 -*-


class Scalar:
    __slots__ = ('init', 'fix', 'lower', 'upper')

    def __new__(cls, value=0, fix=False, lower=float('-INF'), upper=float('INF')):
        self = super(Scalar, cls).__new__(cls)
        try:
            self.init = value.init
            self.fix = value.fix
            self.lower = value.lower
            self.upper = value.upper
        except AttributeError:
            self.init = float(value)
            self.fix = bool(fix) if fix is not None else None
            self.lower = float(lower)
            self.upper = float(upper)
        return self

    def __str__(self):
        if self.fix is None:
            return '<val %.4G>' % (self.init,)
        elif self.fix:
            return '<fix %.4G>' % (self.init,)
        value = '%.4G' % (self.init,)
        if self.lower != float('-INF'):
            value = '%.0G<%s' % (self.lower, value)
        if self.upper != float('INF'):
            value = '%s<%0.G' % (value, self.upper)
        return '<est %s>' % (value,)

    def __repr__(self):
        args = [repr(self.init)]
        if self.fix is not None:
            args += ['fix=%s' % repr(self.fix)]
        if self.lower != float('-INF'):
            args += ['lower=%s' % repr(self.lower)]
        if self.upper != float('INF'):
            args += ['upper=%s' % repr(self.upper)]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(args))

    def __float__(self):
        return float(self.init)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return all(getattr(self, a) == getattr(other, a) for a in self.__slots__)
        else:
            return False


class Var(Scalar):
    pass


class Covar(Scalar):
    pass
