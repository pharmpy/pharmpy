from dataclasses import dataclass
from typing import Literal, Union

from lark.visitors import Interpreter

from .feature import ModelFeature, feature
from .symbols import Name, Wildcard

METABOLITE_WILDCARD = tuple([Name(x) for x in ('PSC', 'BASIC')])


@dataclass(frozen=True)
class Metabolite(ModelFeature):
    modes: Union[tuple[Name[Literal['PSC', 'BASIC']], ...], Wildcard]

    def __add__(self, other):
        if isinstance(self.modes, Wildcard) or isinstance(other.modes, Wildcard):
            return Metabolite(Wildcard())
        else:
            return Metabolite(
                tuple(set(self.modes + tuple([a for a in other.modes if a not in self.modes])))
            )

    def __sub__(self, other):
        if isinstance(other.modes, Wildcard):
            return None
        elif isinstance(self.modes, Wildcard):
            default = METABOLITE_WILDCARD
            all_modes = tuple([a for a in default if a not in other.modes])
        else:
            # NOTE : WILDCARD should not be used here to future proof the method
            all_modes = tuple(set([a for a in self.modes if a not in other.modes]))

        if len(all_modes) == 0:
            all_modes = None

        return Metabolite(all_modes)

    def __eq__(self, other):
        if isinstance(other, Metabolite):
            return set(self.modes) == set(other.modes)
        else:
            return False

    def __len__(self):
        return len(self.eval.modes)

    @property
    def eval(self):
        if isinstance(self.modes, Wildcard):
            return Metabolite(METABOLITE_WILDCARD)
        else:
            return self


class MetaboliteInterpreter(Interpreter):
    def interpret(self, tree):
        children = self.visit_children(tree)
        assert len(children) == 1
        return feature(Metabolite, children)

    def metabolite_modes(self, tree):
        children = self.visit_children(tree)
        return list(Name(child.value.upper()) for child in children)

    def metabolite_wildcard(self, tree):
        return Wildcard()
