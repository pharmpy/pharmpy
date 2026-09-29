"""
NONMEM $SIMULATION record class.
"""
from .option_record import OptionRecord

# NONMEM synonym sets, as verified against real NONMEM (see pharmpy/pharmpy#4823).
# Not simple prefix-truncations of one canonical word (NSUB vs SUBPROBLEMS),
# so listed explicitly rather than resolved via automatic abbreviation matching.
_SUBPROBLEMS_KEYS = frozenset(
    {'SUBPROBLEMS', 'SUBPROBS', 'SUBPROB', 'NSUBPROBLEMS', 'NSUBPROBS', 'NSUB'}
)

def _matches(key: str, keys: frozenset) -> bool:
    return key.upper() in keys


class SimulationRecord(OptionRecord):
    @property
    def nsubs(self) -> int:
        """Number of subproblems. NONMEM treats 0 and 'not given' as 1."""
        for opt in self.all_options:
            if _matches(opt.key, _SUBPROBLEMS_KEYS) and opt.value is not None:
                n = int(opt.value)
                return 1 if n == 0 else n
        return 1
