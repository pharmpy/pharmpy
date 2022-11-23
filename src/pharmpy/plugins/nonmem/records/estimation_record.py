"""
The NONMEM $ESTIMATION record
"""

from dataclasses import dataclass
from functools import cached_property

from .option_record import OptionRecord
from .record import with_parsed_and_generated


@with_parsed_and_generated
@dataclass(frozen=True)
class EstimationRecord(OptionRecord):
    @cached_property
    def likelihood(self):
        like = self.get_option_startswith('LIKE')
        return bool(like)

    @cached_property
    def loglikelihood(self):
        ll = self.get_option_startswith('-2LL')
        if not ll:
            ll = self.get_option_startswith('-2LOG')
        return bool(ll)
