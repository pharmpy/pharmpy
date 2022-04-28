"""
The NONMEM $ESTIMATION record
"""

from .option_record import OptionRecord


class EstimationRecord(OptionRecord):
    @property
    def likelihood(self):
        like = self.get_option_startswith('LIKE')
        return bool(like)

    @property
    def loglikelihood(self):
        ll = self.get_option_startswith('-2LL')
        if not ll:
            ll = self.get_option_startswith('-2LOG')
        return bool(ll)
