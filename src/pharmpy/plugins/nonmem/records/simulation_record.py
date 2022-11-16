"""
NONMEM simulation record class.
"""

from typing import cast

from pharmpy.internals.parse.generic import eval_token

from .record import Record


class SimulationRecord(Record):
    @property
    def nsubs(self):
        """Number of subproblems"""
        n = cast(int, eval_token(self.root.subtree('nsubs').leaf('INT')))
        # NOTE According to NONMEM documentation 0 means 1 here
        return 1 if n == 0 else n
