"""
NONMEM simulation record class.
"""


from .record import Record


class SimulationRecord(Record):
    @property
    def nsubs(self):
        """Number of subproblems"""
        n = self.root.nsubs.INT
        if n == 0:
            # According to NONMEM documentation 0 means 1 here
            n = 1
        return n
