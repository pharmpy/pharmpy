import re
from pathlib import Path

from .option_record import OptionRecord


class TableRecord(OptionRecord):
    @property
    def path(self):
        file_option = self.option_pairs['FILE']
        return Path(file_option)

    @path.setter
    def path(self, value):
        self.set_option('FILE', value)

    @property
    def eta_derivatives(self):
        """List of numbers for etas whose derivative are requested"""
        return self._find_derivatives('G')

    @property
    def epsilon_derivatives(self):
        """List of numbers for epsilons whose derivateives are requested"""
        return self._find_derivatives('H')

    def _find_derivatives(self, ch):
        derivs = []
        regexp = ch + r'(\d+)1$'
        for key, value in self.all_options:
            m = re.match(regexp, key)
            if not m and value is not None:
                m = re.match(regexp, value)
            if m:
                n = m.group(1)
                derivs.append(int(n))
        return derivs
