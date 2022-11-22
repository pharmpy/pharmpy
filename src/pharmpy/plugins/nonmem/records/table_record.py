import re
from dataclasses import dataclass
from pathlib import Path

from .option_record import OptionRecord


@dataclass(frozen=True)
class TableRecord(OptionRecord):
    @property
    def path(self):
        file_option = self.option_pairs['FILE']
        assert file_option is not None
        return Path(file_option)

    def with_path(self, value):
        return self.set_option('FILE', str(value))

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
        for option in self.all_options:
            m = re.match(regexp, option.key)
            if not m and option.value is not None:
                m = re.match(regexp, option.value)
            if m:
                n = m.group(1)
                derivs.append(int(n))
        return derivs
