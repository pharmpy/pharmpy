import datetime

from pharmpy.deps import pandas as pd
from pharmpy.internals.immutable import Immutable

CATEGORIES = ('ERROR', 'WARNING', 'INFORMATION')


class LogEntry(Immutable):
    def __init__(self, category: str, message: str, time: datetime.datetime):
        self._category = category
        self._message = message
        self._time = time

    @classmethod
    def create(cls, category: str, message: str):
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category {category}")
        return cls(category=category, message=message, time=datetime.datetime.now())

    @property
    def category(self):
        """Category of message. One of ERROR, WARNING and INFORMATION"""
        return self._category

    @property
    def message(self):
        """The logged message"""
        return self._message

    @property
    def time(self):
        """Time stamp of log entry
        """
        return self._time

    def to_dict(self):
        return {
            'category': self._category,
            'message': self._message,
            'time': self._time.isoformat(),
        }

    @classmethod
    def from_dict(cls, d):
        d['time'] = datetime.datetime.fromisoformat(d['time'])
        return LogEntry(**d)


class Log:
    """Timestamped error and warning log"""

    def __init__(self):
        self._log = []  # list of LogEntry

    def to_dict(self):
        return {i: entry.to_dict() for i, entry in enumerate(self._log)}

    @classmethod
    def from_dict(cls, d):
        log = cls()
        for entry in d.values():
            log_entry = LogEntry.from_dict(entry)
            log._log.append(log_entry)
        return log

    @property
    def log(self):
        return self._log

    def log_error(self, message):
        """Log an error

        Parameters
        ----------
        message : str
            Error message
        """
        entry = LogEntry.create(category='ERROR', message=message)
        self._log.append(entry)

    def log_warning(self, message):
        """Log a warning

        Parameters
        ----------
        message : str
            Warning message
        """
        entry = LogEntry.create(category='WARNING', message=message)
        self._log.append(entry)

    def to_dataframe(self):
        """Create an overview dataframe from log

        Returns
        -------
        pd.DataFrame
            Dataframe with overview of log entries
        """
        categories = [entry.category for entry in self._log]
        times = [entry.time for entry in self._log]
        messages = [entry.message for entry in self._log]
        df = pd.DataFrame(
            {
                'category': categories,
                'time': times,
                'message': messages,
            }
        )
        return df

    def __repr__(self):
        s = ''
        for entry in self._log:
            s += f'{entry.category:<8} {entry.time.strftime("%Y-%m-%d %H:%M:%S")}\n'
            for line in entry.message.split('\n'):
                s += f'    {line}\n'
        return s
