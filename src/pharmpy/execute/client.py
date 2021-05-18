import datetime

import pandas as pd


class Client:
    def __init__(self):
        self._log = []  # list of tuples of timestamp, category and message

    def log_error(self, message):
        self._log.append((datetime.datetime.now(), 'ERROR', message))

    def log_warning(self, message):
        self._log.append((datetime.datetime.now(), 'WARNING', message))

    def log_as_dataframe(self):
        times = [entry[0] for entry in self._log]
        categories = [entry[1] for entry in self._log]
        messages = [entry[2] for entry in self._log]
        df = pd.DataFrame({'time': times, 'category': categories, 'message': messages})
        return df
