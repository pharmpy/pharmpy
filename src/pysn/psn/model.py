"""Generic model API"""
from .model_apis import api_list

class Model:
    """A generic model of any type

    Will keep track of the filename, native model class and the current api.
    """
    def __init__(self, filename=None):
        """Detects API and loads model if filename given"""
        self.filename = filename
        self._api = None
        if self.filename:
            self.init_model()

    @property
    def type(self):
        """Filetype (API name)"""
        return self._api.module.name if self._api else None

    @type.setter
    def type(self, name):
        self._api = None
        if name:
            self._api = api_list.get(name)

    @property
    def api(self):
        """API (tuple) supporting current filetype"""
        return self._api

    def init_model(self):
        """Detect filetype, and read model with API"""
        self._api = api_list.filedetect(self.filename)
        self.model = self._api.module.Model(self.filename)
        self.input = self.model.input

    def __str__(self):
        return str(self.model)
