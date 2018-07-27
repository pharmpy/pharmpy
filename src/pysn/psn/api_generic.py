from pathlib import Path



def detect(lines):
    return False


class ModelException(Exception):
    pass


class ModelParsingError(ModelException):
    pass


class Model(object):
    """Generic model of any type

    Agnostic of model type. Subclass me to implement non-agnostic API."""
    def __init__(self, path, **kwargs):
        self.path = Path(path).resolve() if path else None
        if self.exists:
            self.load()

    @property
    def exists(self):
        if self.path and self.path.is_file():
            return True

    @property
    def content(self):
        if not self.exists:
            return None
        with open(self.path, 'r') as f:
            content = f.read()
        return content

    def write(self, *args, **kwargs):
        pass

    def validate(self):
        raise NotImplementedError

    def load(self):
        self.input = ModelInput(model)

    def __str__(self):
        if self.exists:
            return self.read

class ModelInput(object):
    def __init__(self, model):
        self.model = model

    @property
    def path(self):
        """Gets the path of the dataset"""
        raise NotImplementedError
