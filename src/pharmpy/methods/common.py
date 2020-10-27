from pathlib import Path


class RunDirectory:
    def __init__(self, method_name):
        i = 1
        while True:
            name = f'{method_name}_dir{i}'
            path = Path(name)
            if not path.is_file():
                path.mkdir()
                self.path = path
                break
            i += 1


class Method:
    def __init__(self):
        self.rundir = RunDirectory(type(self).__name__.lower())
