class ModelExecutionEngine:
    """Base class for execution engines (e.g. NONMEM7 or nlmixr) of a :class:`~pharmpy.Model`.

    Each model object has an instance attached to the engine attribute.
    """

    def commandline(self, model):
        """Returns a command line for executing a model."""
        raise NotImplementedError

    def generate_files(self, model, path):
        """Generate all files needed to execute model in path

        return a list of paths to all files
        """
        raise NotImplementedError
