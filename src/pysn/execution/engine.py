# -*- encoding: utf-8 -*-


class Engine:
    """An execution engine (e.g. NONMEM or similar)."""
    def __init__(self, *args, **kwargs):
        """Setups the engine. Perhaps through input from configuration file."""
        raise NotImplementedError

    def create_command(self, *args, **kwargs):
        """Creates the command line to start execution."""
        raise NotImplementedError

    @property
    def bin(self):
        """Path to main binary."""
        raise NotImplementedError

    @property
    def version(self):
        """Version (of main binary)."""
        raise NotImplementedError

    def __bool__(self):
        """Should only eval True if engine is capable of estimation at any time."""
        return False
