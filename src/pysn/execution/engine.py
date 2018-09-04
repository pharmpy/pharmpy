class Engine:
    """ Base class for an execution engine
        An example of an engine would be NONMEM
    """
    def __init__(self, options):
        """ Setup of the engine. Perhaps through input from configuration file.
        """
        raise NotImplementedError

    def create_command(self, options_goes_here):
        """ Create the command line to start execution
        """
        raise NotImplementedError
