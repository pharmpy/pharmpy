class Environment:
    """ A base class for execution environments that manages the execution of the execution engine
        examples are Windows, Linux, SLURM, SGE etc
    """

    def submit_job(self, command):
        """ Method to start a job in the environment
            Return an object of class Job
        """
        raise NotImplementedError

    def argparse_options(self):
        """ Method to return command line options for argparse
            for this specific environment
        """
        raise NotImplementedError
