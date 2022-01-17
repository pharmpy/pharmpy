class ModelDatabase:
    """Database of results for particular model runs

    This corresponds to the m1 directory and copying up files in PsN
    """

    def store_local_file(self, model, path):
        """Store a file from the local machine"""
        raise NotImplementedError()
