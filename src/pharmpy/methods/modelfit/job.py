class ModelfitJob:
    def __init__(self, workflow):
        self.workflow = workflow
        self.infiles = []
        self.outfiles = []

    def add_infiles(self, source, destination='.'):
        self.infiles.append((source, destination))

    def add_outfiles(self, outfiles):
        self.outfiles.append(outfiles)
