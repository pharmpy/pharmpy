# The NONMEM Model class
import re
from pathlib import Path

import pharmpy.model
from .nmtran_parser import NMTranParser
from pharmpy.parameter import ParameterSet
import pharmpy.plugins.nonmem.input


class Model(pharmpy.model.Model):
    def __init__(self, src, **kwargs):
        parser = NMTranParser()
        self.source = src
        if not self.source.filename_extension:
            self.source.filename_extension = '.ctl'
        self.control_stream = parser.parse(src.code)
        self.input = pharmpy.plugins.nonmem.input.ModelInput(self)

    @staticmethod
    def detect(src, *args, **kwargs):
        """ Check if src represents a NONMEM control stream
        i.e. check if it is a file that contain $PRO
        """
        return bool(re.search(r'^\$PRO', src.code, re.MULTILINE))

    def update_source(self):
        """Update the source"""
        if self.input._dataset_updated:
            datapath = self.input.dataset.pharmpy.write_csv()      # FIXME: If no name set use the model name. Set that when setting dataset to input!
            self.input.path = datapath
            # FIXME: ignore_character et al should be set when setting the dataset. Check if A-Za-z and use @, # remove else use first character
            # FIXME: how to handle IGNORE, ACCEPT? Must be performed when resampling as entire groups might disappear collapsing individuals.
            #           so resampling should be done on parsed dataset. Anonymizing 
        super().update_source()

    def validate(self):
        """Validates NONMEM model (records) syntactically."""
        self.control_stream.validate()

    @property
    def parameters(self):
        """Get the ParameterSet of all parameters
        """
        next_theta = 1
        params = ParameterSet()
        for theta_record in self.control_stream.get_records('THETA'):
            thetas = theta_record.parameters(next_theta)
            params.update(thetas)
            next_theta += len(thetas)
        return params

    def __str__(self):
        return str(self.control_stream)
