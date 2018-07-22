import sys
sys.path.append("..")
import unittest

from records.option_description import OptionDescription, OptionDescriptionList, OptionType  

class TestFunctions(unittest.TestCase):
    def test_get_raw_record_name(self):
        od = OptionDescription({'name' : 'MAXEVALS', 'type' : OptionType.VALUE, 'abbreviate' : True })
        self.assertTrue(od.match("MAX"))

if __name__ == '__main__':
    unittest.main()
