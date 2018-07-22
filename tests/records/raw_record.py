import sys
sys.path.append("..")
import unittest

from pysn.records.raw_record import RawRecord
from pysn.records.record import create_record

class TestFunctions(unittest.TestCase):
    def test_get_raw_record_name(self):
        obj = create_record("$PROB  MYPROB")
        self.assertEqual(obj.__class__.__name__, "RawRecord")
        self.assertEqual(obj.name, "PROBLEM")
        self.assertEqual(obj.raw_name, "PROB")
        self.assertEqual(obj.content, "  MYPROB")


if __name__ == '__main__':
    unittest.main()
