import sys
sys.path.append("..")
import unittest

from records.record import create_record
from records.option_record import SizesRecord

class TestFunctions(unittest.TestCase):
    def test_lexical_tokens(self):
        obj = create_record("$SIZES LTH=28")
        self.assertEqual(list(obj.lexical_tokens()), ["LTH=28"])


if __name__ == '__main__':
    unittest.main()
